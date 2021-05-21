#coding=utf-8
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F

from model.utils import log_normal_density

import mpi4py.MPI as MPI
#from self_attention import SelfAttention
import heapq

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

class Flatten(nn.Module):
    def forward(self, input):

        return input.view(input.shape[0], 1,  -1)


class CNNPolicy(nn.Module):
    def __init__(self, frames, action_space):
        super(CNNPolicy, self).__init__()
        self.logstd = nn.Parameter(torch.zeros(action_space))

        self.act_fea_cv1 = nn.Conv1d(in_channels=frames, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.act_fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)  
        self.act_fc1 = nn.Linear(128*32, 512)
        self.act_fc2 =  nn.Linear(512, 128)
        self.act_encoder = nn.Sequential()
        self.act_encoder.add_module('act_bn',nn.LayerNorm(normalized_shape = [128+3+2+2]))
        self.act_encoder.add_module('act_fc',nn.Linear(128+3+2+2,128))
        self.act_encoder.add_module('act_nl', nn.LeakyReLU())
        self.act_combine1 = nn.Linear(256,128)  # Dim of last layer's output is 128; Dim of this layer's input is 256. The rest 128 is other robots' effect to the current robot
        self.act_combine2 = nn.Linear(128, 32)
        self.actor1 = nn.Linear(32, 1)
        self.actor2 = nn.Linear(32, 1)

        # self.encoder = nn.Sequential(
        #     nn.Linear(128, )
        # )


        self.crt_fea_cv1 = nn.Conv1d(in_channels=frames, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.crt_fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.crt_fc1 = nn.Linear(128*32, 512)
        self.crt_fc2 = nn.Linear(512, 128)
        #self.crt_fc3 = nn.Linear(128+3+2+2, 128)
        self.crt_encoder = nn.Sequential()
        self.crt_encoder.add_module('crt_bn',nn.LayerNorm(normalized_shape = [128+3+2+2]))
        self.crt_encoder.add_module('crt_fc',nn.Linear(128+3+2+2,128))
        self.crt_encoder.add_module('crt_nl', nn.LeakyReLU())
        self.crt_combine1 = nn.Linear(256, 128)
        self.crt_combine2 = nn.Linear(128, 32)
        self.critic = nn.Linear(32, 1)

        # 参数定义
        attend_heads = 4
        hidden_dim = 128
        attend_dim = hidden_dim // attend_heads
        dropout_prob = 0.1
        self.hidden_dim = hidden_dim
        self.attend_heads = attend_heads
        # self.num_attention_heads = num_attention_heads  # 8
        # self.attention_head_size = int(hidden_size / num_attention_heads)  # 16  每个注意力头的维度
        # self.all_head_size = int(self.num_attention_heads * self.attention_head_size)
        # all_head_size = 128 即等于hidden_size, 一般自注意力输入输出前后维度不变

        # query, key, value 的线性变换（上述公式2）self_attention
        # self.query = nn.Linear(hidden_size, self.all_head_size,bias=False)  # 128, 128 用单层网络代替矩阵
        # self.key = nn.Linear(hidden_size, self.all_head_size,bias=False)
        # self.value = nn.Sequential(nn.Linear(hidden_size, self.all_head_size),  nn.LeakyReLU())
        # # self.query_position = nn.Linear(position_len, 16,bias=False)
        # self.key_position = nn.Linear(position_len, 16,bias=False)
        #self.value_position = nn.Linear(position_len, 16)
        self.key_extractors = nn.ModuleList()
        self.selector_extractors = nn.ModuleList()
        self.value_extractors = nn.ModuleList()
        for i in range(attend_heads):
            self.key_extractors.append(nn.Linear(hidden_dim,attend_dim,bias=False))
            self.selector_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            self.value_extractors.append(nn.Sequential(nn.Linear(hidden_dim, attend_dim), nn.LeakyReLU()))

        self.critic_key_extractors = nn.ModuleList()
        self.critic_selector_extractors = nn.ModuleList()
        self.critic_value_extractors = nn.ModuleList()
        for i in range(attend_heads):
            self.critic_key_extractors.append(nn.Linear(hidden_dim,attend_dim,bias=False))
            self.critic_selector_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            self.critic_value_extractors.append(nn.Sequential(nn.Linear(hidden_dim, attend_dim), nn.LeakyReLU()))

        # dropout
        self.dropout = nn.Dropout(dropout_prob)




    def forward(self, x, goal, speed, position,adj):
        """
            returns value estimation, action, log_action_prob
        """
        # action
        #print x.size()
        x_is_4 = False
        if len(x.shape)==4:    #[batch_size, number_of_robot, input_channel, input_dimension] -> [batch_size * number_of_robot,input_channel, input_dimension] for nn.Conv1d input
            origin_size = x.detach().shape
            x = x.view(-1,x.shape[-2],x.shape[-1])
            x_is_4 = True
        a = F.leaky_relu(self.act_fea_cv1(x)) # size:[batch_size or not, n,32,255]  p.s. n is the number of robot swarm
        a = F.leaky_relu(self.act_fea_cv2(a)) # size:[n,32,128]
        if x_is_4:   #[batch_size * number_of_robot, output_channel, output_dimension] ->[batch_size, number_of_robot, output_channel, output_dimension] ->[batch_size, number_of_robot, dimension]
            a = a.view(origin_size[0],origin_size[1],a.shape[1],a.shape[2])
            a = a.view(a.shape[0],a.shape[1], -1)
        else:
            a = a.view(a.shape[0], -1)  # size:[n,4096]  [number_of_robot, output_channel, output_dimension] ->[number_of_robot, dimension]

        a = F.leaky_relu(self.act_fc1(a)) # size:[n,256]
        #a = torch.cat((a, goal, speed), dim=-1) # size:[n,260]
        a = F.leaky_relu(self.act_fc2(a))     # size:[n,128] # actor intention
        a = torch.cat((a,position, goal, speed),-1)   # @HAVEDONE: change to a = torch.cat((a,position, goal, speed),-1), put off goal and speed in the line 113
        a = self.act_encoder(a) # [(batch_size), number_robots,dim]
        # if len(a.shape)==2:
            #adj_lists = self.get_adjacency_list(position.detach())
        b,all_attend_probs = self.limited_self_attention(a,key_extractors=self.key_extractors,
                                        value_extractors=self.value_extractors ,
                                        selector_extractors=self.selector_extractors,
                                        attend_heads=self.attend_heads,hidden_dim=self.hidden_dim,adj_list=adj)
        # else:
        #     b,_  = self.self_attention(a, key_extractors=self.key_extractors,
        #                                     value_extractors=self.value_extractors,
        #                                     selector_extractors=self.selector_extractors,
        #                                     attend_heads=self.attend_heads, hidden_dim=self.hidden_dim)

        #print b.size()
        a = torch.cat((a,b),-1)
        #a = torch.cat((a, goal, speed), dim=-1)
        a = F.leaky_relu(self.act_combine1(a))
        a = F.leaky_relu(self.act_combine2(a))

        mean1 = torch.sigmoid(self.actor1(a))
        mean2 = torch.tanh(self.actor2(a))  # size:[n,1]
        mean = torch.cat((mean1, mean2), dim=-1)

        logstd = self.logstd.expand_as(mean) # size:[1,2]
        std = torch.exp(logstd)
        action = torch.normal(mean, std)

        # action prob on log scale
        logprob = log_normal_density(action, mean, std=std, log_std=logstd)


        v = F.leaky_relu(self.crt_fea_cv1(x))
        v = F.leaky_relu(self.crt_fea_cv2(v))
        if x_is_4:
            v = v.view(origin_size[0],origin_size[1],v.shape[1],v.shape[2])
            v = v.view(v.shape[0],v.shape[1], -1)
        else:
            v = v.view(v.shape[0], -1)

        v = F.leaky_relu(self.crt_fc1(v))
        v = F.leaky_relu(self.crt_fc2(v))
        v = torch.cat((v,position, goal, speed),-1)
        v = self.crt_encoder(v)
        w,_ = self.self_attention(v,key_extractors=self.critic_key_extractors,value_extractors=self.critic_value_extractors ,selector_extractors=self.critic_selector_extractors,attend_heads=self.attend_heads,hidden_dim=self.hidden_dim)
        # w, _ = self.limited_self_attention(v, key_extractors=self.key_extractors,
        #                                                   value_extractors=self.value_extractors,
        #                                                   selector_extractors=self.selector_extractors,
        #                                                   attend_heads=self.attend_heads, hidden_dim=self.hidden_dim,
        #                                                   adj_list=adj)
        v = torch.cat((v,w),-1)
        #v = torch.cat((v, goal, speed), dim=-1)
        v = F.leaky_relu(self.crt_combine1(v))
        v = F.leaky_relu(self.crt_combine2(v))
        #v = F.leaky_relu(self.crt_combine3(v))
        v = self.critic(v)
        if len(v.shape)==2:
            return v, action, logprob, mean ,all_attend_probs
        else:
            return v, action, logprob, mean



    def evaluate_actions(self, x, goal, speed, position,adj, action,this_filter_index=[]):
        v, _, _, mean = self.forward(x, goal, speed, position,adj)
        # print(this_filter_index)
        # print(action)
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        # evaluate
        logprob = log_normal_density(action, mean, log_std=logstd, std=std)
        dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + logstd
        for this_filter in this_filter_index:
            dist_entropy[this_filter[0], this_filter[1]] *= 0
        dist_entropy = dist_entropy.sum(-1).mean()
        return v, logprob, dist_entropy

    def transpose_for_scores(self, x):
        # INPUT:  x'shape = [seqlen, hid_size]  bs:batch size   假设hid_size=128
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # [bs, seqlen, 8, 16]
        x = x.view(*new_x_shape)  #
        return x.permute(0,2,1,3)  # [8, seqlen, 16]

    def self_attention(self,a,key_extractors,value_extractors,selector_extractors,attend_heads,hidden_dim):
        is_unsqueeze = False
        if len(a.shape) == 2:
            size = [1, a.shape[0], a.shape[1]]
            a = a.expand(*size)
            is_unsqueeze = True
        agents = range(len(a[0]))
        if len(a[0]) == 1:
            if is_unsqueeze==True:
                a = a.squeeze(0)
                return torch.zeros(a.shape).to('cuda'), [[]]
            else:
                return a,[[]]
        a = a.permute(1,0,2).contiguous()  #[batch_size, number_robots,dim] ->[number_robots,batch_size,dim]
        all_head_keys = [[k_ext(aa) for aa in a] for k_ext in key_extractors]
        all_head_values = [[k_ext(aa) for aa in a] for k_ext in value_extractors]
        all_head_selectors = [[k_ext(aa) for aa in a] for k_ext in selector_extractors]
        other_all_values = [[] for _ in range(len(agents))]
        all_attend_logits = [[] for _ in range(len(agents))]
        all_attend_probs = [[] for _ in range(len(agents))]
        #all_attend_probs = np.array(all_attend_probs)
        #print('sss',np.array(all_head_keys).shape)   [head_num,num_robot]


        for curr_head_keys, curr_head_values, curr_head_selectors in zip(
                all_head_keys, all_head_values, all_head_selectors):  # size of all_head_keys :[attend_heads,number_robots,dim]
            # iterate over agents
            for i, a_i, selector in zip(range(len(agents)), agents, curr_head_selectors):
                keys = [k for j, k in enumerate(curr_head_keys)]
                values = [v for j, v in enumerate(curr_head_values)]
                # calculate attention across agents
                attend_logits = torch.matmul(selector.view(selector.shape[0], 1, -1),
                                             torch.stack(keys).permute(1, 2, 0))
                # scale dot-products by size of key (from Attention is All You Need)
                scaled_attend_logits = attend_logits / np.sqrt(keys[0].shape[1])

                delete_self_matrix = torch.ones(len(agents)).to('cuda')
                for mmm in range(len(agents)):
                    if mmm ==a_i:
                        delete_self_matrix[mmm] = 0.0
                matrix_size = [1, 1, len(agents)]
                delete_self_matrix = delete_self_matrix.expand(*matrix_size)

                all_attend_weights = F.softmax(scaled_attend_logits, dim=2)
                attend_weights = all_attend_weights*delete_self_matrix

                #print(attend_weights)
                other_values = (torch.stack(values).permute(1, 2, 0) *
                                attend_weights).sum(dim=2)
                other_all_values[i].append(other_values)
                all_attend_logits[i].append(attend_logits)
                all_attend_probs[i].append(attend_weights.detach().to('cpu').numpy())
        for i in range(len(agents)):
            all_attend_probs[i] = np.array(all_attend_probs[i])
        #print(all_attend_probs[12]) # 12*4*12
        all_attend_probs = np.array(all_attend_probs)
        all_attend_probs = all_attend_probs.squeeze()
        all_attend_probs = np.sum(all_attend_probs,axis=1)/attend_heads
        #print(all_attend_probs.shape) #12*12
        # type(other_all_values):list of list of tensor:[num_robots,attend_heads,batch_size,attend_dim]
        other_all_values_To_Tensor = torch.stack(other_all_values[0]).to('cuda')
        for i in range(len(other_all_values)):
            if i>0:
                other_all_values_To_Tensor = torch.cat((other_all_values_To_Tensor,torch.stack(other_all_values[i])))
        other_all_values_To_Tensor = other_all_values_To_Tensor.view(-1, attend_heads,other_all_values_To_Tensor.shape[1],other_all_values_To_Tensor.shape[2])

        other_all_values_To_Tensor = other_all_values_To_Tensor.permute(2,0,1,3).contiguous()
        new_shape = other_all_values_To_Tensor.size()[:-2] + (hidden_dim,)
        b = other_all_values_To_Tensor.view(*new_shape)
        if is_unsqueeze == True:
            b = b.squeeze(0)
        return b,all_attend_probs.T

    def limited_self_attention(self,a,key_extractors,value_extractors,selector_extractors,attend_heads,hidden_dim,adj_list):
        #print('adj_list', adj_list)
        is_unsqueeze = False   # we need to add the batch_size dimension
        if len(a.shape) == 2:
            size_a = [1, a.shape[0], a.shape[1]]
            a = a.expand(*size_a)
            size_adj = [1, adj_list.shape[0], adj_list.shape[1]]
            adj_list = adj_list.expand(*size_adj)
            is_unsqueeze = True
        agents = range(len(a[0]))
        if len(a[0]) == 1:  # if just only one agent
            if is_unsqueeze==True:
                a = a.squeeze(0)
                li = [[1]]
                return torch.zeros(a.shape).to('cuda'),np.array(li)
            else:
                li = [[1]*len(a)]
                return torch.zeros(a.shape).to('cuda'), np.array(li)

        a = a.permute(1,0,2).contiguous()  #[batch_size, number_robots,dim] ->[number_robots,batch_size,dim]
        adj_list = adj_list.permute(1,0,2).contiguous() #[batch_size, number_robots,number_robots] ->[number_robots,batch_size,number_robots]
        all_head_keys = [[k_ext(aa) for aa in a] for k_ext in key_extractors]
        all_head_values = [[k_ext(aa) for aa in a] for k_ext in value_extractors]
        all_head_selectors = [[k_ext(aa) for aa in a] for k_ext in selector_extractors]
        other_all_values = [[] for _ in range(len(agents))]
        all_attend_logits = [[] for _ in range(len(agents))]
        all_attend_probs = [[] for _ in range(len(agents))]
        #all_attend_probs = np.array(all_attend_probs)
        #print('sss',np.array(all_head_keys).shape)   [head_num,num_robot]


        for curr_head_keys, curr_head_values, curr_head_selectors in zip(
                all_head_keys, all_head_values, all_head_selectors):  # size of all_head_keys :[attend_heads,number_robots,dim]
            # iterate over agents
            for i, a_i, selector,adj in zip(range(len(agents)), agents, curr_head_selectors,adj_list):
                keys = [k for j, k in enumerate(curr_head_keys)]
                values = [v for j, v in enumerate(curr_head_values)]
                # calculate attention across agents
                attend_logits = torch.matmul(selector.view(selector.shape[0], 1, -1),
                                             torch.stack(keys).permute(1, 2, 0))
                # scale dot-products by size of key (from Attention is All You Need)
                scaled_attend_logits = attend_logits / np.sqrt(keys[0].shape[1])

                # adj_matrix = torch.zeros(len(agents)).to('cuda')
                # for mmm in range(len(agents)):
                #     if mmm in adj:
                #         adj_matrix[mmm] = 1.0
                # matrix_size = [1, 1, len(agents)]
                # adj_matrix = adj_matrix.expand(*matrix_size)
                #print('adj_matrix', adj_matrix)
                all_attend_weights = F.softmax(scaled_attend_logits, dim=2)
                adj = adj.unsqueeze(1)
                attend_weights = all_attend_weights * adj
                other_values = (torch.stack(values).permute(1, 2, 0) *
                                attend_weights).sum(dim=2)
                other_all_values[i].append(other_values)
                all_attend_logits[i].append(attend_logits)
                all_attend_probs[i].append(attend_weights.detach().to('cpu').numpy())
        for i in range(len(agents)):
            all_attend_probs[i] = np.array(all_attend_probs[i])
        #print(all_attend_probs[12]) # 12*4*12
        all_attend_probs = np.array(all_attend_probs)
        all_attend_probs = all_attend_probs.squeeze()
        all_attend_probs = np.sum(all_attend_probs,axis=1)/attend_heads
        #print(all_attend_probs.T)
        #print(all_attend_probs.shape) #12*12
        # type(other_all_values):list of list of tensor:[num_robots,attend_heads,batch_size,attend_dim]
        other_all_values_To_Tensor = torch.stack(other_all_values[0]).to('cuda')
        for i in range(len(other_all_values)):
            if i>0:
                other_all_values_To_Tensor = torch.cat((other_all_values_To_Tensor,torch.stack(other_all_values[i])))
        other_all_values_To_Tensor = other_all_values_To_Tensor.view(-1, attend_heads,other_all_values_To_Tensor.shape[1],other_all_values_To_Tensor.shape[2])

        other_all_values_To_Tensor = other_all_values_To_Tensor.permute(2,0,1,3).contiguous()
        new_shape = other_all_values_To_Tensor.size()[:-2] + (hidden_dim,)
        b = other_all_values_To_Tensor.view(*new_shape)
        if is_unsqueeze == True:
            b = b.squeeze(0)
        return b,all_attend_probs




class SelectorNet(nn.Module):
    def __init__(self, frames):
        super(SelectorNet, self).__init__()
        self.act_fea_cv1 = nn.Conv1d(in_channels=frames, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.act_fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.act_fc1 = nn.Linear(128*32, 512)
        self.act_fc2 =  nn.Linear(512, 128)
        self.dqn_fc = nn.Sequential(
            nn.Linear(128+3+2+2+3+2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, position, goal, speed,other_position,other_speed):
        """
            returns an integer to estimate the value of communication
        """
        is_unsqueeze = False
        if len(x.shape)==2:
            x = x.unsqueeze(0)
            position = position.unsqueeze(0)
            goal = goal.unsqueeze(0)
            speed = speed.unsqueeze(0)
            other_position = other_position.unsqueeze(0)
            other_speed = other_speed.unsqueeze(0)
            is_unsqueeze = True
        a = F.leaky_relu(self.act_fea_cv1(x)) # size:[batch_size or not, n,32,255]  p.s. n is the number of robot swarm
        a = F.leaky_relu(self.act_fea_cv2(a)) # size:[n,32,128]
        a = a.view(a.shape[0], -1)  # size:[n,4096]  [number_of_robot, output_channel, output_dimension] ->[number_of_robot, dimension]

        a = F.leaky_relu(self.act_fc1(a)) # size:[n,256]
        #a = torch.cat((a, goal, speed), dim=-1) # size:[n,260]
        a = F.leaky_relu(self.act_fc2(a))     # size:[n,128] # actor intention
        a = torch.cat((a,position, goal, speed,other_position,other_speed),-1)   # @HAVEDONE: change to a = torch.cat((a,position, goal, speed),-1), put off goal and speed in the line 113
        a = self.dqn_fc(a)
        if is_unsqueeze:
            a.squeeze(0)
        return a








if __name__ == '__main__':
    from torch.autograd import Variable

    net = CNNPolicy(3, 2)

    observation = Variable(torch.randn(2, 3))
    v, action, logprob, mean = net.forward(observation)
    print(v)

