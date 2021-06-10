import copy
import heapq

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

def get_influence_list(state_list,selector,bandwidth=3):
    num_robot = len(state_list)
    #print(num_robot)
    s_list, goal_list, speed_list, position_list = [], [], [], []
    for i in state_list:
        s_list.append(i[0])
        goal_list.append(i[1])
        speed_list.append(i[2])
        position_list.append(i[3])

    s_list = np.asarray(s_list)
    goal_list = np.asarray(goal_list)
    speed_list = np.asarray(speed_list)
    position_list = np.asarray(position_list)

    s_list = Variable(torch.from_numpy(s_list)).float().cuda()
    goal_list = Variable(torch.from_numpy(goal_list)).float().cuda()
    speed_list = Variable(torch.from_numpy(speed_list)).float().cuda()
    position_list = Variable(torch.from_numpy(position_list)).float().cuda()
    # v, a, logprob, mean, all_attend_probs = policy(s_list, goal_list, speed_list, position_list)  # generate NaN???
    # v, a, logprob = v.data.cpu().numpy(), a.data.cpu().numpy(), logprob.data.cpu().numpy()
    # scaled_action = np.clip(a, a_min=action_bound[0], a_max=action_bound[1])
    q_list = []
    communication_lists = []
    for this in range(num_robot):
        dist_list = []
        for other in range(num_robot):
            q = selector(s_list[this],position_list[this],goal_list[this],speed_list[this],position_list[other],speed_list[other])
            dist_list.append(q.detach().data.cpu().numpy())
            #print(position_list[this],position_list[other],q)
        # print(s)
        # print('-----',this,'-------------')
        # print(dist_list)
        q_list.append(copy.deepcopy(dist_list))
        communication_index = get_large_index(dist_list, bandwidth + 1)
        # print(communication_index)
        communication_lists.append(communication_index)
    adj_list = np.zeros((num_robot, num_robot))
    for i in range(num_robot):
        for index in communication_lists[i][:-1]:
            if index == i:
                #print('ssssss',index,communication_lists[i])
                adj_list[i][communication_lists[i][-1]] = 1
            else:
                adj_list[i][index] = 1
    #print(q_list)
    return adj_list,np.array(q_list).squeeze()  # size:[12*3]

def get_large_index(m, num):
    max_number = heapq.nlargest(num, m)
    max_index = []
    for t in max_number:
        index = m.index(t)
        max_index.append(index)
        m[index] = 0
    return max_index

def dqn_update(selector, selector_optimizer,mse_selector, batch_size, memory, filter_index, ):
    obss, goals, speeds, positions, rewards, last_q, dones = memory
    #print(advs.shape[0],'sssssssssssss',rewards.shape[0])
    sampler = BatchSampler(SubsetRandomSampler(list(range(batch_size))), batch_size=batch_size,
                           drop_last=False)
    for i, index in enumerate(sampler):
        this_filter_index = [(index.index(filter_[0]),filter_[1]) for filter_ in filter_index if filter_[0] in index]
        sampled_obs = Variable(torch.from_numpy(obss[index])).float().cuda()
        sampled_goals = Variable(torch.from_numpy(goals[index])).float().cuda()
        sampled_speeds = Variable(torch.from_numpy(speeds[index])).float().cuda()
        sample_positions = Variable(torch.from_numpy(positions[index])).float().cuda()
        sampled_rewards = Variable(torch.from_numpy(rewards[index])).float().cuda()
        # print(last_q[index])
        # print('-----------------')
        sample_last_q = Variable(torch.from_numpy(last_q[index])).float().cuda()
        sample_dones = Variable(torch.from_numpy(dones[index])).float().cuda()

        sampled_obs = sampled_obs.permute(1, 0, 2,3).contiguous()
        sample_positions = sample_positions.permute(1, 0, 2).contiguous()
        sampled_goals = sampled_goals.permute(1, 0, 2).contiguous()
        sampled_speeds = sampled_speeds.permute(1, 0, 2).contiguous()

        num_robot = len(sampled_obs)
        eval_q = []
        for this, s in enumerate(sampled_obs):
            dist_list = []
            for other in range(num_robot):
                q = selector(s, sample_positions[this], sampled_goals[this], sampled_speeds[this], sample_positions[other],sampled_speeds[other])
                dist_list.append(q)
            eval_q.append(dist_list)
        eval_q_tensor = []
        for i in eval_q:
            eval_q_tensor.append(torch.stack(i))
        eval_q_tensor = torch.stack(eval_q_tensor)
        # print(eval_q_tensor.shape)
        eval_q_tensor = eval_q_tensor.squeeze(3).permute(2,0, 1).contiguous()

        sampled_rewards = sampled_rewards.unsqueeze(2)

        sampled_rewards = sampled_rewards.repeat(1, 1, num_robot)
        sample_dones = sample_dones.unsqueeze(2)
        sample_dones = sample_dones.repeat(1, 1, num_robot)

        # print(sampled_rewards.shape)
        # print(sample_last_q.shape)
        # print(sample_dones.shape)
        # print(sample_adjs.shape)
        # print('------------')
        if len(sample_last_q.shape) == 1:
            sample_last_q = sample_last_q.unsqueeze(-1)
            sample_last_q = sample_last_q.unsqueeze(-1)
        target_q = sampled_rewards + 0.90 * sample_last_q * sample_dones

        for this_filter in this_filter_index:
            eval_q_tensor[this_filter[0], this_filter[1]] *= 0
            target_q[this_filter[0], this_filter[1]] *= 0
        eval_q_tensor = eval_q_tensor.view(-1,1)
        target_q = target_q.view(-1, 1)
        loss = mse_selector(eval_q_tensor,target_q) #rewards are all the same, update in this way will make all evaluated q the same, too
        #print('dqn loss: ',loss)
        selector_optimizer.zero_grad()
        loss.backward()
        selector_optimizer.step()
    #print('update dqn')


def dqn_update1(selector, selector_optimizer,mse_selector, batch_size, memory,  ):
    obss, goals, speeds, positions, rewards, last_q, dones = memory
    #print(advs.shape[0],'sssssssssssss',rewards.shape[0])
    sampler = BatchSampler(SubsetRandomSampler(list(range(batch_size))), batch_size=batch_size,
                           drop_last=False)
    for i, index in enumerate(sampler):
        sampled_obs = Variable(torch.from_numpy(obss[index])).float().cuda()
        sampled_goals = Variable(torch.from_numpy(goals[index])).float().cuda()
        sampled_speeds = Variable(torch.from_numpy(speeds[index])).float().cuda()
        sample_positions = Variable(torch.from_numpy(positions[index])).float().cuda()
        sampled_rewards = Variable(torch.from_numpy(rewards[index])).float().cuda()
        # print(last_q[index])
        # print('-----------------')
        sample_last_q = Variable(torch.from_numpy(last_q[index])).float().cuda()
        sample_dones = Variable(torch.from_numpy(dones[index])).float().cuda()

        sampled_obs = sampled_obs.permute(1, 0, 2,3).contiguous()
        sample_positions = sample_positions.permute(1, 0, 2).contiguous()
        sampled_goals = sampled_goals.permute(1, 0, 2).contiguous()
        sampled_speeds = sampled_speeds.permute(1, 0, 2).contiguous()

        num_robot = len(sampled_obs)
        eval_q = []
        for this, s in enumerate(sampled_obs):
            dist_list = []
            for other in range(num_robot):
                q = selector(s, sample_positions[this], sampled_goals[this], sampled_speeds[this], sample_positions[other],sampled_speeds[other])
                dist_list.append(q)
            eval_q.append(dist_list)
        eval_q_tensor = []
        for i in eval_q:
            eval_q_tensor.append(torch.stack(i))
        eval_q_tensor = torch.stack(eval_q_tensor)
        # print(eval_q_tensor.shape)
        eval_q_tensor = eval_q_tensor.squeeze(3).permute(2,0, 1).contiguous()

        sampled_rewards = sampled_rewards.unsqueeze(2)

        sampled_rewards = sampled_rewards.repeat(1, 1, num_robot)
        sample_dones = sample_dones.unsqueeze(2)
        sample_dones = sample_dones.repeat(1, 1, num_robot)

        # print(sampled_rewards.shape)
        # print(sample_last_q.shape)
        # print(sample_dones.shape)
        # print(sample_adjs.shape)
        # print('------------')
        if len(sample_last_q.shape) == 1:
            sample_last_q = sample_last_q.unsqueeze(-1)
            sample_last_q = sample_last_q.unsqueeze(-1)
        target_q = sampled_rewards + 0.90 * sample_last_q * sample_dones
        eval_q_tensor = eval_q_tensor.view(-1,1)
        target_q = target_q.view(-1, 1)
        loss = mse_selector(eval_q_tensor,target_q) #rewards are all the same, update in this way will make all evaluated q the same, too
        #print('dqn loss: ',loss)
        selector_optimizer.zero_grad()
        loss.backward()
        selector_optimizer.step()
