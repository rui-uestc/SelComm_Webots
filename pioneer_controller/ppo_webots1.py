import heapq
import os
import logging
import sys
import socket
import traceback

import numpy as np
import random
import rospy
import torch
import torch.nn as nn
from mpi4py import MPI

from torch.optim import Adam
from collections import deque

from model.net import CNNPolicy, SelectorNet
# from stage_world1 import StageWorld
from webots_world1 import WebotsWorld
from model.ppo import ppo_update_stage1, generate_train_data
from model.ppo import generate_action
from model.ppo import transform_buffer
from model.dqn import get_influence_list, dqn_update1

import argparse
from controller import Supervisor
import os

import time
import copy
import tf




# from tensorboardX import SummaryWriter
# writer = SummaryWriter('runs/rewards')


comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

MAX_EPISODES = 50000
LASER_BEAM = 512
LASER_HIST = 3
HORIZON = 128  # 128
GAMMA = 0.99
LAMDA = 0.95
BATCH_SIZE = 32  # Because I have changed the code, the total number of sample pool is always 128 instead of 128*NUM_ENV, so BATCH_SIZE should be less than 128 to make sense.
EPOCH = 4
COEFF_ENTROPY = 5e-4
CLIP_VALUE = 0.1
NUM_ENV = 8
OBS_SIZE = 512
ACT_SIZE = 2
LEARNING_RATE = 5e-5
BANDWIDTH = 3

# hyperparameters for DQN
DQN_LR = 0.01
MEMORY_SIZE = 2000
MEMORY_THRESHOLD = 32
V_NETWORK_ITERATION = 100
STAY_SELECTOR = 5  # 5


def run(comm, env, policy, policy_path, action_bound, optimizer,
        selector, target_selector, selector_optimizer, mse_selector, mode):
    # rate = rospy.Rate(5)
    buff = []
    dqn_buff = deque()
    global_update = 0
    dvn_update_count = 0
    global_step = 0

    if env.index == 0:
        env.reset_world()

    for id in range(MAX_EPISODES):  # refresh for a agent
        env.reset_pose()
        env.generate_goal_point()
        terminal = False
        ep_reward = 0
        step = 0
        dqn_reward = 0

        obs = env.get_laser_observation()
        obs_stack = deque([obs, obs, obs])  # transform to three channes
        goal = np.asarray(env.get_local_goal())
        speed = np.asarray(env.get_self_speed())
        position = np.asarray(env.get_position())
        state = [obs_stack, goal, speed, position]
        # print('position',position,'local_goal',goal,'dist',np.sqrt(goal[0]**2+goal[1]**2))

        while not terminal and env.robot.step(100) != -1:
            state_list = comm.gather(state, root=0)
            # print state_list[0][3]
            position_list = comm.gather(position, root=0)
            # print('sssss',position)
            if env.index == 0:
                if mode is 'position':
                    adj_list = get_adjacency_list(position_list, bandwidth=BANDWIDTH)
                    # print(adj_list)
                    # adj_list *= 0
                elif mode is 'random':
                    adj_list = [[0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                                [1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                                [1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                                [0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1],
                                [0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1],
                                [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                                [1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                                [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1],
                                [0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1],
                                [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0],
                                ]
                    adj_list = np.array(adj_list)
                    # adj_list *= 0

                elif mode is 'dqn':
                    if step % STAY_SELECTOR == 0:
                        adj_list, _ = get_influence_list(state_list, selector=selector, bandwidth=BANDWIDTH)
                    # adj_list_position = get_adjacency_list(position_list, bandwidth=BANDWIDTH)
                    # print(adj_list)
                    # print(adj_list_position)
                    # # adj_list = [[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,],
                    # #              [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,],
                    # #              [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1,],
                    # #              [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1,],
                    # #              [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1,],
                    # #              [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1,],
                    # #              [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,],
                    # #              [0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,],
                    # #              [0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,],
                    # #              [0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,],
                    # #              [0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,],
                    # #              [0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,]]
                    # # adj_list = np.array(adj_list)
                    # print('similarity ratio: ', ((adj_list * adj_list_position).sum())/(NUM_ENV*BANDWIDTH))
                    # print('--------------------')
                    # adj_list *= 0
                else:
                    traceback.print_exc()
                for state_, adj_ in zip(state_list, adj_list):
                    state_.append(np.asarray(adj_))
            ''' generate actions at rank == 0   
            Only robot whose rank==0 has the state_list which contains other's message. Messages in other robots is Nonef
            Robot 0 plays a role of central node.
            '''
            v, a, logprob, scaled_action, all_attend_probs = generate_action(env=env, state_list=state_list,
                                                                             policy=policy, action_bound=action_bound)
            # print(all_attend_probs) #12*12 dig0 matrix   axis=1 means the weights

            # execute actions
            real_action = comm.scatter(scaled_action, root=0)
            env.control_vel(real_action)

            # rate.sleep()
            rospy.sleep(0.001)

            # get informtion
            r, terminal, result = env.get_reward_and_terminate(step)
            ep_reward += r
            global_step += 1
            dqn_reward += r

            step += 1

            # get next state
            s_next = env.get_laser_observation()
            left = obs_stack.popleft()
            obs_stack.append(s_next)
            goal_next = np.asarray(env.get_local_goal())
            speed_next = np.asarray(env.get_self_speed())
            position_next = np.asarray(env.get_position())
            state_next = [obs_stack, goal_next, speed_next, position_next]

            if global_step % HORIZON == 0:
                state_next_list = comm.gather(state_next, root=0)
                position_next_list = comm.gather(position_next, root=0)
                if env.index == 0:
                    if mode is 'position':
                        adj_next_list = get_adjacency_list(position_next_list, bandwidth=BANDWIDTH)
                    elif mode is 'random':
                        adj_next_list = get_random_list(position_next_list, bandwidth=BANDWIDTH)
                    elif mode is 'dqn':
                        adj_next_list, _ = get_influence_list(state_next_list, selector=selector, bandwidth=BANDWIDTH)
                    else:
                        traceback.print_exc()
                    for state_next_, adj_next_ in zip(state_next_list, adj_next_list):
                        state_next_.append(np.asarray(adj_next_))
                last_v, _, _, _, _ = generate_action(env=env, state_list=state_next_list, policy=policy,
                                                     action_bound=action_bound)

            # next_q for DQN
            state_next_list = comm.gather(state_next, root=0)
            if env.index == 0:
                if mode is 'dqn':
                    _, next_q = get_influence_list(state_next_list, selector=target_selector, bandwidth=BANDWIDTH)
                else:
                    next_q = None

            # print(all_attend_probs)
            # add transitons in buff and update policy
            r_list = comm.gather(r, root=0)
            if step % STAY_SELECTOR == 0:
                dqn_reward = dqn_reward / STAY_SELECTOR
            dqn_reward_list = comm.gather(dqn_reward, root=0)
            # global reward using attention
            # if r_list is not None:
            #     #print('before',r_list)
            #     r_list = r_list + 1.0 * np.matmul(r_list,all_attend_probs.T)
            # print('after', r_list)
            ''' mean reward
            if r_list is not None:
                sum = 0
                for i in r_list:
                    sum += i
                r_list = [sum/len(r_list) for _ in range(len(r_list))]
                #print(r_list)
            '''

            terminal_list = comm.gather(terminal, root=0)
            '''
            if env.index == 0: # index is initialized by rank
                buff.append((state_list, a, r_list, terminal_list, logprob, v))
                if len(buff) > HORIZON - 1:
                    s_batch, goal_batch, speed_batch, position_batch,adj_batch, a_batch, r_batch, d_batch, l_batch, v_batch = \
                        transform_buffer(buff=buff)
                    t_batch, advs_batch = generate_train_data(rewards=r_batch, gamma=GAMMA, values=v_batch,
                                                              last_value=last_v, dones=d_batch, lam=LAMDA)
                    memory = (s_batch, goal_batch, speed_batch, position_batch,adj_batch, a_batch, l_batch, t_batch, v_batch, r_batch, advs_batch)
                    ppo_update_stage1(policy=policy, optimizer=optimizer, batch_size=BATCH_SIZE, memory=memory,
                                            epoch=EPOCH, coeff_entropy=COEFF_ENTROPY, clip_value=CLIP_VALUE, num_step=HORIZON,
                                            num_env=NUM_ENV, frames=LASER_HIST,
                                            obs_size=OBS_SIZE, act_size=ACT_SIZE, global_update = global_update)
                    #  For Network debug
                    # act_fea_cv1params = policy.state_dict()['act_fea_cv1.weight']
                    # actor1params =  policy.state_dict()['actor1.weight']
                    # queryparams = policy.state_dict()['query.bias'][0]
                    # keyparams = policy.state_dict()['key.bias'][0]
                    # valueparams = policy.state_dict()['value.bias'][0]
                    # act_fc3params =  policy.state_dict()['act_fc3.bias']
                    # writer.add_scalar('query.bias',queryparams,global_step=global_update)
                    # writer.add_scalar('key.bias', keyparams, global_step=global_update)
                    # writer.add_scalar('value.bias', valueparams, global_step=global_update)
                    # writer.add_scalar('act_fea_cv1.weight', act_fea_cv1params[0][0][0], global_step=global_update)
                    # writer.add_scalar('actor1.weight', actor1params[0][0], global_step=global_update)
                    # writer.add_scalar('act_fc3.weight', act_fc3params[0], global_step=global_update)


                    buff = []
                    global_update += 1
            '''
            if env.index == 0:
                is_update = False
                buff.append((state_list, a, r_list, terminal_list, logprob, v, next_q))
                if step % STAY_SELECTOR == 1:
                    dqn_state_list = state_list
                    dqn_a = a
                    dqn_terminal_list = terminal_list
                    dqn_logprob = logprob
                    dqn_v = v

                if step % STAY_SELECTOR == 0:
                    dqn_next_q = next_q
                    dqn_buff.append(
                        (dqn_state_list, dqn_a, dqn_reward_list, dqn_terminal_list, dqn_logprob, dqn_v, dqn_next_q))
                    if len(dqn_buff) > MEMORY_SIZE:
                        dqn_buff.popleft()

                if len(buff) > HORIZON - 1:
                    s_batch, goal_batch, speed_batch, position_batch, adj_batch, a_batch, r_batch, d_batch, l_batch, v_batch, next_q_batch = \
                        transform_buffer(buff=buff)
                    # print d_batch.shape  [HORIZON, NUM_ENV]
                    # print len(filter_index)
                    t_batch, advs_batch = generate_train_data(rewards=r_batch, gamma=GAMMA, values=v_batch,
                                                              last_value=last_v, dones=d_batch, lam=LAMDA)
                    memory = (
                    s_batch, goal_batch, speed_batch, position_batch, adj_batch, a_batch, l_batch, t_batch, v_batch,
                    r_batch, advs_batch, next_q_batch, d_batch)
                    ppo_update_stage1(policy=policy, optimizer=optimizer, batch_size=BATCH_SIZE, memory=memory,
                                      epoch=EPOCH, coeff_entropy=COEFF_ENTROPY, clip_value=CLIP_VALUE, num_step=HORIZON,
                                      num_env=NUM_ENV, frames=LASER_HIST,
                                      obs_size=OBS_SIZE, act_size=ACT_SIZE, global_update=global_update)
                    is_update = True

                    buff = []
                    global_update += 1
                if len(dqn_buff) > HORIZON and mode is 'dqn' and (step % STAY_SELECTOR == 0):  # and is_update==True:
                    s_batch, goal_batch, speed_batch, position_batch, adj_batch, a_batch, r_batch, d_batch, l_batch, v_batch, next_q_batch = \
                        transform_buffer(buff=dqn_buff)
                    dqn_memory = (s_batch, goal_batch, speed_batch, position_batch, r_batch, next_q_batch, d_batch)
                    dqn_update1(selector=selector, selector_optimizer=selector_optimizer, mse_selector=mse_selector,
                                batch_size=BATCH_SIZE, memory=dqn_memory)
                    dvn_update_count += 1
                    if dvn_update_count % V_NETWORK_ITERATION == 0:
                        target_selector.load_state_dict(selector.state_dict())
                        print('update target selector')

            state = state_next
            position = position_next

            if step % STAY_SELECTOR == 0:
                dqn_reward = 0

        if env.index == 0:
            if global_update != 0 and global_update % 20 == 0:
                torch.save(policy.state_dict(), policy_path + '/Stage1_{}.pth'.format(global_update))
                if mode is 'dqn':
                    torch.save(selector.state_dict(), policy_path + '/dqn_stage2_{}.pth'.format(global_update))
                logger.info('########################## model saved when update {} times#########'
                            '################'.format(global_update))

        drift = np.sqrt((env.goal_point[0] - env.init_pose[0]) ** 2 + (env.goal_point[1] - env.init_pose[1]) ** 2)

        if step > 1:
            logger.info('Env %02d, Goal (%05.1f, %05.1f), Episode %05d, Step %03d, Reward %-5.1f, Drift %05.1f, %s' % \
                        (env.index, env.goal_point[0], env.goal_point[1], id + 1, step, ep_reward, drift, result))
            logger_cal.info(ep_reward)

        # if env.index == 0:
        #     writer.add_scalar('reward of robot 0',ep_reward,global_step=global_update)
        # if env.index == 0:
        #     print all_result


def get_small_index(m, num):
    max_number = heapq.nsmallest(num, m)
    max_index = []
    for t in max_number:
        index = m.index(t)
        max_index.append(index)
        m[index] = 0
    return max_index


def get_adjacency_list(robots_position, bandwidth=3):
    num_robot = len(robots_position)
    communication_lists = []
    for robot_index in range(num_robot):
        dist_list = []
        for i in range(num_robot):
            dist = (robots_position[robot_index][0] - robots_position[i][0]) ** 2 + (
                        robots_position[robot_index][1] - robots_position[i][1]) ** 2
            dist_list.append(dist)
        communication_index = get_small_index(dist_list, bandwidth + 1)
        communication_lists.append(communication_index[1:])
    adj_list = np.zeros((num_robot, num_robot))
    for i in range(num_robot):
        for index in communication_lists[i]:
            adj_list[i][index] = 1
    return adj_list  # size:[12*3]


def get_random_list(robots_position, bandwidth=3):
    num_robot = len(robots_position)
    communication_lists = []
    for robot_index in range(num_robot):
        dist_list = []
        for i in range(num_robot):
            dist = random.random()
            dist_list.append(dist)
        communication_index = get_small_index(dist_list, bandwidth + 1)
        communication_lists.append(communication_index[1:])
    adj_list = np.zeros((num_robot, num_robot))
    for i in range(num_robot):
        for index in communication_lists[i]:
            adj_list[i][index] = 1
    return adj_list  # size:[12*3]




if __name__ == '__main__':


    # config args
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--mode', type=str, default='dqn',help='random, dqn, or random')
    parser.add_argument('-nr','--NumRobots', type=int, required=True)
    parser.add_argument('-np','--NumPedestrians', type=int, required=True)
    sys_args = parser.parse_args()

    # config log
    hostname = socket.gethostname()
    if not os.path.exists('./log/' + hostname):
        os.makedirs('./log/' + hostname)
    output_file = './log/' + hostname + '/output.log'
    cal_file = './log/' + hostname + '/cal.log'

    # config log
    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(output_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    logger_cal = logging.getLogger('loggercal')
    logger_cal.setLevel(logging.INFO)
    cal_f_handler = logging.FileHandler(cal_file, mode='a')
    file_handler.setLevel(logging.INFO)
    logger_cal.addHandler(cal_f_handler)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # webots
    robot_name = 'Robot' + str(comm_rank)
    os.environ['WEBOTS_ROBOT_NAME'] = robot_name
    robot = Supervisor()

    env = WebotsWorld(512, index=rank, robot=robot, num_robot=sys_args.NumRobots, num_pedestrian=sys_args.NumPedestrians)
    reward = None
    action_bound = [[-12.3, -12.3], [12.3, 12.3]]

    # torch.manual_seed(1)
    # np.random.seed(1)
    if rank == 0:
        policy_path = 'policy'
        # policy = MLPPolicy(obs_size, act_size)
        policy = CNNPolicy(frames=LASER_HIST, action_space=2)
        policy.cuda()
        opt = Adam(policy.parameters(), lr=LEARNING_RATE)
        mse = nn.MSELoss()

        selector = SelectorNet(frames=LASER_HIST)
        target_selector = SelectorNet(frames=LASER_HIST)
        selector.cuda()
        target_selector.cuda()
        opt_selector = Adam(selector.parameters(), lr=DQN_LR)
        mse_selector = nn.MSELoss()

        if not os.path.exists(policy_path):
            os.makedirs(policy_path)

        file = policy_path + '/stage2_1620.pth'
        if os.path.exists(file):
            logger.info('####################################')
            logger.info('############Loading Model###########')
            logger.info('####################################')
            state_dict = torch.load(file)
            policy.load_state_dict(state_dict)
        else:
            logger.info('#####################################')
            logger.info('############Start Training###########')
            logger.info('#####################################')

        file_dqn = policy_path + '/dqn_stage2_1620.pth'
        if os.path.exists(file_dqn):
            logger.info('####################################')
            logger.info('############Loading DQN Model###########')
            logger.info('####################################')
            dqn_state_dict = torch.load(file_dqn)
            selector.load_state_dict(dqn_state_dict)
        else:
            logger.info('#####################################')
            logger.info('############Start DQN Training###########')
            logger.info('#####################################')
        # selector_dict = selector.state_dict()
        # pretrained_dict = {k: v for k, v in state_dict.items() if k in selector_dict}
        # selector_dict.update(pretrained_dict)
        # selector.load_state_dict(selector_dict)
        # target_selector.load_state_dict(selector_dict)
    else:
        policy = None
        policy_path = None
        opt = None
        selector = None
        target_selector = None
        opt_selector = None
        mse_selector = None

    try:
        mode = sys_args.mode
        print('#################MODE: ', mode, '#######################')
        run(comm=comm, env=env, policy=policy, policy_path=policy_path, action_bound=action_bound, optimizer=opt,
            selector=selector, target_selector=target_selector, selector_optimizer=opt_selector,
            mse_selector=mse_selector, mode=mode)
    except KeyboardInterrupt:
        traceback.print_exc()


