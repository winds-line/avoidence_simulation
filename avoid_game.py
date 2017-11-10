import pygame
import time
import cv2
import random
from avoid_util import *
import math
import numpy as np
from RL_brain_memory import DeepQNetwork

obstacle_num = 10
obstacles = []
obstacle_positions_x = []
obstacle_positions_y = []
sleep_time = 0.1

pygame.init()
width, height = 400, 800
screen = pygame.display.set_mode((width, height))
player = pygame.image.load('ball1.png')
position_x = np.zeros(1)
position_y = np.zeros(1)
OBSERVE = 100000 # timesteps to observe before training
EXPLORE = 3000000 # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 50000



def init_game():
    position_x[0] = width/2 - 10
    position_y[0] = 780
    for _ in range(obstacle_num):
        temp_obstacle = pygame.image.load('ball2.png')
        obstacles.append(temp_obstacle)
        if random.random() < 0.5:
            obstacle_positions_x.append(0)
        else:
            obstacle_positions_x.append(width-20)
        obstacle_positions_y.append(random.randint(0, 780))
    screen.fill(0)
    screen.blit(player, (position_x, position_y))
    for index in range(obstacle_num):
        screen.blit(obstacles[index], (obstacle_positions_x[index], obstacle_positions_y[index]))
    pygame.display.flip()
    time.sleep(sleep_time)


def init_obstacle(index):
    if random.random() < 0.5:
        obstacle_positions_x[index] = 0
    else:
        obstacle_positions_x[index] = width - 20
    obstacle_positions_y[index] = random.randint(0, 780)


def obstacle_go():
    for index in range(obstacle_num):
        temp_x, temp_y = obstacle_rand_go()
        speed = random.randint(1, 3)
        obstacle_positions_x[index] += temp_x*speed
        obstacle_positions_x[index] += temp_y*speed
        if obstacle_positions_x[index] < 0 or obstacle_positions_x[index] > width - 20 or obstacle_positions_y[index] < 0 or obstacle_positions_y[index]>height-20:
            init_obstacle(index)


def player_go(temp_lr, temp_step):
    if temp_step <= OBSERVE + EXPLORE and temp_step >= OBSERVE:
        temp_lr -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
    if temp_lr < 0.0001:
        temp_lr = 0.0001
    if temp_step <= OBSERVE:
        action = random.randint(0, 13)
    else:
        if random.random() < temp_lr:
            action = random.randint(0, 13)
        else:
            action = RL.choose_action(s_t)

    die = False
    if action <= 12:
        # print('action', action)
        temp_rad = math.pi - action * math.pi/12
        temp_x = round(20*math.cos(temp_rad))
        temp_y = round(20*math.sin(temp_rad))
        position_x[0] += temp_x
        # print('y', temp_y)
        position_y[0] -= temp_y
    else:
        temp_y = - 2
    if position_x[0] < 0 or position_x[0] > width - 20 or position_y[0] < 0 or position_y[0] > height - 20:
        position_x[0] = width / 2 - 10
        position_y[0] = 780
    reward = temp_y/20
    for index in range(obstacle_num):
        dist = math.pow(obstacle_positions_x[index] - position_x[0], 2) + math.pow(obstacle_positions_y[index] - position_y[0], 2)
        if dist <= 400:
            position_x[0] = width / 2 - 10
            position_y[0] = 780
            reward = -1
            die = True
            break
    if step >= OBSERVE:
        loss = RL.learn(temp_lr)
        print('step:', step, 'reward', reward, 'action', action, 'lr:', lr, 'loss', loss)
    return reward, die, action, lr


init_game()
lr = INITIAL_EPSILON
RL = DeepQNetwork(13, learning_rate=0.001, reward_decay=0.9, e_greedy=0.9, replace_target_iter=1, memory_size=50000, output_graph=True)
observation_init = pygame.surfarray.array3d(pygame.display.get_surface())
observation_init = cv2.cvtColor(observation_init, cv2.COLOR_BGR2GRAY)
ret1, observation_init = cv2.threshold(cv2.resize(observation_init, (80, 80)), 1, 255, cv2.THRESH_BINARY)
s_t = np.stack((observation_init, observation_init, observation_init, observation_init), axis=2)
step = 0
total_reward = 0.0
reward_list = []
while 1:
    if step >= OBSERVE+EXPLORE:
        break
    obstacle_go()
    r, Terminal, a, lr = player_go(lr, step)
    screen.fill(0)
    screen.blit(player, (position_x, position_y))
    for i in range(obstacle_num):
        screen.blit(obstacles[i], (obstacle_positions_x[i], obstacle_positions_y[i]))
    pygame.display.flip()
    observation = pygame.surfarray.array3d(pygame.display.get_surface())
    observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
    ret, observation = cv2.threshold(cv2.resize(observation, (80, 80)), 1, 255, cv2.THRESH_BINARY)
    s_t1 = np.append(observation, s_t[:, :, :3], axis=1)
    RL.store_transition(s_t, a, r, s_t1)
    s_t = s_t1
    step += 1
    total_reward += r
    if Terminal:
        s_t = np.stack((observation, observation, observation, observation), axis=1)
        total_reward = 0.0
    time.sleep(sleep_time)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit(0)

# RL.plot_cost()
# import matplotlib.pyplot as plt
# plt.plot(np.arange(len(reward_list)), reward_list)
# plt.ylabel('Reward')
# plt.xlabel('training steps')
# plt.show()
# print('game over')
# print(step)
