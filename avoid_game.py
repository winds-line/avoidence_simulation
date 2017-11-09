import pygame
import time
import cv2
import random
from avoid_util import *
import math
import numpy as np

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
    for i in range(obstacle_num):
        screen.blit(obstacles[i], (obstacle_positions_x[i], obstacle_positions_y[i]))
    pygame.display.flip()
    time.sleep(sleep_time)


def init_obstacle(index):
    if random.random() < 0.5:
        obstacle_positions_x[index] = 0
    else:
        obstacle_positions_x[index] = width - 20
    obstacle_positions_y[index] = random.randint(0, 780)


def obstacle_go():
    for i in range(obstacle_num):
        temp_x, temp_y = obstacle_rand_go()
        speed = random.randint(1, 3)
        obstacle_positions_x[i] += temp_x*speed
        obstacle_positions_x[i] += temp_y*speed
        if obstacle_positions_x[i] < 0 or obstacle_positions_x[i] > width - 20 or obstacle_positions_y[i] < 0 or obstacle_positions_y[i]>height-20:
            init_obstacle(i)


def player_go():
    action = random.randint(0, 13)
    if action <= 12:
        print('action', action)
        temp_rad = math.pi - action * math.pi/12
        temp_x = round(20*math.cos(temp_rad))
        temp_y = round(20*math.sin(temp_rad))
        position_x[0] += temp_x
        print('y', temp_y)
        position_y[0] -= temp_y
    if position_x[0] < 0 or position_x[0] > width - 20 or position_y[0] < 0 or position_y[0] > height - 20:
        position_x[0] = width / 2 - 10
        position_y[0] = 780
    for i in range(obstacle_num):
        dist = math.pow(obstacle_positions_x[i] - position_x[0], 2) + math.pow(obstacle_positions_y[i] - position_y[0], 2)
        if dist <= 400:
            position_x[0] = width / 2 - 10
            position_y[0] = 780
            break


init_game()
while 1:
    obstacle_go()
    player_go()
    screen.fill(0)
    screen.blit(player, (position_x, position_y))
    for i in range(obstacle_num):
        screen.blit(obstacles[i], (obstacle_positions_x[i], obstacle_positions_y[i]))
    pygame.display.flip()
    time.sleep(sleep_time)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit(0)
