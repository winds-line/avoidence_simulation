import math
import random

def left_15(x, y, x_d, y_d, speed):
    x = x + 5 * x_d * speed
    y = y + 19 * y_d
    return x, y


def left_30(x, y, x_d, y_d, speed):
    x = x + 10 * x_d * speed
    y = y + 17 * y_d
    return x, y


def left_45(x, y, x_d, y_d, speed):
    x = x + 14 * x_d * speed
    y = y + 14 * y_d * speed
    return x, y


def right_15(x, y, x_d, y_d, speed):
    x = x + 19 * x_d * speed
    y = y + 5 * y_d * speed
    return x, y


def right_30(x, y, x_d, y_d, speed):
    x = x + 17 * x_d * speed
    y = y + 10 * y_d * speed
    return x, y


def right_45(x, y, x_d, y_d, speed):
    x = x + 14 * x_d * speed
    y = y + 14 * y_d * speed
    return x, y


def forward(x, y, x_d, y_d, speed):
    x = x + 20 * x_d * speed
    y = y + 20 * y_d * speed
    return x, y


def obstacle_rand_go():
    temp = random.randint(0, 23)
    x = round(20*math.sin(temp*(math.pi/12)))
    y = round(20*math.cos(temp*(math.pi/12)))
    return x, y