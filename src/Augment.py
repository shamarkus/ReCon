import random
import numpy as np
import math
from math import sin,cos
import random
import  torch
from torch import nn
import torch.nn.functional as F

class SelectFrames(object):
    def __init__(self, frames):
        self.frames = frames
    def __call__(self, data_numpy):
        return data_numpy[:, :self.frames, :, :]

class RandomHorizontalFlip(object):
    def __init__(self, p = 0.5):
        self.p = p
    def __call__(self, data_numpy):
        C, T, V, M = data_numpy.shape
        if random.random() < self.p:
            time_range_order = [i for i in range(T)]
            time_range_reverse = list(reversed(time_range_order))
            return data_numpy[:, time_range_reverse, :, :]
        else:
            return data_numpy.copy()

class Rotate(object):
    def __init__(self, axis = None, angle = None, ):
        self.first_axis = axis
        self.first_angle = angle
    def __call__(self, data_numpy):
        if self.first_axis != None:
            axis_next = self.first_axis
        else:
            axis_next = random.randint(0,2)

        if self.first_angle != None:
            if isinstance(self.first_angle, list):
                angle_big = self.first_angle[0] + self.first_angle[1]
                angle_small = self.first_angle[0] - self.first_angle[1]
                angle_next = random.uniform(angle_small, angle_big)
            else:
                angle_next = self.first_angle
        else:
            # angle_list = [0, 90, 180, 270]
            # angle_next = random.sample(angle_list, 1)[0]
            angle_next = random.uniform(0, 30)

        temp = data_numpy.copy()
        angle = math.radians(angle_next)
        # x
        if axis_next == 0:
            R = np.array([[1, 0, 0],
                          [0, cos(angle), sin(angle)],
                          [0, -sin(angle), cos(angle)]])
        # y
        if axis_next == 1:
            R = np.array([[cos(angle), 0, -sin(angle)],
                          [0, 1, 0],
                          [sin(angle), 0, cos(angle)]])

        # z
        if axis_next == 2:
            R = np.array([[cos(angle), sin(angle), 0],
                          [-sin(angle), cos(angle), 0],
                          [0, 0, 1]])
        R = R.transpose()
        temp = np.dot(temp.transpose([1, 2, 3, 0]), R)
        temp = temp.transpose(3, 0, 1, 2)
        return temp


class Gaus_noise(object):
    def __init__(self, mean= 0, std = 0.02):
        self.mean = mean
        self.std = std
    def __call__(self, data_numpy):
        temp = data_numpy.copy()
        C, T, V, M = data_numpy.shape
        noise = np.random.normal(self.mean, self.std, size=(C, T, V, M))
        return temp + noise

class Shear(object):
    def __init__(self, s1 = None, s2 = None):
        self.s1 = s1
        self.s2 = s2

    def __call__(self, data_numpy):
        temp = data_numpy.copy()
        if self.s1 != None:
            s1_list = self.s1
        else:
            s1_list = [random.uniform(-1, 1),random.uniform(-1, 1),random.uniform(-1, 1)]
            # print(s1_list[0])
        if self.s2 != None:
            s2_list = self.s2
        else:
            s2_list = [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]

        R = np.array([[1,     s1_list[0], s2_list[0]],
                      [s1_list[1], 1,     s2_list[1]],
                      [s1_list[2], s2_list[2], 1]])

        R = R.transpose()
        temp = np.dot(temp.transpose([1, 2, 3, 0]), R)
        temp = temp.transpose(3, 0, 1, 2)
        return  temp

class Diff(object):
    def __call__(self, data_numpy):
        C, T, V, M = data_numpy.shape
        x_new = np.zeros((C, T, V, M))
        for t in range(T - 1):
            x_new[:, t, :, :] = data_numpy[:, t + 1, :, :] - data_numpy[:, t, :, :]
        return x_new

'''========================================================'''

# b: crop and resize
def subsample(data_numpy, time_range):
    C, T, V, M = data_numpy.shape
    if isinstance(time_range, int):
        all_frames = [i for i in range(T)]
        time_range = random.sample(all_frames, time_range)
        time_range.sort()
    x_new = np.zeros((C, T, V, M))
    x_new[:, time_range, :, :] = data_numpy[:, time_range, :, :]
    return x_new

# ok
# c: crop,resize (and flip)
def subSampleFlip(data_numpy, time_range):
    C, T, V, M = data_numpy.shape
    assert T >= time_range, "frames longer than data"
    if isinstance(time_range, int):
        all_frames = [i for i in range(T)]
        time_range = random.sample(all_frames, time_range)
        time_range_order = sorted(time_range)
        time_range_reverse =  list(reversed(time_range_order))
    x_new = np.zeros((C, T, V, M))
    x_new[:, time_range_order, :, :] = data_numpy[:, time_range_reverse, :, :]
    return x_new

# ok
# d: color distort.(drop)
def zero_out_axis(data_numpy, axis):
    # x, y, z -> axis : 0,1,2
    temp = data_numpy.copy()
    C, T, V, M = data_numpy.shape
    x_new = np.zeros((T, V, M))
    temp[axis] = x_new
    return temp


# e: color distort. (jitter)
def diff_on_axis(data_numpy, axis):
    temp = data_numpy.copy()
    C, T, V, M = data_numpy.shape
    for t in range(T - 1):
        temp[axis, t, :, :] = data_numpy[axis, t+1, :, :] - data_numpy[axis, t, :, :]
        temp[axis, -1, :, :] = np.zeros((V, M))
    return temp



# ok
# f: rotate
def rotate(data_numpy, axis, angle):
    temp = data_numpy.copy()
    angle = math.radians(angle)
    # x
    if axis == 0:
        R = np.array([[1, 0, 0],
                      [0, cos(angle), sin(angle)],
                      [0, -sin(angle), cos(angle)]])
    # y
    if axis == 1:
        R = np.array([[cos(angle), 0, -sin(angle)],
                       [0,1,0],
                       [sin(angle), 0, cos(angle)]])

    # z
    if axis == 2:
        R = np.array([[cos(angle),sin(angle),0],
                       [-sin(angle),cos(angle),0],
                       [0,0,1]])
    R = R.transpose()
    temp = np.dot(temp.transpose([1,2,3,0]),R)
    temp = temp.transpose(3,0,1,2)
    return temp

def gaus_noise(data_numpy, mean= 0, std = 0.01):
    temp = data_numpy.copy()
    C, T, V, M = data_numpy.shape
    noise = np.random.normal(mean, std, size=(C, T, V, M ))
    return temp + noise


class GaussianBlurConv(nn.Module):
    def __init__(self, channels=3, kernel=15, sigma=[0.1, 0.6]):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        self.kernel = kernel
        self.min_max_sigma = sigma
        radius = int(kernel / 2)
        self.kernel_index = np.arange(-radius, radius + 1)

    def __call__(self, x):

        sigma = random.uniform(self.min_max_sigma[0], self.min_max_sigma[1])
        blur_flter = np.exp(-np.power(self.kernel_index, 2.0) / (2.0 * np.power(sigma, 2.0)))
        kernel = torch.from_numpy(blur_flter).unsqueeze(0).unsqueeze(0)
        kernel = kernel.double()
        kernel = kernel.repeat(self.channels, 1, 1) 
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

        prob = np.random.random_sample()
        if prob < 0.5:
            x = x.transpose(0,2) 
            x = F.conv2d(x, self.weight, padding=(0, int((self.kernel - 1) / 2 )),   groups=self.channels)
            x = x.transpose(0,2) 
        return x

def temporal_slice(data_numpy, step):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    return data_numpy.reshape(C, T / step, step, V, M).transpose(
        (0, 1, 3, 2, 4)).reshape(C, T / step, V, step * M)


def random_shift(data_numpy):
    C, T, V, M = data_numpy.shape
    data_shift = np.zeros(data_numpy.shape)
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()

    size = end - begin
    bias = random.randint(0, T - size)
    data_shift[:, bias:bias + size, :, :] = data_numpy[:, begin:end, :, :]

    return data_shift


def openpose_match(data_numpy):
    C, T, V, M = data_numpy.shape
    assert (C == 3)
    score = data_numpy[2, :, :, :].sum(axis=1)
    # the rank of body confidence in each frame (shape: T-1, M)
    rank = (-score[0:T - 1]).argsort(axis=1).reshape(T - 1, M)

    # data of frame 1
    xy1 = data_numpy[0:2, 0:T - 1, :, :].reshape(2, T - 1, V, M, 1)
    # data of frame 2
    xy2 = data_numpy[0:2, 1:T, :, :].reshape(2, T - 1, V, 1, M)
    # square of distance between frame 1&2 (shape: T-1, M, M)
    distance = ((xy2 - xy1) ** 2).sum(axis=2).sum(axis=0)

    # match pose
    forward_map = np.zeros((T, M), dtype=int) - 1
    forward_map[0] = range(M)
    for m in range(M):
        choose = (rank == m)
        forward = distance[choose].argmin(axis=1)
        for t in range(T - 1):
            distance[t, :, forward[t]] = np.inf
        forward_map[1:][choose] = forward
    assert (np.all(forward_map >= 0))

    # string data
    for t in range(T - 1):
        forward_map[t + 1] = forward_map[t + 1][forward_map[t]]

    # generate data
    new_data_numpy = np.zeros(data_numpy.shape)
    for t in range(T):
        new_data_numpy[:, t, :, :] = data_numpy[:, t, :, forward_map[
                                                             t]].transpose(1, 2, 0)
    data_numpy = new_data_numpy

    # score sort
    trace_score = data_numpy[2, :, :, :].sum(axis=1).sum(axis=0)
    rank = (-trace_score).argsort()
    data_numpy = data_numpy[:, :, :, rank]

    return data_numpy


