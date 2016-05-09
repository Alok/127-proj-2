#!/usr/bin/env python

import math
import random
import os
import sys
import subprocess
import functools
import itertools

import scipy
import matplotlib.pyplot as plt
import numpy as np

from numpy.fft import fft
from numpy.fft import fftshift
from numpy.fft import ifft
from numpy.fft import ifft2
from numpy.fft import ifftshift


# i choose to let the molecule sit in the cube [-1,1]^3
D = 1


def b(v, F, idx):
    x_i = np.dot(v, F[idx][:, 0])
    y_i = np.dot(v, F[idx][:, 1])
    z_i = np.dot(v, F[idx][:, 2])
    vec = [x_i, y_i, z_i]
    return vec


def l(x, y, z):
    def rect(a):
        return 1 if (np.abs(a) <= D) else 0

    def delta(a, b):
        return np.infty if (a**2 + b**2 == 1) else 0
    return delta(x, y) * rect(z)


def r_hat(r):

    return fftshift(fft(r))

# inv FFT of {B(x,y,z) / H(x,y,z)}
# x_i = (x,y,z) * F[:,0]
# y_i = (x,y,z) * F[:,1]
# z_i = (x,y,z) * F[:,2]
# B = sum over all imgs of FFT(b_i)
# H = sum over imgs of D * sinc


def unpack(v):
    x = v[0]
    y = v[1]
    z = v[2]
    return x, y, z


def B(v, F):
    N = len(F)
    return np.sum([fft(b(v, F, i)) for i in range(N)])


def H(v, F):
    N = len(F)
    return np.sum([D * np.sinc(D * math.pi * np.dot(v, F[i][:, 2])) for i in range(N)])


def p(v, F):
    return ifft(B(v, F) / H(v, F))
