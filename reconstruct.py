#!/usr/bin/env python

import math
import random
import os
import sys
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

# TODO someone implement interpolation or explain it to me so I can do it
# properly

# We choose to let the molecule sit in the cube [-1,1]^3.
D = 1


def b(v, F, I, idx):
    x_i = np.dot(v, F[idx][:, 0])
    y_i = np.dot(v, F[idx][:, 1])
    z_i = np.dot(v, F[idx][:, 2])
    vec = [x_i, y_i, z_i]

    return ifft(fft(I[idx] * fft(l(vec))))


def l(v):
    x, y, z = unpack(v)

    def rect(alpha):
        return 1 if (np.abs(alpha) <= D) else 0

    def delta(alpha, beta):
        return np.infty if (alpha**2 + beta**2 == 1) else 0
    return delta(x, y) * rect(z)


def r_hat(r):
    return fftshift(fft(r))


def unpack(v):
    x = v[0]
    y = v[1]
    z = v[2]
    return x, y, z


def B(v, F, I):
    N = len(F)
    return np.sum([fft(b(v, F, I, i)) for i in range(N)])


def H(v, F, I):
    N = len(F)
    return np.sum([D * np.sinc(D * math.pi * np.dot(v, F[i][:, 2])) for i in range(N)])


def p(v, F, I):
    return ifft(B(v, F, I) / H(v, F, I))
