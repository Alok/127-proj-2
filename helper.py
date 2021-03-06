#!/usr/bin/env python
# encoding: utf-8

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


def dft_3D(arr):
    # TODO check that this actually computes the 3d transform
    return np.fft.fftn(arr)

def inv_2D(arr):
    return np.fft.ifft2(arr)
