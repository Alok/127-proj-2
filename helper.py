#!/usr/bin/env python3
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
import sklearn
import numpy as np

from ptpdb import set_trace

def dft_3D(arr):
    # TODO check that this actually computes the 3d transform
    return scipy.fftn(arr)

def inv_2D(arr):
    return scipy.ifft2(arr)
