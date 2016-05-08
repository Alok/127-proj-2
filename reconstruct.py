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

def l(x,y,z):
    def rect(a):
        return 1 if (-D/2 <= a <= D/2) else 0
    return delta(x, y) * rect(z)


