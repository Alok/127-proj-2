#!/usr/bin/env python

import vanheel
import numpy as np
import copy
import scipy

import scipy.interpolate
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from mpl_toolkits.mplot3d import Axes3D
from MRCFile import MRCFile
from numpy import abs
from numpy import cos
from numpy import sin
from numpy import sqrt

def project_fst(mol, R):
    # generating 3D fourier transform
    dft_mol = np.fft.fftn(mol)

    # shifting the index of fourier transform
    shifted_dft_mol = np.fft.fftshift(dft_mol)

    # generating the interpolater
    l = len(mol)
    N = 2 * l
    interpolate_step = np.linspace(0, l - 1, l)
    interpolated_grid = scipy.interpolate.RegularGridInterpolator((interpolate_step, interpolate_step, interpolate_step), shifted_dft_mol)

def rodrigues_rotation(v, viewing_dir, angle):
    # v is (3,1) array
    # k is unit vector describing an axis of rotation
    cross = np.cross

    a = viewing_dir[:,0]
    b = viewing_dir[:,1]
    ab = np.cross(a, b)
    k = ab / np.linalg.norm(ab)
    v * cos(angle) + (cross(k, v)) *sin(angle) + k * np.dot(k, v) * (1 - cos(angle))


    # rotate mol with interpolation so P_f is in xy or (parallel to it)

    # zero pad mol to M^3 array with np.pad
    # zero padding should be even on each side

    # 3D dft mol
    # extract slice where z =0
