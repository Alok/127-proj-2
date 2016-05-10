#!/usr/bin/env python

import math
import os
import sys

import scipy
import matplotlib.pyplot as plt
import numpy as np


def reconstruct(images, orientations):
    """images is set of 2d arrays, where I[x][y] will give the intensity
    of detected point (x, y) on the image.
    orientations is a set of 3 by 3 matrix that give the viewing angle of
    corresponding image.
    We assume that the images are produced using our own projection_fst.
    So the image is a square with length equals the original length of mol."""
    # N is the length of mol
    N = len(images[0])

    # Generate the first back projection
    B_1 = np.zeros((N, N, N))
    I_0 = images[0]

    # fill array with I_0
    # B_1[:, :, :] = I_0
    B_1[:] = I_0

    # For rest of images, calculate the back projection
    for n in xrange(1, len(images)):
        I_i = images[n]
        R = orientations[n]
        B_i = np.zeros((N, N, N))

        B_1[:] = I_i

        # generating the interpolater
        N_range = np.linspace(-1, 1, N)
        gri_image = scipy.interpolate.RegularGridInterpolator((N_range, N_range, N_range), B_i, bounds_error=False, fill_value=0)

        # rotating the molecule
        x, y, z = np.meshgrid(N_range, N_range, N_range)

        C = [x.flatten(), y.flatten(), z.flatten()]

        B_i = gri_image(np.dot(R.T, C).T)
        B = np.add(B_1, B_i)

    P = np.fft.fftn(np.fft.fftshift(B))

    for i in xrange(N):
        for j in xrange(N):
            for k in xrange(N):

                H = 0
                temp = np.asarray( [- (N - 1) / 2 + j, (N - 1) / 2 + i, - (N - 1) / 2 + k])

                for n in xrange(0, len(images)):
                    val = np.dot(temp.T, orientations[n][2])
                    H = H + (math.sin(N * (math.pi) * val) / math.pi * val)

                P[i][j][k] = P[i][j][k] * (1 / H)

    P = np.fft.ifftn(np.fft.ifftshift(P))

    return P
