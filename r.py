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

    for i in xrange(N):
        # construct the Image * rectangle of the image in 3D
        B_1[:, :, 0] = I_0

    # For rest of images, calculate the back projection
    for n in xrange(1, len(images)):
        Ii = images[n]
        R = orientations[n]
        Bi = np.zeros(N * N * N)
        Bi = np.reshape(Bi, (N, N, N))

        for j in xrange(N):
            # construct the Image * rectangle of the image in 3D
            Bi[:, :, j] = Ii
            # generating the interpolater

        x = np.linspace(-1, 1, N)
        gri_image = scipy.interpolate.RegularGridInterpolator((x, x, x), Bi, bounds_error=False, fill_value=0)

        # rotating the molecule
        X, Y, Z = np.meshgrid(np.linspace(-1, 1, N),
                              np.linspace(-1, 1, N),
                              np.linspace(-1, 1, N))

        C = [X.flatten(), Y.flatten(), Z.flatten()]

        Bi = gri_image(np.dot(R.T, C).T)
        Bi = np.reshape(Bi, (N, N, N))
        B_1 = np.add(B_1, Bi)

    P = np.fft.fftn(np.fft.fftshift(B1))

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
