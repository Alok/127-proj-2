#!/usr/bin/env python

import math
import os
import sys

import scipy
import matplotlib.pyplot as plt
import numpy as np


def reconstruct(images, orientations):
    """
    `images` is set of 2d arrays, where I[x][y] will give the intensity
    of detected point (x, y) on the image.
    `orientations` is a set of 3 by 3 matrix that give the viewing angle of
    corresponding image.
    We assume that the images are produced using our own projection_fst.
    So the image is a square with length equals the original length of mol.
    """
    # D is the length of mol
    D = len(images[0])

    # Generate the first back projection
    B = np.zeros((D, D, D))

    # For rest of images, calculate the back projection
    for i in xrange(len(images)):
        I = images[i]
        R = orientations[i]
        B_i = np.zeros((D, D, D))

        # tile matrices numpy
        B_test = np.tile(I, (B.shape))
        B_i[:] = I
        assert B_i == B_test

        # generating the interpolater
        N_range = np.linspace(-1, 1, D)
        grid_image = scipy.interpolate.RegularGridInterpolator(
            (N_range, N_range, N_range), B_i, bounds_error=False, fill_value=0)

        # rotating the molecule
        x, y, z = np.meshgrid(N_range, N_range, N_range)

        C = [x.flatten(), y.flatten(), z.flatten()]

        B_i = grid_image(np.dot(R.T, C).T)
        B += B_i

    P = np.fft.fftn(np.fft.fftshift(B))

    def transform_coordinates(v):
        """(3,) vector of indices -> (x, y, z) tuple of coordinates"""
        i, j, k = v[0], v[1], v[2]
        return np.array([- (D - 1) / 2 + j, (D - 1) / 2 + i, - (D - 1) / 2 + k])

    def calculate_H(vec):
        return np.sum([np.sinc(D * np.dot(vec, orientation)) for orientation in orientations])

    H = np.indices(B.shape)

    vectorized_transform_coordinates = np.vectorize(transform_coordinates)
    vectorized_calculate_H = np.vectorize(calculate_H)

    H = vectorized_calculate_H(vectorized_transform_coordinates(H))

    P = B / H

    P = np.fft.ifftn(np.fft.ifftshift(P))

    return P
