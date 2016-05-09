#!/usr/bin/env python

import math
import sys
import copy
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import vanheel
from mpl_toolkits.mplot3d import Axes3D
from MRCFile import MRCFile


def project_fst(mol, R):
    """
    mol: is a 3d array where mol[h][k][l] gives the electron density at that pt

    R: 3x3 rotation matrix that gives the viewing angle of camera
    (actually is a change of basis matrix for the molecule).
    """

    # generating 3D fourier transform
    dft_mol = np.fft.fftn(mol)

    # shifting the index of fourier transform
    F = np.fft.fftshift(dft_mol)

    # Size of image is 2 * (the original length)
    l = len(mol)
    N = 2 * l

    # generating the interpolater
    grid_step = np.linspace(0, l - 1, l)
    interpolated_grid = scipy.interpolate.RegularGridInterpolator((grid_step, grid_step, grid_step), F)

    # generating output 2D data
    # Output is 1 image, which is a function of (x, y)
    I = np.zeros(N * N, dtype= 'complex')
    I = np.reshape(I, (N, N))

    for i in xrange(l):
        for j in xrange(l):
            # in the image, (i = 0, j = 0) grid is the lower bottom
            # grid, shifting the indexes to normal (x, y) coordinates
            temp = [-(N - 1) / 2 + j,  (N - 1) / 2 - i, 0]

            # Find the cooresponding unrotated voxel
            # transpose shifts from [ [a] ,
            #                        [b] ,
            #                        [c] ]
            # to the correct rotation
            temp = np.dot(R.T, temp)
            x = temp[0]
            y = temp[1]
            z = temp[2]

            # if the unrotated voxel is in the mol
            # interpolate the value of that voxel in unrotated 3D frequency domain
            # and set it as the value for I[i, j]
            # if np.all(np.less(np.abs(temp), [(l - 1) / 2, (l - 1) / 2, (l - 1) / 2] * 3)):
            if np.all(np.less(np.abs(temp), [(l - 1) / 2] * 3)):

                # temp = [(l - 1) / 2 - y, (l - 1) / 2 + x, (l - 1) / 2 - z]
                temp = [(l - 1) / 2 - y, (l - 1) / 2 + x, 0]

                # temp = [(l - 1) / 2 - temp[0], (l - 1) / 2 - temp[1], (l - 1) / 2 - temp[2]]
                tempval = interpolated_grid(temp)
                # get real part of tempval
                I[i][j] = tempval[0]

            else:
                # if the unrotated voxel is not outs of bound of mol
                I[i][j] = 0

    # obtain the slice across origin, then perform inverse fourier transform
    I = np.fft.fftshift(I)
    I = np.fft.ifft2(I)
    return I


def draw_image(I, save= True, img_name='z-', show = False):
    """
    Draw the image as a 2d map, assuming the images are produced
    as above by project_fst.
    """

    l = len(I)
    # drawing the output image
    x = np.linspace(0, l - 1, l)
    X, Y = np.meshgrid(x, x)
    img = plt.pcolormesh(X, Y, I, cmap='gray')
    if save:
        plt.savefig('./fig/' + img_name + '.png')
    if show:
        plt.show()
    return img


def produce_projection_image(a=0, b=0, c=0, mrc_file='./zika_192.mrc'):
    """
    mrc_file: string of MRC file name
    a: rotation about x-axis (in degrees)
    b: rotation about y-axis (in degrees)
    c: rotation about z-axis (in degrees)
    """
    cos = math.cos
    sin = math.sin

    f = MRCFile(mrc_file)
    f.load_all_slices()
    # ar stands for degree in radian
    ar = a * math.pi / 180
    br = b * math.pi / 180
    cr = c * math.pi / 180

    Ra = [[1,        0,        0],
          [0,        cos(ar),  -sin(ar)],
          [0,        sin(ar),  cos(ar)]]

    Rb = [[cos(br),  0,        sin(br)],
          [0,        1,        0],
          [-sin(br), 0,        cos(br)]]

    Rc = [[cos(cr),  -sin(cr), 0],
          [sin(cr),  cos(cr),  0],
          [0,        0,        1]]

    R = np.dot(np.dot(Ra, Rb), Rc)
    I = project_fst(f.slices, R)
    return I, R

def produce_random_images(n):
    """
    Produce n images of mol of different orientations.
    The first image is always of the unrotated mol.
    """

    images = []
    orientations = []
    A = []
    image, orientation = produce_projection_image(0, 0, 0)
    images.append(image)
    orientations.append(orientation)
    A.append([0, 0, 0])
    for _ in xrange(1, n):
        a = np.random.randint(0, 180)
        b = np.random.randint(0, 180)
        c = np.random.randint(0, 180)
        image, orientation = produce_projection_image(a, b, c)
        images.append(image)
        orientations.append(orientation)
        A.append([a, b, c])
    return images, orientations, A


if sys.argv[1] == 'x':
    for i in range(91,181):
        name = 'x/{}'.format(i)
        a_1 = i
        b_1 = int(sys.argv[2])
        c_1 = int(sys.argv[3])
        arg = produce_projection_image(a_1, b_1, c_1)[0]
        draw_image(arg, img_name = name)


elif sys.argv[1] == 'y':
    for i in range(91):
        name = 'y/{}'.format(i)
        a_1 = int(sys.argv[2])
        b_1 = i
        c_1 = int(sys.argv[3])
        arg = produce_projection_image(a_1, b_1, c_1)[0]
        draw_image(arg, img_name = name)

elif sys.argv[1] == 'z':
    for i in range(91):
        name = 'z/{}'.format(i)
        a_1 = int(sys.argv[2])
        b_1 = int(sys.argv[3])
        c_1 = i
        arg = produce_projection_image(a_1, b_1, c_1)[0]
        draw_image(arg, img_name = name)
else:
    draw_image(produce_projection_image(a=int(sys.argv[1]),b=int(sys.argv[2]),c=int(sys.argv[3]))[0], save=False, show =True)
