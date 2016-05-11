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
from numpy.fft import fftn
from numpy.fft import fftshift
from numpy.fft import ifft2
from numpy import cos
from numpy import sin
from numpy import sqrt

def project_fst(mol, R):
    N = mol.shape[0]
    N_range = np.linspace(-1, 1, N)
    rgi = scipy.interpolate.RegularGridInterpolator((N_range, N_range, N_range), mol, bounds_error=False, fill_value=0)
    # generating 3D fourier transform
    x, y, z = np.meshgrid(N_range, N_range, N_range)
    A = [x.flatten(), y.flatten(), z.flatten()]
    mol_hat = rgi(np.dot(R, A).T).reshape(N, N, N)
    return np.real(ifft2(fftshift(fftn(mol_hat))[:,:,0]))



def draw_image(I, save= True, img_name='z-', show = False):
    """
    Draw the image as a 2d map, assuming the images are produced
    as above by project_fst.
    """

    D = len(I)
    x = np.linspace(-1, 1, D)
    X, Y = np.meshgrid(x, x)
    img = plt.pcolormesh(X, Y, I)
    if save:
        plt.savefig('./fig/' + img_name + '.png')
    if show:
        plt.show()
    return img


def produce_projection_image(a=0, b=0, c=0, mrc_file='./zika_153.mrc'):
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

def produce_random_images(n, filename = 'zika_153.mrc'):
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
        rand = np.random.randint
        a, b, c = rand(0, 180), rand(0, 180), rand(0, 180)

        image, orientation = produce_projection_image(a, b, c, filename)
        images.append(image)
        orientations.append(orientation)
        A.append([a, b, c])
    return images, orientations, A
