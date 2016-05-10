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

f = MRCFile('./zika_153.mrc')
f.load_all_slices()

F = np.eye(3)
F = np.array([ [1, 0, 0],
              [0,  sqrt(3)/2, .5],
              [0,  -.5, sqrt(3)/2] ])
img = project_fst(f.slices, F)

def draw_image(I, save= True, img_name='z-', show = False):
    """
    Draw the image as a 2d map, assuming the images are produced
    as above by project_fst.
    """

    l = len(I)
    # drawing the output image
    # x = np.linspace(0, l - 1, l)
    x = np.linspace(-1, 1, l)
    X, Y = np.meshgrid(x, x)
    img = plt.pcolormesh(X, Y, I)
    if save:
        plt.savefig('./fig/' + img_name + '.png')
    if show:
        plt.show()
    return img

draw_image(img, save = False, show = True)
