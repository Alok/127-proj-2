#!/usr/bin/env python3
# encoding: utf-8

from helper import *

"""
Using the projection slice theorem, write a function project_fst(mol, R) that
simulates an electron microscopy image of the molecule mol from the microscope
viewing direction given by the rotation matrix R.
"""


def restrict_to_2D(mat):
    return mat[:2, :2]

# R is F
def project_fst(mol, R):
    # compute 3d dft of mol
    dft_mol = [dft_3D(three_dim_arr) for three_dim_arr in mol]
    slice = [restrict_to_2D(three_dim_arr) for three_dim_arr in dft_mol]
    I_f = # inv fourier of slice

# TODO interpolation
