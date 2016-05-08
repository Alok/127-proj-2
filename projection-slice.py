#!/usr/bin/env python

from helper import *
from MRCFile import MRCFile

"""
Using the projection slice theorem, write a function project_fst(mol, R) that
simulates an electron microscopy image of the molecule mol from the microscope
viewing direction given by the rotation matrix R.
"""


def restrict_to_2D(mat):
    return mat[:, :, 0]

# R is F
def project_fst(mol, R):
    # dft_mol = [dft_3D(three_dim_arr) for three_dim_arr in mol]
    # compute 3d dft of mol
    dft_mol = dft_3D(mol)
    slice_2D = restrict_to_2D(dft_mol)
    I_f = inv_2D(slice_2D)
    return I_f

c = np.vectorize(isinstance)

# TODO interpolation



# Setup file object
f = MRCFile('zika_153.mrc')

# Actually load volume slices from disk
f.load_all_slices()
a = project_fst(f.slices, None)
print [x for x in a if not isinstance(x, complex)]
plt.imshow(a)
