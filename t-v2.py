from MRCFile import MRCFile
import vanheel
import numpy as np
import copy
import scipy.interpolate
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm


'''mol is a 3d array, contains information of sample points of that macromolecular
where mol[h][k][l] gives the electron density at that particular point.
R is a 3 by 3 rotation matrix that gives the viewing angle of camera (actually is a
chang of basis matrix for the molecule. '''
def project_fst(mol, R):
    # generating 3D fourier transform
    F = np.fft.fftn(mol)
    # shifting the index of fourier transform
    F = np.fft.fftshift(F)
    # generating the interpolater
    l = len(mol)
    x = np.linspace(0, l -1, l)
    gri = scipy.interpolate.RegularGridInterpolator((x,x,x), F)

    #generating my output 2D data
    #The output is an image, which is a function of (x, y)
    #the size of image is 2*the original length
    I = np.zeros(4*l*l)
    I = np.reshape(I, (2*l, 2*l))
    for i in xrange(0, 2*l):
        for j in xrange(0, 2*l):
            # in the image, (i = 0, j = 0) grid is the lower bottom
            # grid, shifting the indexes to normal (x, y) coordinates
            temp = [-(2*l - 1)/2 + j, (2*l - 1)/2 - i, 0]
            # Find the cooresponding unrotated voxel
            temp = np.dot(R.T, temp)
            # if the unrotated voxel is in the mol
            # interpolate the value of that voxel in unrotated 3D frequency domain
            # and set it as the value for I[i, j]
            if np.all(np.less(np.abs(temp), [(l-1)/2, (l-1)/2, (l-1)/2])):
                temp = [(l - 1)/2 - temp[1], (l - 1)/2 + temp[0], (l-1)/2 - temp[2]]
                tempval = gri(temp)
                I[i][j] = tempval[0]
    # obtain the slice across origin, then perform inverse fourier transform
    I = np.fft.ifftshift(I)
    I = np.fft.ifft2(I)
    #draw_image(I)
    return I

''' draw the image as a 2d map. assuming the images are produced
by above projection_fst.'''
def draw_image(I):
    l = len(I)
    # drawing the output image
    x = np.linspace(0, l - 1, l)
    X,Y = np.meshgrid(x, x)
    plt.pcolormesh(X, Y, I, cmap = cm.gray)
    plt.show()

'''produce the projection image of zika_193 mol
in the rotated a degree about x axis,
b degree about y axis, c degree about z axis.'''
def produce_image(a, b, c):
    f = MRCFile('zika_153.mrc')
    f.load_all_slices()
    #ar stands for degree in radian
    ar = a*math.pi/180
    br = b*math.pi/180
    cr = c*math.pi/180
    Ra = [[1, 0, 0], [0, math.cos(ar), math.sin(ar)], [0, -math.sin(ar), math.cos(ar)]]
    Rb = [[math.cos(br), 0, -math.sin(br)], [0, 1, 0], [math.sin(br), 0, math.cos(br)]]
    Rc = [[math.cos(cr), math.sin(cr), 0], [-math.sin(cr), math.cos(cr), 0], [0, 0, 1]]
    R = np.dot(np.dot(Ra, Rb), Rc)
    I = project_fst(f.slices, R)
    return I, R


'''produce n images of mol of different orientations.
The first image is alway taken of the unrotated mol.'''
def produce_random_images(n):
    images = []
    orientations = []
    angles = []
    image, orientation = produce_image(0, 0, 0)
    images.append(image)
    orientations.append(orientation)
    angles.append([0, 0, 0])
    for i in xrange(1, n):
        a = np.random.randint(0, 180)
        b = np.random.randint(0, 180)
        c = np.random.randint(0, 180)
        image, orientation = produce_image(a, b, c)
        images.append(image)
        orientations.append(orientation)
        angles.append([a, b, c])
    return images, orientations, angles
