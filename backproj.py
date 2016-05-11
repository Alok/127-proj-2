#!/usr/bin/env python

from project_fst import *
#
#
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
    B = np.zeros(D**3)
    B = np.reshape(B, (D, D, D))

    # For rest of images, calculate the back projection
    for i in xrange(len(images)):
        I = images[i]
        R = orientations[i]
        B_i = np.zeros((D, D, D))

        # tile matrices
        # for j in xrange(D):
            # B_i[:,:,j] = I

        B_i[:] = I
        # B_test = np.zeros((D,D,D))
        # B_test[:] = I

        # print B_test[0][2][4]
        # print B_i[2][4][0]

        # generating the interpolater
        N_range = np.linspace(-1, 1, D)
        grid_image = scipy.interpolate.RegularGridInterpolator(
            (N_range, N_range, N_range), B_i, bounds_error=False, fill_value=0)

        # rotating the molecule
        x, y, z = np.meshgrid(N_range, N_range, N_range)

        C = [x.flatten(), y.flatten(), z.flatten()]

        B_i = grid_image(np.dot(R.T, C).T).reshape((D, D, D))
        B += B + B_i

    P = np.fft.fftn(np.fft.fftshift(B))

    def transform_coordinates(v):
        """(3,) vector of indices -> (x, y, z) tuple of coordinates"""
        i, j, k = v[0], v[1], v[2]
        return np.array([- (D - 1) / 2 + j, (D - 1) / 2 + i, - (D - 1) / 2 + k])

    def calculate_H(vec):
        return np.sum([np.sinc(D * np.dot(vec, orientation)) for orientation in orientations])
    def create_coord_matrix():
        mat = np.zeros(B.shape)
        for i_p in xrange(D):
            for j_p in xrange(D):
                for k_p in xrange(D):
                    mat[i_p,j_p,k_p] = [i_p, j_p, k_p]

    # H = create_coord_matrix()
    # print H.shape
    # print H[0][0][0]
    # print ("H initialized")
    #
    #
    # vectorized_transform_coordinates = np.vectorize(transform_coordinates)
    # vectorized_calculate_H = np.vectorize(calculate_H)
    #
    # H = vectorized_calculate_H(vectorized_transform_coordinates(H))
    #
    # P = P / H
    #
    for i in xrange(D):
        for j in xrange(D):
            for k in xrange(D):

                H = 0
                temp = np.asarray( [- (D - 1) / 2 + j, (D - 1) / 2 + i, - (D - 1) / 2 + k])

                for i_p in xrange(0, len(images)):
                    val = np.dot(temp.T, orientations[i_p][2])
                    H = H + (math.sin(D * (math.pi) * val) / math.pi * val)

                P[i][j][k] = P[i][j][k] * (1 / H)

    P = np.real(np.fft.ifftn(np.fft.ifftshift(P)))

    return P

imgs, viewing_angles, _ = produce_random_images(2, 'zika_153.mrc')
array_data = reconstruct(imgs, viewing_angles)

g = MRCFile('zika_153.mrc')
g.slices = array_data
g.write_file('backproj_test.mrc', overwrite=True)
