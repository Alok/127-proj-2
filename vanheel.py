"""vanheel.py - recover microscope orientations using common lines and the van Heel algorithm.

    Definition: Suppose F_i and F_j are the microscope orientations corresponding to images i and j. Then, the 2D vector l[i,j] is a common line if

       np.dot(F_i, l[i,j]) = np.dot(F_j, l[j,i])

    Remark: Common lines are *oriented*. The orientation of the line is taken as the direction the vector l[i,j] is pointing. You can detect the orientation of a pair of common lines by seeing in which direction the lines correlate.

    We assume that the common lines l for N images are an NxNx2 ndarray.

    Usage:

        F = estimate_orientations( l )
"""
import numpy as np


def rotate_cc_about_vector(vector, theta):
    """Return the 3x3 matrix that rotates 3D space by an angle theta about vector in the counter-clockwise direction, looking down along the vector towards the origin."""

    assert (vector.ndim == 1 and vector.shape[0] == 3)

    # We construct the matrix by rotating vector to the x-axis, then rotating
    # about x, then undoing the initial rotation

    e_1, e_2, e_3 = np.eye(3)[:, 0], np.eye(3)[:, 1], np.eye(3)[:, 2]

    f_1 = vector / np.linalg.norm(vector)

    f_2 = np.cross(f_1, np.random.random(3))
    f_2 = f_2 / np.linalg.norm(f_2)

    f_3 = np.cross(f_1, f_2)
    f_3 = f_3 / np.linalg.norm(f_3)

    # Note that this all to np.array() returns a 3x3 array with f_1, f_2, f_3
    # as ROWS, e.g., S is the linear transformation that maps f_1 -> e_1, f_2
    # -> e_2, f_3 -> e_3
    S = np.array([f_1, f_2, f_3])

    # Now S*vector is the x-axis, so we can perform the desired rotation by
    # theta

    R = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])

    return np.dot(S.T, np.dot(R, S))


def is_same_line(l_1, l_2):
    """return true if span l_1 == span l_2"""
    l_1 = l_1 / np.linalg.norm(l_1)
    l_2 = l_2 / np.linalg.norm(l_2)

    return (np.allclose(l_1, l_2) or np.allclose(-l_1, l_2))


def compute_plane_angles_from_common_lines(l, i, j, k):
    """Compute the dihedral angles between planes P_i, P_j and P_k from the common lines data l"""
    alpha = np.arccos(np.abs(np.dot(l[i, j], l[i, k])))
    beta = np.arccos(np.abs(np.dot(l[j, i], l[j, k])))
    gamma = np.arccos(np.abs(np.dot(l[k, i], l[k, j])))

    cos_alpha = np.cos(alpha)
    cos_beta = np.cos(beta)
    cos_gamma = np.cos(gamma)

    sin_alpha = np.sin(alpha)
    sin_beta = np.sin(beta)
    sin_gamma = np.sin(gamma)

    psi_ij = np.arccos((cos_gamma - cos_alpha * cos_beta) /
                       (sin_alpha * sin_beta))
    psi_ik = np.arccos((cos_beta - cos_alpha * cos_gamma) /
                       (sin_alpha * sin_gamma))
    psi_jk = np.arccos((cos_alpha - cos_beta * cos_gamma) /
                       (sin_beta * sin_gamma))

    if psi_ij > np.pi / 2:
        psi_ij = np.pi - psi_ij

    if psi_ik > np.pi / 2:
        psi_ik = np.pi - psi_ik

    if psi_jk > np.pi / 2:
        psi_jk = np.pi - psi_jk

    return psi_ij, psi_ik, psi_jk


def random_rotation_matrix():
    e_1 = 2 * np.random.rand(3) - 1
    e_1 /= np.linalg.norm(e_1)

    e_2 = np.cross(e_1, 2 * np.random.rand(3) - 1)
    e_2 /= np.linalg.norm(e_2)

    e_3 = np.cross(e_1, e_2)
    e_3 /= np.linalg.norm(e_3)

    return np.array([e_1, e_2, e_3]).T


def compute_common_lines(F):
    N = len(F)
    l = np.zeros((N, N, 2))

    for i in range(N):
        for j in range(i + 1, N):
            V_ij = np.cross(F[i][:, 2], F[j][:, 2])
            V_ij /= np.linalg.norm(V_ij)

            l[i, j] = np.dot((F[i][:, 0:2]).T, V_ij)
            l[j, i] = np.dot((F[j][:, 0:2]).T, V_ij)

    return l


def estimate_orientations_3(l):
    psi_12, psi_13, psi_23 = compute_plane_angles_from_common_lines(l, 0, 1, 2)

    F_1 = np.eye(3)

    v_12 = l[0, 1]
    v_12 /= np.linalg.norm(v_12)

    w_12 = np.array([-v_12[1], v_12[0]])

    v_21 = l[1, 0]
    v_21 /= np.linalg.norm(v_21)

    w_21 = np.array([-v_21[1], v_21[0]])

    v_13 = l[0, 2]
    v_13 /= np.linalg.norm(v_13)

    w_13 = np.array([-v_13[1], v_13[0]])

    v_31 = l[2, 0]
    v_31 /= np.linalg.norm(v_31)

    w_31 = np.array([-v_31[1], v_31[0]])

    V_12 = np.hstack([v_12, 0])

    W_12 = np.cross([0, 0, 1], V_12)

    V_13 = np.hstack([v_13, 0])

    W_13 = np.cross([0, 0, 1], V_13)

    R_2 = np.array([V_12, W_12]).T
    R_3 = np.array([V_13, W_13]).T

    assert np.allclose(np.dot(R_2.T, R_2), np.eye(2))
    assert np.allclose(np.dot(R_3.T, R_3), np.eye(2))

    v_23 = l[1, 2]
    v_23 /= np.linalg.norm(v_23)

    v_32 = l[2, 1]
    v_32 /= np.linalg.norm(v_32)
    F_2 = np.zeros((3, 3))
    F_3 = np.zeros((3, 3))

    for rot_2, em_2 in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        S_2 = rotate_cc_about_vector(V_12, rot_2 * psi_12)
        iota_2 = np.linalg.inv(np.array([v_21, em_2 * w_21]).T)
        F_2[:, 0:2] = np.dot(S_2, np.dot(R_2, iota_2))
        F_2[:, 2] = np.cross(F_2[:, 0], F_2[:, 1])

        assert is_same_line(
            np.dot(F_1[:, 0:2], v_12), np.dot(F_2[:, 0:2], v_21))

        for rot_3, em_3 in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            S_3 = rotate_cc_about_vector(V_13, rot_3 * psi_13)
            iota_3 = np.linalg.inv(np.array([v_31, em_3 * w_31]).T)
            F_3[:, 0:2] = np.dot(S_3, np.dot(R_3, iota_3))
            F_3[:, 2] = np.cross(F_3[:, 0], F_3[:, 1])

            assert is_same_line(
                np.dot(F_1[:, 0:2], v_13), np.dot(F_3[:, 0:2], v_31))

            if np.allclose(np.dot(F_2[:, 0:2], v_23), np.dot(F_3[:, 0:2], v_32)):
                return F_1, F_2, F_3

    assert False, "Unexpected error - it should not be possible to get here."


def estimate_orientations(l):
    """estimates N microscope orientations from common lines l

    Input:
        l NxNx2 ndarray of common lines, l[i,j] is a 2-vector representing giving the common line between planes i and j in image i

    Output:
        F list of microscope orientations
    """
    assert l.ndim is 3 and l.shape[
        -1] is 2, "Common lines must be an NxNx2 ndarray"

    N = l.shape[0]

    for i in range(N):
        for j in range(i + 1, N):
            l[i, j] /= np.linalg.norm(l[i, j])
            l[j, i] /= np.linalg.norm(l[j, i])

    F = [None] * 3

    F[0], F[1], F[2] = estimate_orientations_3(l)

    for i in range(3, N):
        # Dock plane i into the first three
        V_1i = np.dot(F[0][:, 0:2], l[0, i])
        V_2i = np.dot(F[1][:, 0:2], l[1, i])

        v_i1 = l[i, 0]
        v_i2 = l[i, 1]

        iota_i = np.linalg.inv(np.array([v_i1, v_i2]).T)
        R = np.array([V_1i, V_2i]).T
        F.append(np.dot(R, iota_i))

    return F


def test_rotate_cc_about_vector():
    p = np.array([np.sqrt(2) / 2, np.sqrt(2) / 2, np.sqrt(2) / 2])

    R = rotate_cc_about_vector(np.array([1, 0, 0]), -np.pi / 4)
    assert np.isclose(np.dot(R, p)[2], 0)

    R = rotate_cc_about_vector(np.array([0, 1, 0]), -np.pi / 4)
    assert np.isclose(np.dot(R, p)[0], 0)

    R = rotate_cc_about_vector(np.array([0, 0, 1]), -np.pi / 4)
    assert np.isclose(np.dot(R, p)[1], 0)


def test_compute_common_lines():
    N = 3

    F = []

    for i in range(N):
        F.append(random_rotation_matrix())

    l = compute_common_lines(F)

    for i in range(N):
        for j in range(i + 1, N):
            assert np.allclose( np.dot(F[i][:, 0:2], l[i, j]), np.dot(F[j][:, 0:2], l[j, i]))


def test_plane_angles():
    F = []

    for i in range(3):
        F.append(random_rotation_matrix())

    l = compute_common_lines(F)

    psi_12, psi_13, psi_23 = compute_plane_angles_from_common_lines(l, 0, 1, 2)

    assert np.isclose( np.arccos(np.abs(np.dot(F[0][:, 2], F[1][:, 2]))), psi_12)
    assert np.isclose( np.arccos(np.abs(np.dot(F[0][:, 2], F[2][:, 2]))), psi_13)
    assert np.isclose( np.arccos(np.abs(np.dot(F[1][:, 2], F[2][:, 2]))), psi_23)


def test_estimate_orientations():
    N = 100

    F = []

    for i in range(N):
        F.append(random_rotation_matrix())

    l = compute_common_lines(F)

    G = estimate_orientations(l)

    J = np.eye(3)
    J[2, 2] = -1

    R_p = np.dot(F[0], G[0].T)

    if not np.allclose(np.dot(R_p, G[1])[:, 0:2], F[1][:, 0:2]):
        R_p = np.dot(F[0], np.dot(J, G[0].T))
        assert np.allclose(np.dot(R_p, G[1])[:, 0:2], F[1][:, 0:2])

    for i in range(N):
        assert np.allclose(np.dot(R_p, G[i])[:, 0:2], F[i][:, 0:2])


def test_estimate_orientations_3():
    N = 3

    F = []

    for i in range(N):
        F.append(random_rotation_matrix())

    l = compute_common_lines(F)

    F_1, F_2, F_3 = estimate_orientations_3(l)

    F_1 = np.dot(F_1.T, F_1)
    F_2 = np.dot(F_1.T, F_2)
    F_3 = np.dot(F_1.T, F_3)

    G_1 = np.dot(F[0].T, F[0])
    G_2 = np.dot(F[0].T, F[1])
    G_3 = np.dot(F[0].T, F[2])

    J = np.eye(3)
    J[2, 2] = -1

    assert np.allclose(F_1[:, 0:2], G_1[:, 0:2]) or np.allclose( F_1[:, 0:2], np.dot(J, G_1[:, 0:2]))
    assert np.allclose(F_2[:, 0:2], G_2[:, 0:2]) or np.allclose( F_2[:, 0:2], np.dot(J, G_2[:, 0:2]))
    assert np.allclose(F_3[:, 0:2], G_3[:, 0:2]) or np.allclose( F_3[:, 0:2], np.dot(J, G_3[:, 0:2]))

