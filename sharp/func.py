import numpy as np 
import math as math
import matplotlib.pyplot as plt 
import skimage.io as io 
import skimage.util as util 
import skimage.filters as filt
import scipy.stats as stats 

from typing import Tuple

def color_dot_product(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    '''
    Element-by-element dot product in a 2D array of vectors.

    :param A: Input color image A
    :param B: Input color image B 
    :return: An array in which index [i,j,:] is the dot product of A[i,j,:] and B[i,j,:].
    '''
    return np.sum(A.conj() * B, axis = 2)


def color_sobel_edges(I: np.ndarray) -> Tuple[np.ndarray, float]:
    '''
    Sobel vector gradient for color images.
    
    :param I: Input image.
    :return: A 2-tuple which the 1st entry is a 2-d array containing the gradient magnitudes for each pixel,
    the 2nd entry contains the gradient directions for each pixel. 
    '''

    chan_R = I[:, :, 0]
    chan_G = I[:, :, 1]
    chan_B = I[:, :, 2]

    horizon_R = filt.sobel_h(chan_R)
    horizon_G = filt.sobel_h(chan_G)
    horizon_B = filt.sobel_h(chan_B)

    vertical_R = filt.sobel_v(chan_R)
    vertical_G = filt.sobel_v(chan_G)
    vertical_B = filt.sobel_v(chan_B)

    g_x = np.dstack((horizon_R, horizon_G, horizon_B))
    g_y = np.dstack((vertical_R, vertical_G, vertical_B))

    g_xx = color_dot_product(g_x, g_x)
    g_yy = color_dot_product(g_y, g_y)
    g_xy = color_dot_product(g_x, g_y)

    grad_direction_x2 = np.arctan2(2 * g_xy, g_xx - g_yy)

    grad_magnitude = np.sqrt(((g_xx + g_yy) + \
                              (g_xx - g_yy) * np.cos(grad_direction_x2) + \
                              2 * g_xy * np.sin(grad_direction_x2)) / 2)

    return (grad_magnitude, grad_direction_x2 / 2)

def kurtosis_sharpness(I: np.ndarray) -> float:
    '''
    Compute the kurtosis-based sharpness measure.

    :param I: Input image.
    :return: A numeric value which is the kurtosis-based sharpness of input image I. 
    '''
    I_grad_magnitude = color_sobel_edges(I)[0]
    k = stats.kurtosis(I_grad_magnitude.flatten())
    return math.log(k + 3)

def test_blur_measure(I: np.ndarray, min_sigma: float, max_sigma: float) -> np.ndarray:
    '''
    Apply different amounts of Gaussian blur to an image. 
    
    :param I: Input image.
    :param min_sigma: The minimum standard deviation for Gaussian Kernel. 
    :param max_sigma: The maximum standard deviation for Gaussian Kernel.
    :return: A 2-d array which the 1st column consisting of the sigma values used and 
    the 2nd column consisting of computed kurtosis values for each sigma.
    '''
    i: int = math.ceil(min_sigma)
    j: int = math.floor(max_sigma) + 1
    result = np.zeros((j - i, 2))
    for s in range(i, j):
        I_blur = filt.gaussian(I, sigma = s, multichannel = True)
        result[s - i, :] = [s, kurtosis_sharpness(I_blur)]
    return result


def sharpness_map(I: np.ndarray, window_size: int) -> np.ndarray:
    '''
    Compute the local sharpnesses map for the given window size.

    :param I: Input image.
    :param window_size: Size of window (in pixels). 
    :return: A 2-d array of local sharpnesses.
    '''
    s_map = np.zeros((np.size(I, 0) // window_size,
                      np.size(I, 1) // window_size))
    for r in range(0, np.size(s_map, 0)):
        for c in range(0, np.size(s_map, 1)):
            s_map[r, c] = kurtosis_sharpness(I[r * window_size: (r + 1) * window_size - 1, \
                                               c * window_size: (c + 1) * window_size - 1])
    return s_map
