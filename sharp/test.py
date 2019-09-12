import numpy as np 
import math as math
import matplotlib.pyplot as plt 
import skimage.io as io 
import skimage.util as util 
import skimage.filters as filt
import scipy.stats as stats 

def color_dot_product(A, B):
    '''
    Element-by-element dot product in a 2D array of vectors.

    :return: An array in which index [i,j,:] is the dot product of A[i,j,:] and B[i,j,:].
    '''
    return np.sum(A.conj()*B, axis=2)

def color_sobel_edges(I):
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

    #grad_direction = np.arctan(2 * g_xy / (g_xx - g_yy)) /2
    #s1 = g_xx - g_yy
    #s2 = np.divide(g_xy, s1, out=np.zeros_like(g_xy), where=s1!=0)
    #s2 = np.arctan2(g_xy, s1)
    g_xx_m_yy = g_xx - g_yy 
    grad_direction_x2 = np.arctan2(2 * g_xy, g_xx_m_yy)

    grad_magnitude = np.sqrt(((g_xx+g_yy) + g_xx_m_yy*np.cos(grad_direction_x2) + 2*g_xy*np.sin(grad_direction_x2)) /2) 

    return(grad_magnitude, grad_direction)

def kurtosis_sharpness(I):
    I_grad_magnitude = color_sobel_edges(I)[0]
    k = stats.kurtosis(I_grad_magnitude.flatten())
    return math.log(k + 3)

def test_blur_measure(I, min_sigma, max_sigma):
    i = math.ceil(min_sigma)
    j = math.floor(max_sigma) + 1
    result = np.zeros((j-i, 2))
    for s in range(i, j):
        I_blur = filt.gaussian(I, sigma = s, multichannel = True)
        result[s-i, :] = [s, kurtosis_sharpness(I_blur)]
    return result

def sharpness_map(I, window_size):
    s_map = np.zeros((np.size(I, 0) // window_size, np.size(I, 1) // window_size)) 
    for r in range(0, np.size(s_map, 0)):
        for c in range(0, np.size(s_map, 1)):
            s_map[r, c] = kurtosis_sharpness(I[r*window_size:(r+1)*window_size-1, c*window_size:(c+1)*window_size-1])
    return s_map

img = io.imread('./waterfall.jpg')
img_float = util.img_as_float(img)
#img_filt = img_float[:, :, 0]
#print(img_float[0,0])
#plt.imshow(img_filt)
#plt.show()
#color_sobel_edges(img_float)
#x = test_blur_measure(img_float, 1, 30)
#plt.plot(x[:,0], x[:,1])
#plt.show()
#print(x)
s = sharpness_map(img_float, 100)
plt.imshow(s)
plt.colorbar()
plt.show()
