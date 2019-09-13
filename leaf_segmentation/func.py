import skimage.util as util
import skimage.io as io
import skimage.morphology as morph
import skimage.segmentation as seg
import numpy as np
import gc as gc

from typing import Callable

def segLeaf(I: np.ndarray) -> np.ndarray:
    '''
    Segment a leaf image.
    :param I: Color leaf image to segment.
    :return: Logical image where True pixels represent foreground (i.e. leaf pixels).
    '''
    I_t = threshold_global(I, dist_metric = dist_green, second_pass = True)
    leaf_rough = max_component(morph.remove_small_objects(I_t)) 
    return morph.remove_small_holes(leaf_rough) 

def max_component(I: np.ndarray, connectivity: int = 1) -> np.ndarray:
    '''
    Find the major component of an image.
    :param I: Input image.
    :return: Logical image where True pixels represent major component.
    '''
    label = morph.label(I, connectivity = connectivity)
    label_freq = np.bincount(label.flatten())
    largest_label = np.argmax(label_freq[1:]) + 1   # excluding label 0
    return label == largest_label   

def threshold_global(I: np.ndarray, dist_metric: Callable[[np.ndarray], float], second_pass: bool = False) -> np.ndarray:
    '''
    Segmentation by global thresholding.
    :param I: Input image.
    :param dist_metric: A function which defines the distance from one pixel to a certain point.
    :second_pass: Boolean, if True a 2-pass thresholding will be performed. May either perform better or overkill.
    :return: Logical image where True pixels are less than an auto-detected threshold in terms of user-defined metric.
    '''
    I_dist_map = dist_map(I, dist_metric = dist_green)
    T = iter_detect_t(I_dist_map)
    if second_pass:
        T = iter_detect_t(I_dist_map, part = I_dist_map < T)
    return I_dist_map < T

def dist_green(p: np.ndarray, greenness: float = 0.1) -> float:
    '''
    Distance to GREEN 
    :param p: A RGB pixel.
    :param greenness: A value between 0 and 1 which larger value defines 'purer' GREEN.
    :return: A numeric value which is the 'distance' from the input color to GREEN.
    '''    
    return max(p[0] - p[1] * (1 - greenness), 0) + max(p[2] - p[1] * (1 - greenness), 0)

def dist_map(I: np.ndarray, dist_metric: Callable[[np.ndarray], float]) -> np.ndarray:
    '''
    Transform a color image to grayscale follow from user-specified metric. 
    :param I: Input color image.
    :param dist_metric: A function which defines the distance from one pixel to a certain point.
    :return: A grayscale image transformed from the input color image.  
    '''
    I_dist_map = np.zeros((np.size(I, 0), np.size(I, 1)))
    for r in range(0, np.size(I_dist_map, 0)):
        for c in range(0, np.size(I_dist_map, 1)):
            I_dist_map[r, c] = dist_metric(I[r, c])
    return I_dist_map

def iter_detect_t(I_float: np.ndarray, part: np.ndarray = None, eps: float = 0.1, init_guess: float = None) -> float:
    '''
    Detect the optimal threshold iteratively.
    :param I_float: Input grayscale image.
    :param part: A logical 2-D array of the same size as I_float that restricts the region of input image. By default every entry is True.
    :param eps: Epsilon the convergence criterion.
    :param init_guess: The initial guess of the threshold supplied by user; will take the mean of whole image if not specified.
    :return: An optimal threshold to segment the input image. 
    '''
    if part is None:
        part = np.ones_like(I_float, dtype = bool)
    I = I_float[part]
    if init_guess is None:
        init_guess = np.mean(I)
    cur_T = init_guess
    prev_T = float('inf')
    while abs(cur_T - prev_T) > eps:
        less_than_T = I < cur_T
        forground = I[less_than_T]
        background = I[np.logical_not(less_than_T)]
        prev_T = cur_T
        cur_T = (np.mean(forground) + np.mean(background)) / 2
    return cur_T

def dice_similar_coef(I_seg: np.ndarray, I_true: np.ndarray) -> float:
    '''
    Dice Similarity Coefficient (DSC)
    :param I_seg: Segmented image.
    :param I_true: Ground truth.
    :return: Computed DSC between the segmented image and the ground truth.
    '''
    return 2 * np.sum(np.logical_and(I_seg, I_true)) / (np.sum(I_seg) + np.sum(I_true))

def mean_sq_dist(I_seg: np.ndarray, I_true: np.ndarray) -> float:
    '''
    Mean Squared Distance (MSD) similarity
    :param I_seg: Segmented image.
    :param I_true: Ground truth.
    :return: Computed MSD between the segmented image and the ground truth.
    '''
    I_seg_bp = boundary_points(I_seg)
    I_true_bp = boundary_points(I_true)
    sq_dist: float = 0
    m: int = np.size(I_seg_bp, 0)
    for i in range(0, m):
        sq_dist += bp_shortest_dist(I_seg_bp[i], I_true_bp) ** 2
    return sq_dist / m


def hausdorff_dist(I_seg: np.ndarray, I_true: np.ndarray) -> float:
    '''
    Hausdorff Distance (HD) similarity
    :param I_seg: Segmented image.
    :param I_true: Ground truth.
    :return: Computed HD between the segmented image and the ground truth.
    '''
    I_seg_bp = boundary_points(I_seg)
    I_true_bp = boundary_points(I_true)
    return max(bp_max_dist_A2B(I_seg_bp, I_true_bp), \
               bp_max_dist_A2B(I_true_bp, I_seg_bp))

# The below are helper functions

def bp_shortest_dist(x: np.ndarray, points: np.ndarray) -> float:
    '''
    Helper for bp_max_dist_A2B, find the shortest distance from one point to a set of point.
    :param x: A single point.
    :param B: A set of points.
    :return: The shortest distance from x to B.
    '''
    x_mat = np.tile(x, (np.size(points, 0), 1))
    return np.min(np.linalg.norm(x_mat - points, axis = 1))

def bp_max_dist_A2B(A: np.ndarray, B: np.ndarray) -> float:
    '''
    Helper for hausdorff_dist, find max-min distance between two segmented images.
    :param A: Input image A.
    :param B: Input image B.
    :return: The max-min distance between A and B.
    '''
    max_d: float = 0
    for i in range(0, np.size(A, 0)):
        d = bp_shortest_dist(A[i], B)
        if d > max_d:
            max_d = d
    return max_d

def boundary_points(B: np.ndarray) -> np.ndarray:
    '''
    Helper for hausdorff_dist, find the boundary points of a segmented image.  
    :param B: Segmented image.
    :return: A set of boundary points.
    '''
    bp = np.where(seg.find_boundaries(B > 0, connectivity = 2, mode = 'inner') > 0)
    return np.transpose(np.vstack(bp)) 
