import numpy as np
import scipy.ndimage as nd
import skimage.segmentation as seg
import skimage.morphology as morph

def HoCS(B: np.ndarray, min_scale: int, max_scale: int, increment: int, num_bins: int) -> np.ndarray:
    '''
    Computes a histogram of curvature scale for the shape in the binary image B.  
    Boundary fragments due to holes are ignored.
    :param B: A binary image consisting of a single foreground connected component.
    :param min_scale: smallest scale to consider (minimum 1)
    :param max_scale: largest scale to consider (max_scale > min_scale)
    :param increment:  increment on which to compute scales between min_scale and max_scale
    :param num_bins: number of bins for the histogram at each scale
    :return: 1D array of histograms concatenated together in order of increasing scale.
    
    Note: a curvature values are between 0.0 and 1.0 are interpreted thusly:
       0.0 - values close to 0 arise from highly convex points.
       0.5 - no curvature
       1.0 - values close to 1 arise from highly concave points.
    '''

    if max_scale < min_scale:
        raise ValueError('max_scale must be larger than min_scale')
    if num_bins < 1:
        raise ValueError('num_bins must be >= 1')
    
    b_holes_filled = nd.binary_fill_holes(B)
    boundary_points = seg.find_boundaries(b_holes_filled, connectivity=1, mode='inner') 
    boundary_points = np.where(boundary_points > 0)
    boundary_points = np.transpose(np.vstack(boundary_points))
    
    histograms = list()
    for radius in np.arange(min_scale, max_scale + 1, increment):

        disk = morph.disk(radius) 
        disk = disk / np.sum(disk)
        
        convolved = nd.filters.convolve(util.img_as_float(b_holes_filled), disk)
        curvatures = convolved[boundary_points[ : , 0], boundary_points[ : , 1]]
        
        h, bin_edges = np.histogram(curvatures, bins=num_bins, range=(0.0, 1.0))
        histograms.append(h / len(curvatures))
        
    return np.hstack(histograms)

# Test your HoCS function

import skimage.io as io
import skimage.util as util
import matplotlib.pyplot as plt 

img = util.img_as_float(io.imread('./leaftraining/threshimage_0001.png'))
img_hist = HoCS(img, 5, 25, 10, 10)
plt.bar(np.arange(len(img_hist)), img_hist)

# Calculate training features

import os as os
from tqdm import tqdm

def get_features(path: str) -> tuple:
    features = list()
    for root, dirs, files in os.walk(path):
        for f in tqdm(sorted(files)):
            if f[-4 : ] == '.png':
                #print('Calculating features for', f)
                B = io.imread(os.path.join(root, f))
                f = HoCS(B, min_scale=5, max_scale=30, increment=5, num_bins=15)
                features.append(f)
    return np.vstack(features), sorted(files) 

train_labels = np.zeros(30, dtype='int')
train_labels[10:20] = 1
train_labels[20:] = 2

train_data, train_files = get_features(os.path.join('.', 'leaftraining'))
np.savetxt("train_data.csv", train_data, delimiter=",")

# Prepare the testing data

test_labels = np.zeros(129, dtype = 'int')
test_labels[50:77] = 1
test_labels[77:] = 2

test_data, test_files = get_features(os.path.join('.', 'leaftesting'))
np.savetxt("testing_data.csv", test_data, delimiter=",")

# Dimensionality reduction using PCA and t-SNE
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

reductor_pca = PCA(n_components = 3)
train_data_pca = reductor_pca.fit_transform(train_data)
reductor_pca.explained_variance_ratio_

train_data_pca = pd.DataFrame(data = train_data_pca, columns = ['PC-1', 'PC-2', 'PC-3'])
plt.figure(figsize=(9,6))
sns.scatterplot(
    data = train_data_pca, 
    x = 'PC-1', y = 'PC-2',
    hue=train_labels
)

# Train the KNN classifier using the feature vectors from the training images

#from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
#from sklearn.gaussian_process import GaussianProcessClassifier
#from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.cluster import KMeans

classifiers = [
    KNeighborsClassifier(n_neighbors=3),
    SVC(kernel="linear", C=0.025),
    #SVC(gamma=2, C=1),
    #GaussianProcessClassifier(1.0 * RBF(1.0)),
    #GaussianNB(),
    #DecisionTreeClassifier(),
    #RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    #MLPClassifier(alpha=1, max_iter=1000),
    #AdaBoostClassifier(),
    #QuadraticDiscriminantAnalysis(),
    KMeans(n_clusters=3, init=np.take(train_data, [1, 11, 21], axis = 0))]



# Classfiy the testing features

for clf in classifiers:
    clf.fit(train_data, train_labels)
    predict_labels = clf.predict(test_data)
    correct_labels = (predict_labels == test_labels)
    
    # Compute and print out the classification rate.
    correct_rate = np.sum(correct_labels) / len(predict_labels)
    print('The classification rate was', correct_rate * 100, 'percent.')

    # obtain the filenames of images that were incorrectly classified.
    incorrectly_classified = \
        [test_files[i] for i in range(len(test_files)) if not correct_labels[i]]

    # Print out the names of incorrectly classified images.
    for f in incorrectly_classified:
        print(f, 'was incorrectly classified.')
    print()  
    
    # Compute and print out the confusion matrix.
    confusion = np.zeros((3, 3), dtype = 'int')
    
    for i in range(len(predict_labels)):
        confusion[test_labels[i] - 1, predict_labels[i] - 1] += 1
        
    print('The confusion matrix is:')
    for x in confusion:
        print('{:5}, {:5}, {:5}'.format(x[0], x[1], x[2]))
    print()
