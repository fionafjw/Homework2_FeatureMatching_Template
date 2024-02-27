import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, feature, img_as_int
from skimage.measure import regionprops
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter

from scipy.spatial.distance import cdist

import math

def plot_feature_points(image, x, y):
    '''
    Plot feature points for the input image. 
    
    Show the feature points given on the input image. Be sure to add the images you make to your writeup. 
    Useful functions: Some helpful (not necessarily required) functions may include
        - matplotlib.pyplot.imshow, matplotlib.pyplot.scatter, matplotlib.pyplot.show, matplotlib.pyplot.savefig
    
    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates of feature points
    :y: np array of y coordinates of feature points
    '''
    plt.imshow(image, cmap="gray")
    plt.scatter(x, y, alpha=0.9, s=3)
    plt.show()

def get_feature_points(image, window_width):
    '''
    Returns a set of feature points for the input image

    (Please note that we recommend implementing this function last and using cheat_feature_points()
    to test your implementation of get_feature_descriptors() and match_features())

    Implement the Harris corner detector (See Szeliski 7.1.1) to start with.
    You do not need to worry about scale invariance or keypoint orientation estimation
    for your Harris corner detector.
    You can create additional feature point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious (false/fake) feature point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on EdStem with any questions

        - skimage.feature.peak_local_max
        - skimage.measure.regionprops


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :feature_width: the width and height of each local window in pixels

    :returns:
    :xs: an np array of the x coordinates (column indices) of the feature points in the image
    :ys: an np array of the y coordinates (row indices) of the feature points in the image

    :optional returns (may be useful for extra credit portions):
    :confidences: an np array indicating the confidence (strength) of each feature point
    :scale: an np array indicating the scale of each feature point
    :orientation: an np array indicating the orientation of each feature point

    '''


    xgrads = gaussian_filter1d(image, 2, axis=0, order=1)
    ygrads = gaussian_filter1d(image, 2, axis=1, order=1)

    # constants
    alpha = 0.05
    min_distance = 12

    # square and multiply gradients
    ix2 = xgrads ** 2
    iy2 = ygrads ** 2
    ixy = np.multiply(xgrads, ygrads)

    # perform gaussians
    gix2 = gaussian_filter(ix2, 2)
    giy2 = gaussian_filter(iy2, 2)
    gixy = gaussian_filter(ixy, 2)

    # get cornerness score
    c = np.multiply(gix2, giy2) - gixy ** 2 - alpha * (gix2 + giy2) ** 2

    # dynamic threshold
    threshold = ((np.sum(c) / c.shape[0] / c.shape[1]) * 99.5 + 0.5 * np.max(c)) / 100

    # threshold the cornerness
    thresholded = c < threshold
    c[thresholded] = 0

    # non-maxima suppression
    coords = feature.peak_local_max(c, min_distance)

    # get individual coordinate arrays
    xs = coords[:,1]
    ys = coords[:,0]
    return xs, ys

# Takes in an x,y location and returns gradient at that point in (mag, dir) form
def gradient(g, x, y):
    xgd = g[x,y,0]
    ygd = g[x,y,1]
    mag = math.sqrt(xgd * xgd + ygd * ygd)
    dir = math.atan2(ygd, xgd) % (2 * math.pi)
    return (mag, dir)

# Takes in an x,y location and returns a descriptor
def sift_descriptor(image, x, y, feature_width, grads):
    # initialize descriptor array
    descriptor = np.zeros(128)

    # initialize arrays for magnitude and direction of gradiants
    grad_array_mag = np.zeros((feature_width, feature_width))
    grad_array_dir = np.zeros((feature_width, feature_width))

    # store half feature width for readability
    half = int(feature_width / 2)

    # get magnitudes and directions for all pixels in feature
    for i in range(-half, half):
        for j in range(-half, half):
            g = gradient(grads, i + x, j + y)
            grad_array_mag[i + half, j + half] = g[0]
            grad_array_dir[i + half, j + half] = g[1]

    # loops for each sector of the feature
    for i in range(4):
        for j in range(4):

            # index of where to start this sector's bins in the descriptor array
            base_index = 8 * (4 * i + j)
            # loop over each pixel in the sector
            for k in range(int(feature_width / 4)):
                for l in range(int(feature_width / 4)):
                    # find which bin to put this gradient in
                    extra_index = int(grad_array_dir[i * 4 + k, j * 4 + l] / (2 * math.pi) * 8)   # floor to integer
                    # add the magnitude of this gradient to the bin
                    descriptor[base_index + extra_index] += grad_array_mag[i * 4 + k, j * 4 + l]

    # return normalized descriptor
    return descriptor / np.linalg.norm(descriptor)

def get_feature_descriptors(image, x_array, y_array, window_width, mode):
    '''
    Returns a set of feature descriptors for a given set of feature points.

    (Please note that we reccomend implementing this function after you have implemented
    match_features)

    To start with, normalize patches as your local feature descriptor. You will 
    then need to implement the more effective SIFT-like feature descriptor.
    (See Szeliski 7.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each descriptor_window_image_width/4.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on EdStem with any questions

        - skimage.filters (library)


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates (column indices) of feature points
    :y: np array of y coordinates (row indices) of feature points
    :window_width: in pixels, is the local window width. You can assume
                    that feature_width will be a multiple of 4 (i.e. every cell of your
                    local SIFT-like feature will have an integer width and height).
    :mode: either "patch" or "sift". Switches between image patch descriptors
           and SIFT descriptors

    If you want to detect and describe features at multiple scales or
    particular orientations you can add input arguments.

    :returns:
    :features: np array of computed features. features[i] is the descriptor for 
               point (x[i], y[i]), so the shape of features should be 
               (len(x), feature dimensionality). For standard SIFT, feature 
               dimensionality is 128. `Num points` may be less than len(x) if 
               some points are rejected, e.g., if out of bounds.

    '''
    # get gradients (for SIFT only)
    xgrads = gaussian_filter1d(image, 2, axis=0, order=1)
    ygrads = gaussian_filter1d(image, 2, axis=1, order=1)
    grads = np.stack([xgrads, ygrads], axis=2)

    features = []
    for i in range(len(x_array)):
        # get pixel coordinates of features
        xc = int(x_array[i])
        yc = int(y_array[i])
        # check if feature would be out of bounds
        half = window_width // 2
        if (xc < half) or (yc < half) or (yc + half >= image.shape[0]) or (xc + half >= image.shape[1]):
            continue
        
        if mode == "patch":
            # Cut out image patch
            patch = image[yc - half : yc + half, xc - half : xc + half]
            # Flatten
            vec = patch.flatten()
            # Normalize
            vec /= np.linalg.norm(vec)
            # Append to feature list
            features.append(vec)

        if mode == 'sift':
        # grad_x = np.gradient(image, axis=0)
        # grad_y = np.gradient(image, axis=1)
            grad_x = gaussian_filter1d(image, 2, axis=0, order=1)
            grad_y = gaussian_filter1d(image, 2, axis=1, order=1)
            grad_mg = np.sqrt(np.square(grad_x) + np.square(grad_y))
            grad_or = np.array(np.arctan2(grad_y, grad_x) % (2 * np.pi))

            cell_width = window_width // 4
            for (x, y) in zip(x_array, y_array):
                if x < window_width // 2 or x + window_width // 2 >= image.shape[1] or y < window_width // 2 or y + window_width // 2 >= image.shape[0]:
                    continue
                descriptor = np.zeros(128)
                for i in range(4):
                    for j in range(4):
                        histogram = np.zeros(8)
                        x_s = x - window_width // 2 + i * cell_width
                        x_e = x_s + cell_width
                        y_s = y - window_width // 2 + j * cell_width
                        y_e = y_s + cell_width
                        cell_mg = np.array(grad_mg[y_s:y_e, x_s:x_e]).flatten()
                        cell_or = np.array(grad_or[y_s:y_e, x_s:x_e]).flatten()
                        for k in range(cell_width**2):
                            histogram[int(((cell_or[k] + np.pi) / (np.pi / 4)) % 8)] += cell_mg[k]
                        # print('orientation: ', cell_or)
                        # print('histogram', histogram, '\n')
                        descriptor[(i+j)*8:(i+j+1)*8] = histogram
                features.append(descriptor)
            features = np.array(features) / np.linalg.norm(features)

    return features

    return np.asarray(features)

def match_features(im1_features, im2_features):
    '''
    Matches feature descriptors of one image with their nearest neighbor in the other. 
    Implements the Nearest Neighbor Distance Ratio (NNDR) Test to help threshold
    and remove false matches.

    Please implement the "Nearest Neighbor Distance Ratio (NNDR) Test" ,
    Equation 7.18 in Section 7.1.3 of Szeliski.

    For extra credit you can implement spatial verification of matches.

    Remember that the NNDR will return a number close to 1 for feature 
    points with similar distances. Think about how you might want to threshold
    this ratio (hint: see lecture slides)

    This function does not need to be symmetric (e.g., it can produce
    different numbers of matches depending on the order of the arguments).

    A match is between a feature in im1_features and a feature in im2_features. We can
    represent this match as a the index of the feature in im1_features and the index
    of the feature in im2_features

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on EdStem with any questions

        - zip (python built in function)
        - np.argsort()

    :params:
    :im1_features: an np array of features returned from get_feature_descriptors() for feature points in image1
    :im2_features: an np array of features returned from get_feature_descriptors() for feature points in image2

    :returns:
    :matches: an np array of dimension k x 2 where k is the number of matches. The first
            column is an index into im1_features and the second column is an index into im2_features
    '''

    # create empty matches array
    matches = np.zeros((im1_features.shape[0], 2))

    # use cdist to get distance matrix (euclidian by default)
    # dists = cdist(im1_features, im2_features)
    dists = np.sqrt(np.sum(im1_features ** 2, 1).reshape(-1, 1) + np.sum(im2_features ** 2, 1).reshape(1, -1) - 2 * im1_features @ im2_features.T)

    # sort and argsort
    sorted_dist_indices = np.argsort(dists, axis=1)
    sorted_dists = np.sort(dists, axis=1)

    # sorted for best matches from image 2 to 1
    sorted_dist_indices_transpose = np.argsort(dists, axis=0)

    # set matches to be from each feature in image 1 to the most similar feature in image 2
    matches[:, 0] = np.arange(im1_features.shape[0])
    matches[:, 1] = sorted_dist_indices[:, 0]

    # get ratio of best match similarity to next best match similarity
    ratios = sorted_dists[:, 0] / sorted_dists[:, 1]

    # discount matches that only go in one direction
    for i in range(matches.shape[0]):
        if sorted_dist_indices_transpose[0, int(matches[i, 1])] != i:
            ratios[i] *= 1.1

    # Optimal threshold determined from lecture slides
    optimal_threshold = 0.85

    # Threshold
    mask = (ratios < optimal_threshold)
    ratios = ratios[mask]
    matches = matches[mask]

    return matches
