import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, feature
from skimage.measure import regionprops
from scipy import ndimage
from skimage.color import rgb2gray
from skimage import feature

def plot_feature_points(image, xs, ys):
    '''
    Plot feature points for the input image. 
    
    Show the feature points (x, y) over the image. Be sure to add the plots you make to your writeup!

    Useful functions: Some helpful (but not necessarily required) functions may include:
        - plt.imshow
        - plt.scatter
        - plt.show
        - plt.savefig
    
    :params:
    :image: a grayscale or color image (depending on your implementation)
    :xs: np.array of x coordinates of feature points
    :ys: np.array of y coordinates of feature points
    '''

    # TODO: Your implementation here!
    if image.ndim == 2:
        plt.imshow(image, cmap = 'gray')
    else:
        plt.imshow(image)
    
    plt.scatter(xs, ys, c = 'red', marker = 'o', s = 40, label = 'feature points')

    plt.savefig('../results/mypoints.jpg')
    plt.show()

def get_feature_points(image, window_width):
    '''
    Implement the Harris corner detector to return feature points for a given image.

    You do not need to worry about scale invariance or keypoint orientation estimation
    for your Harris corner detector.

    If you're finding spurious (false/fake) feature point detections near the boundaries,
    it is safe to suppress the gradients / corners near the edges of the image.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on EdStem with any questions

        - skimage.feature.peak_local_max (experiment with different min_distance values to get good results)
        - skimage.measure.regionprops
          
    Note: You may decide it is unnecessary to use feature_width in get_feature_points, or you may also decide to 
    use this parameter to exclude the points near image edges.

    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :window_width: the width and height of each local window in pixels

    :returns:
    :xs: an np.array of the x coordinates (column indices) of the feature points in the image
    :ys: an np.array of the y coordinates (row indices) of the feature points in the image

    :optional returns (may be useful for extra credit portions):
    :confidences: an np.array indicating the confidence (strength) of each feature point
    :scale: an np.array indicating the scale of each feature point
    :orientation: an np.array indicating the orientation of each feature point

    '''

    # STEP 1: Calculate the gradient (partial derivatives on two directions).
    # STEP 2: Apply Gaussian filter with appropriate sigma.
    # STEP 3: Calculate Harris cornerness score for all pixels.
    # STEP 4: Peak local max to eliminate clusters. (Try different parameters.)

    # TODO: Your implementation here!
    # These are placeholders - replace with the coordinates of your feature points!
    xs = np.random.randint(0, image.shape[1], size=100)
    ys = np.random.randint(0, image.shape[0], size=100)
    #count = 0

    #max_coordinates = feature.peak_local_max(image, min_distance=1)

    if image.ndim == 3:
        image = rgb2gray(image)

    #height, width = image.shape
    #w = window_width//2
    #padded_image = np.pad(image, ((1, 1), (1, 1)), 'constant')
    #padded_image = ndimage.gaussian_filter(padded_image, sigma = 1)
    sigma = window_width/4
    #image = ndimage.gaussian_filter(image, sigma = sigma)

    Ix = ndimage.sobel(image, 1) #1 is horizontal?
    Iy = ndimage.sobel(image, 0) 

    Ixx = np.multiply(Ix, Ix)
    Iyy = np.multiply(Iy, Iy)
    Ixy = np.multiply(Ix, Iy)

    Ixx = ndimage.gaussian_filter(Ixx, sigma=sigma)
    Iyy = ndimage.gaussian_filter(Iyy, sigma=sigma)
    Ixy = ndimage.gaussian_filter(Ixy, sigma=sigma)

    #k is a constant
    k = 0.06
    
    detM = np.multiply(Ixx, Iyy) - np.multiply(Ixy, Ixy)
    #print(detM.shape)
    traceM = Ixx + Iyy
    #print(traceM.shape)

    #C is a matrix containing all cornerness scores
    cornerness = detM - k*np.multiply(traceM, traceM)
    #print(cornerness)

    #thresholding
    c_max = np.max(cornerness)
    thresh = 0.1 * c_max

    # Find local maxima of the response
    min_d = window_width//2
    coordinates = feature.peak_local_max(cornerness, min_distance=min_d, threshold_abs=thresh)
    #print(coordinates)

    # Extract x and y coordinates
    xs = coordinates[:, 1]
    #print(xs)
    ys = coordinates[:, 0]
    #print(ys)

    '''
    #coordinates are probably off
    for i in range(height-w):
        for j in range(width-w):
            x = i+w
            y = j+w

            window_Ixx = Ixx[x-w : x+w+1, y-w : y+w+1]
            window_Iyy = Iyy[x-w : x+w+1, y-w : y+w+1]
            window_Ixy = Ixy[x-w : x+w+1, y-w : y+w+1]

            #second moment matrix
            M = np.array([[np.sum(window_Ixx), np.sum(window_Ixy)], 
                          [np.sum(window_Ixy), np.sum(window_Iyy)]])
            #cornerness score
            C = np.linalg.det(M) - k*np.power(np.trace(M), 2)

            #threshold value
            thresh = 0.5
            coordinate = [i, j]
            if C > thresh and coordinate in max_coordinates:
                if count >= 100:
                    np.append(xs, i)
                    np.append(ys, j)
                else:
                    xs[count] = i
                    ys[count] = j
                    count += 1
    '''
    return xs, ys


def get_feature_descriptors(image, xs, ys, window_width, mode):
    '''
    Computes features for a given set of feature points.

    To start with, use image patches as your local feature descriptor. You will 
    then need to implement the more effective SIFT-like feature descriptor. Use 
    the `mode` argument to toggle between the two.
    (Original SIFT publications at http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) feature descriptor should have:
    (1) A 4x4 grid of cells, each feature_width / 4 pixels square.
    (2) Each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4 x 4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length.

    This is a design task, so many options might help but are not essential.
    - To perform interpolation such that each gradient
    measurement contributes to multiple orientation bins in multiple cells
    A single gradient measurement creates a weighted contribution to the 4 
    nearest cells and the 2 nearest orientation bins within each cell, for 
    8 total contributions.

    - To compute the gradient orientation at each pixel, we could use oriented 
    kernels (e.g. a kernel that responds to edges with a specific orientation). 
    All of your SIFT-like features could be constructed quickly in this way.

    - You could normalize -> threshold -> normalize again as detailed in the 
    SIFT paper. This might help for specular or outlier brightnesses.

    - You could raise each element of the final feature vector to some power 
    that is less than one.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on EdStem with any questions

        - skimage.filters (library)

    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :xs: np.array of x coordinates (column indices) of feature points
    :ys: np.array of y coordinates (row indices) of feature points
    :window_width: in pixels, is the local window width. You can assume
                    that window_width will be a multiple of 4 (i.e. every cell of your
                    local SIFT-like window will have an integer width and height).
    :mode: a string, either "patch" or "sift". Switches between image patch descriptors
           and SIFT descriptors

    If you want to detect and describe features at multiple scales or
    particular orientations you can add input arguments. Make sure input arguments 
    are optional or the autograder will break.

    :returns:
    :features: np.array of computed features. features[i] is the descriptor for 
               point (x[i], y[i]), so the shape of features should be 
               (len(x), feature dimensionality). For standard SIFT, `feature
               dimensionality` is typically 128. `num points` may be less than len(x) if
               some points are rejected, e.g., if out of bounds.
    '''
    if mode == "sift":
        return get_feature_descriptors_SIFT(image, xs, ys, window_width)
    else:
        return get_feature_descriptors_patch(image, xs, ys, window_width)

def get_feature_descriptors_patch(image, xs, ys, window_width):
    # IMAGE PATCH STEPS
    # STEP 1: For each feature point, cut out a window_width x window_width patch 
    #         of the image around that point (as you will in SIFT)
    # STEP 2: Flatten this image patch into a 1-dimensional vector (hint: np.flatten())

    return None

def get_feature_descriptors_SIFT(image, xs, ys, window_width):
    # SIFT STEPS
    # STEP 1: Calculate the gradient (partial derivatives on two directions) on all pixels.
    # STEP 2: Decompose the gradient vectors to magnitude and orientation (angle).
    # STEP 3: For each feature point, calculate the local histogram based on related 4x4 grid cells.
    #         Each cell is a square with feature_width / 4 pixels length of side.
    #         For each cell, we assign these gradient vectors corresponding to these pixels to 8 bins
    #         based on the orientation (angle) of the gradient vectors. 
    # STEP 4: Now for each cell, we have a 8-dimensional vector. Appending the vectors in the 4x4 cells,
    #         we have a 128-dimensional feature.
    # STEP 5: Don't forget to normalize your feature.

    # TODO: Your implementation here!
    # These are placeholders - replace with the coordinates of your feature points!
    features = np.zeros((len(xs), 128))

    #'''
    #STEP 1: For each feature point, cut out a window_width x window_width patch 
    #        of the image around that point (as you will in SIFT)
    w2 = window_width//2 #half of window size
    
    for i in range(len(xs)):
        col = xs[i]
        row = ys[i]

        window = image[row-w2 : row+w2, col-w2 : col+w2]
        if(window.shape != (window_width, window_width)):
            return features

        # STEP 1: Calculate the gradient (partial derivatives on two directions) on all pixels.
        grad_x = ndimage.sobel(window, 1)
        grad_y = ndimage.sobel(window, 0)

        # STEP 2: Decompose the gradient vectors to magnitude and orientation (angle).
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        grad_ori_index = np.round(np.arctan2(grad_y, grad_x) * 4 / np.pi) #convert to degrees
        grad_ori_index[grad_ori_index < 0] += 8
        
        print(grad_ori_index.shape)
        assert window.shape == grad_ori_index.shape

        # STEP 3: For each feature point, calculate the local histogram based on related 4x4 grid cells.
        #         Each cell is a square with window_width / 4 pixels length of side.
        #         For each cell, we assign these gradient vectors corresponding to these pixels to 8 bins
        #         based on the orientation (angle) of the gradient vectors. 
        j = 0
        jj = 0
        for u in range(0, 16, 4):
            for v in range(0, 16, 4):
                jj = j*8
                j += 1
                for k in range(u, u+4):
                    for l in range(v, v+4):
                        bin_index = grad_ori_index[k, l]
                        cur_mag = grad_mag[k, l]

                        # STEP 4: Now for each cell, we have a 8-dimensional vector. Appending the vectors in the 4x4 cells,
                        #         we have a 128-dimensional features
                        #print(i, int(jj+bin_index), cur_mag)
                        features[i][int(jj+bin_index)] += cur_mag
    # STEP 5: Don't forget to normalize your feature.
    features[features>0.2] = 0.2
    #print(features)


    #'''
    return features


def match_features(im1_features, im2_features):
    '''
    Matches feature descriptors of one image with their nearest neighbor in the other.

    Implements the Nearest Neighbor Distance Ratio (NNDR) Test to help threshold
    and remove false matches.

    Please implement the "Nearest Neighbor Distance Ratio (NNDR) Test".

    For extra credit you can implement spatial verification of matches.

    Remember that the NNDR will return a number close to 1 for feature 
    points with similar distances. Think about how you might want to threshold
    this ratio (hint: see lecture slides for NNDR)

    This function does not need to be symmetric (e.g., it can produce
    different numbers of matches depending on the order of the arguments).

    A match is between a feature in im1_features and a feature in im2_features. We can
    represent this match as a the index of the feature in im1_features and the index
    of the feature in im2_features

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on EdStem with any questions

        - np.argsort()

    :params:
    :im1_features: an np.array of features returned from get_feature_descriptors() for feature points in image1
    :im2_features: an np.array of features returned from get_feature_descriptors() for feature points in image2

    :returns:
    :matches: an np.array of dimension k x 2 where k is the number of matches. The first
            column is an index into im1_features and the second column is an index into im2_features
    '''
    
    # STEP 1: Calculate the distances between each pairs of features between im1_features and im2_features.
    # STEP 2: Sort and find closest features for each feature
    # STEP 3: Compute NNDR for each match
    # STEP 4: Remove matches whose ratios do not meet a certain threshold 

    # TODO: Your implementation here!
    # These are placeholders - replace with the coordinates of your feature points!
    matches = np.random.randint(0, min(len(im1_features), len(im2_features)), size=(50, 2))

    return matches
