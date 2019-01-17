"""

SMSI GMM with gray scale Image

"""

import time

import cv2
import numpy as np
from matplotlib import pyplot
from scipy import ndimage
from sklearn.cluster import KMeans


def initialise_parameters(features, d=None, means=None, covariances=None, weights=None):
    """
    Initialises parameters: means, covariances, and mixing probabilities
    if undefined.

    Arguments:
    features -- input features data set
    """
    if not means or not covariances:
        val = 250
        n = features.shape[0]
        # Shuffle features set
        indices = np.arange(n)
        np.random.shuffle(np.arange(n))
        features_shuffled = np.array([features[i] for i in indices])

        # Split into n_components subarrays
        divs = int(np.floor(n / k))

        # Estimate means/covariances (or both)
        if not means:
            kmeans = KMeans(n_clusters=k, random_state=0).fit(features.reshape(-1, 1))
            means = kmeans.cluster_centers_

        if not covariances:
            covariances = [val * np.identity(d) for i in range(k)]

    if not weights:
        weights = [float(1 / k) for i in range(k)]

    return (means, covariances, weights)


# gaussian function
def gau(mean, var, varInv, feature, d):
    var_det = np.linalg.det(var)
    a = np.sqrt(2 * (np.pi ** d) * var_det)
    b = np.exp(-0.5 * np.dot((feature - mean), np.dot(varInv, (feature - mean).transpose())))
    return b / a


def smsi_likeli(mean, var, varInv, s_value, feature, d):
    temp = []
    for x in range(k):
        temp.append(s_value * gau(mean[x], var[x], varInv[x], feature, d))
    return temp


def saliency(img):
    c = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    mag = np.sqrt(c[:, :, 0] ** 2 + c[:, :, 1] ** 2)
    spectralResidual = np.exp(np.log(mag) - cv2.boxFilter(np.log(mag), -1, (3, 3)))

    c[:, :, 0] = c[:, :, 0] * spectralResidual / mag
    c[:, :, 1] = c[:, :, 1] * spectralResidual / mag
    c = cv2.dft(c, flags=(cv2.DFT_INVERSE | cv2.DFT_SCALE))
    mag = c[:, :, 0] ** 2 + c[:, :, 1] ** 2
    cv2.normalize(cv2.GaussianBlur(mag, (9, 9), 3, 3), mag, 0., 1., cv2.NORM_MINMAX)
    pyplot.subplot(2, 2, 2)
    pyplot.imshow(mag)

    return mag


def neighbor_prob(img, fun):
    mask = np.ones((3, 3))
    result = ndimage.generic_filter(img, function=fun, footprint=mask, mode='constant', cval=np.NaN)
    return result


def sliding_window(im):
    rows, cols = im.shape
    final = np.zeros((rows, cols, 3, 3))
    for x in (0, 1, 2):
        for y in (0, 1, 2):
            im1 = np.vstack((im[x:], im[:x]))
            im1 = np.column_stack((im1[:, y:], im1[:, :y]))
            final[x::3, y::3] = np.swapaxes(im1.reshape(int(rows / 3), 3, int(cols / 3), -1), 1, 2)
    return final


input_path = "/Users/Srikanth/PycharmProjects/Gaussain/datasets/input/eagle.jpg"

img = cv2.imread(input_path, 0)
pyplot.subplot(2, 2, 1)
pyplot.imshow(img)
# no of clusters
k = 2
o_shape = img.shape

# Gray scale
d = 1

# Array of pixels
feat = img.reshape(-1)

# Total no of pixels
N = len(feat)

# s = saliency(input_path)
s = saliency(img)

means, covariances, weights = initialise_parameters(features=feat, d=d)
covariances_Inv = [np.linalg.inv(covariances[i]) for i in range(k)]
meanPrev = [np.array([0]) for i in range(k)]
iteration = []
logLikelihoods = []
counterr = 0
smsi_init_weights = np.ones((o_shape[0] * o_shape[1], k))

# Ri value
R_value = neighbor_prob(s, np.nansum)
R_value = R_value.reshape(-1)
s_value = s.reshape(-1)

while abs((np.asarray(meanPrev) - np.asarray(means)).sum()) > 0.01:
    start = time.time()
    smsi_resp = []
    for i, feature in enumerate(feat):
        smsi_classLikelihoods = smsi_likeli(means, covariances, covariances_Inv, s_value[i], feature, d)
        smsi_resp.append(smsi_classLikelihoods)

    smsi_resp = np.asarray(smsi_resp)

    final_resp = []
    for cluster in range(k):
        test = smsi_resp[:, cluster]
        result = ndimage.generic_filter(test.reshape(o_shape[0], o_shape[1]), np.nansum, footprint=np.ones((3, 3)),
                                        mode='constant', cval=np.NaN).reshape(-1)
        final_resp.append(result * smsi_init_weights[:, cluster] / R_value)

    # numerator of gama
    smsi_resp_num = np.asarray(final_resp).T

    # denominator of gama
    final_resp_den = smsi_resp_num.sum(axis=1)

    # gama value of the smsi
    final_smsi_resp = []
    for cluster in range(k):
        final_smsi_resp.append(smsi_resp_num[:, cluster] / final_resp_den)

    final_smsi_resp = np.asarray(final_smsi_resp).T

    # SMSI MEAN
    ###############################################################################################
    #################################### SMSI MEAN ################################################

    smsi_mean_den = final_smsi_resp.sum(axis=0)
    smsi_mean_num_s_x = s_value * feat
    smsi_mean_num = []
    smsi_mean_num_s_x = ndimage.generic_filter(smsi_mean_num_s_x.reshape(o_shape[0], o_shape[1]), np.nansum,
                                               footprint=np.ones((3, 3)), mode='constant', cval=np.NaN).reshape(-1)
    for i in range(k):
        smsi_mean_num.append(final_smsi_resp[:, i] * (smsi_mean_num_s_x / R_value))
    smsi_mean_num = np.asarray(smsi_mean_num).T
    smsi_mean = (smsi_mean_num.sum(axis=0) / smsi_mean_den).T
    meanPrev = means
    means = smsi_mean

    #################################################################################################
    ################################# CO VAR ########################################################
    f_u = np.asarray([feat - means[i] for i in range(k)]).T
    f_u = f_u ** 2

    covar_num_s_x_u = np.asarray([s_value * f_u[:, i] for i in range(k)]).T

    covar_num_s_x_u = np.asarray(
        [ndimage.generic_filter(covar_num_s_x_u[:, i].reshape(o_shape[0], o_shape[1]), np.nansum,
                                footprint=np.ones((3, 3)), mode='constant', cval=np.NaN).reshape(-1) for i in
         range(k)]).T

    covar_num = np.asarray([final_smsi_resp[:, i] * (covar_num_s_x_u[:, i] / R_value) for i in range(k)]).T

    covar = covar_num.sum(axis=0) / smsi_mean_den

    covariances = [covar[i] * np.identity(d) for i in range(k)]
    covariances_Inv = [np.linalg.inv(covariances[i]) for i in range(k)]

    #################################################################################################
    ################################ WEIGHTS #########################################################
    # weights for smsi
    smsi_weigths = []
    smsi_weights_num = []

    # s*gama
    w_num = np.asarray([final_smsi_resp[:, i] * s_value for i in range(k)]).T
    w_numerator = np.asarray([ndimage.generic_filter(w_num[:, i].reshape(o_shape[0], o_shape[1]), np.nansum,
                                                     footprint=np.ones((3, 3)), mode='constant', cval=np.NaN).reshape(
        -1) for i in range(k)]).T

    w_den = w_numerator.sum(axis=1)

    smsi_weights = np.asarray([w_numerator[:, i] / w_den for i in range(k)]).T

    smsi_init_weights = smsi_weights
    condition = abs((np.asarray(meanPrev) - np.asarray(means)).sum())
    print(counterr, condition, " Means: ", smsi_mean, "Time: ", time.time() - start)
    counterr += 1

segmentedImage = np.zeros((N), np.uint8)

for i, resp in enumerate(final_smsi_resp):
    max = resp.argmax()
    segmentedImage[i] = 255 - 255 * means[max]

segmentedImage = segmentedImage.reshape(o_shape[0], o_shape[1])
pyplot.subplot(2, 2, 3)
pyplot.imshow(segmentedImage)
pyplot.show()
