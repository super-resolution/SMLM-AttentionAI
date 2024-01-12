#from src.data import *
import matplotlib.pyplot as plt
from scipy import interpolate
import tensorflow as tf
from scipy.ndimage import filters
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from pathlib import Path
import numpy as np
import scipy
from scipy.ndimage.filters import gaussian_filter, uniform_filter
from time import time
from functools import wraps

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f'func:{f.__name__} args:[{args}{kw}] took: {te-ts} sec')
        return result
    return wrap


def result_image_to_coordinates(result_tensor, coord_list=None, threshold=0.2, sigma_thresh=0.1):
    result_array = []
    test = []
    for i in range(result_tensor.shape[0]):
        classifier =result_tensor[i, :, :, 2]

        # plt.imshow(classifier)
        # plt.show()
        if np.sum(classifier) > threshold:
            classifier[np.where(classifier < threshold)] = 0

            indices = get_coords(classifier).T
            x = result_tensor[i, indices[0], indices[1], 0]
            y = result_tensor[i, indices[0], indices[1], 1]
            p = result_tensor[i, indices[0], indices[1], 2]
            dx = result_tensor[i, indices[0], indices[1], 3]  # todo: if present
            dy = result_tensor[i, indices[0], indices[1], 4]
            N = result_tensor[i, indices[0], indices[1], 5]

            for j in range(indices[0].shape[0]):
                    if coord_list:
                        result_array.append(np.array([coord_list[i][0] + float(indices[0][j]) + (x[j])
                                                         , coord_list[i][1] + float(indices[1][j]) + y[j], coord_list[i][2],
                                                      dx[j], dy[j], N[j], p[j]]))
                        test.append(np.array([x[j], y[j]]))
                    else:
                        result_array.append(np.array([float(indices[0][j]) + (x[j])
                                                    ,float(indices[1][j]) + y[j], i,
                                                      dx[j], dy[j], N[j], p[j]]))
                        test.append(np.array([x[j], y[j]]))
    print(np.mean(np.array(test), axis=0))
    return np.array(result_array)

def predict_sigma(psf_crops, result_array, ai):
    perfect_psf_array = []
    print("evaluated")
    for i in range(result_array.shape[0]):
        if result_array[i,2]>0.95 and result_array[i,2]<1.25:
            perfect_psf_array.append(psf_crops[i,:,:,1].numpy())
    perfect_psfs = np.transpose(np.array(perfect_psf_array),(1,2,0))
    x=[]
    for i in range(perfect_psfs.shape[2]//10):
       x.append(ai.predict(tf.convert_to_tensor(perfect_psfs[np.newaxis,:,:,i*10:10+i*10])))
    x = np.array(x)
    print(np.mean(x,axis=0))

def extrude_perfect_psf(psf_crops, result_array):
    perfect_result_array = []
    perfect_psf_array = []
    for i in range(result_array.shape[0]):
        if result_array[i,2]>0.95 and result_array[i,2]<1.3:
            perfect_result_array.append(result_array[i,0:2])
            perfect_psf_array.append(psf_crops[i])
    coords = np.array(perfect_result_array)
    perfect_psfs = np.array(perfect_psf_array)

    full_psf = np.zeros((72,72))
    for i in range(perfect_psfs.shape[0]):
        psf = perfect_psfs[i]
        point = coords[i]
        new = scipy.ndimage.shift(psf[:, :, 1], (4.5-point[1]/8, 4.5-point[0]/8))

        c_spline = interpolate.interp2d(np.arange(0, 9, 1), np.arange(0, 9, 1), new, kind='cubic')

        full_psf += c_spline(np.arange(0, 9, 0.125), np.arange(0, 9, 0.125))
    plt.imshow(full_psf)
    plt.show()
    return full_psf[32:,32:]/np.sum(full_psf[32:,32:])
    #todo: return psf and use as matrix

def FRC_loss(i1, i2):
    dim = min(i1.shape[0], i1.shape[1], i2.shape[0], i2.shape[1])
    if dim % 2 != 0:
        dim -= 1
    i1 = i1[np.newaxis, 0:dim, 0:dim, np.newaxis]
    i2 = i2[np.newaxis, 0:dim, 0:dim, np.newaxis]
    size = dim
    size_half = size // 2

    r = np.zeros([size])
    r[:size_half] = np.arange(size_half) + 1
    r[size_half:] = np.arange(size_half, 0, -1)

    c = np.zeros([size])
    c[:size_half] = np.arange(size_half) + 1
    c[size_half:] = np.arange(size_half, 0, -1)

    [R, C] = np.meshgrid(r, c)

    help_index = np.round(np.sqrt(R ** 2 + C ** 2))
    kernel_list = []

    for i in range(1, 102):
        new_matrix = np.zeros(shape=[size, size])
        new_matrix[help_index == i] = 1
        kernel_list.append(new_matrix)

    kernel_list = tf.constant(kernel_list, dtype=tf.complex64)

    i1 = tf.squeeze(i1, axis=0)
    i1 = tf.squeeze(i1, axis=-1)

    i2 = tf.squeeze(i2, axis=0)
    i2 = tf.squeeze(i2, axis=-1)

    i1 = tf.cast(i1, dtype=tf.complex64)
    i2 = tf.cast(i2, dtype=tf.complex64)

    I1 = tf.signal.fft2d(i1)
    I2 = tf.signal.fft2d(i2)

    A = tf.multiply(I1, tf.math.conj(I2))
    B = tf.multiply(I1, tf.math.conj(I1))
    C = tf.multiply(I2, tf.math.conj(I2))

    A_val = tf.reduce_mean(tf.multiply(A, kernel_list), axis=(1, 2))
    B_val = tf.reduce_mean(tf.multiply(B, kernel_list), axis=(1, 2))
    C_val = tf.reduce_mean(tf.multiply(C, kernel_list), axis=(1, 2))

    res = tf.abs(A_val) / tf.sqrt(tf.abs(tf.multiply(B_val, C_val)))

    return 1.0 - tf.reduce_sum(res) / 102.0





def read_thunderstorm_drift_json(path):
    import json
    from scipy.interpolate import CubicSpline
    with open(path, 'r') as f:
        data = json.load(f)

    def get_knots_drift(name):
        knots = data[name]["knots"]
        drift = []
        polynom = data[name]['polynomials']

        for poly in polynom:
            coeff = poly["coefficients"]
            drift.append(coeff[0])
        drift.append(coeff[0] + coeff[1] * (knots[-1] - knots[-2] - 1))
        return knots, drift

    knots_x, drift_x = get_knots_drift("xFunction")

    x = np.arange(knots_x[0], knots_x[-1] + 1)
    poly_x = CubicSpline(knots_x, drift_x)
    x_drift = poly_x(x)
    knots_y, drift_y = get_knots_drift("yFunction")
    poly_y = CubicSpline(knots_y, drift_y)
    y_drift = poly_y(x)
    return np.stack([x_drift, y_drift],axis=-1)

def read_thunderstorm_drift(path):
    import pandas as pd
    from scipy.interpolate import interp1d
    data = pd.read_csv(path, sep=",")
    frames_x = data['X2'].values
    frames_y = data['X3'].values
    drift_x = data['Y2'].values
    drift_y = data['Y3'].values
    x_pol = interp1d(frames_x, drift_x, kind='cubic')
    y_pol = interp1d(frames_y, drift_y, kind='cubic')
    X = np.arange(1, 4800)#todo: image shape
    Y = np.arange(1, 4800)
    driftx_n = x_pol(X)
    drifty_n = y_pol(Y)
    return np.stack([driftx_n, drifty_n],axis=-1)


def get_reconstruct_coords(tensor, th, neighbors=3):
    import copy
    filter = np.ones((neighbors,neighbors))
    filter[0::2,0::2] = 0
    tensor = copy.deepcopy(tensor)
    convolved = scipy.ndimage.convolve(tensor, filter)#todo: norm?
    indices = np.where(convolved >th)
    x = []
    y = []
    for i in range(indices[0].shape[0]):
        ind_x_min = indices[0][i]-1
        ind_y_min = indices[1][i]-1
        if ind_x_min<0:
            ind_x_min = 0
        if ind_y_min < 0:
            ind_y_min = 0
        t = tensor[ind_x_min:indices[0][i]+2, ind_y_min:indices[1][i]+2]
        max_ind = np.where(t==t.max())
        x.append(max_ind[0][0] + indices[0][i]-1)
        y.append(max_ind[1][0] + indices[1][i]-1)
        if convolved[indices[0][i],indices[1][i]] > 2*th:
            second = np.partition(t.flatten(), -2)[-2]
            second_ind = np.where(t == second)
            x.append(second_ind[0][0] + indices[0][i] - 1)
            y.append(second_ind[1][0] + indices[1][i] - 1)

    indices = np.unique(np.array((np.array(x).astype(np.int32), np.array(y).astype(np.int32))),axis=1)
    return indices

def get_coords(reconstruct, neighbors=5):
    neighborhood = np.ones((neighbors,neighbors)).astype(np.bool)
    #create cross structure
    #if neighbors ==3:
        #neighborhood[0::2,0::2] = False
    # apply the local maximum filter; all pixel of maximal value
    # in their neighborhood are set to 1
    local_max = filters.maximum_filter(reconstruct, footprint=neighborhood) == reconstruct

    # local_max is a mask that contains the peaks we are
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.

    # we create the mask of the background
    background = (reconstruct == 0)

    # a little technicality: we must erode the background in order to
    # successfully subtract it form local_max, otherwise a line will
    # appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background
    fig,axs = plt.subplots(2)

    axs[0].imshow(detected_peaks)
    axs[1].imshow(reconstruct)
    plt.show()
    coords = np.array(np.where(detected_peaks != 0))
    return coords.T

def get_root_path():
    return str(Path(__file__).parent.parent)