import numpy as np
from imageio import imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from scipy.misc import imread

GRASCALE_REPRE = 1
RGB_REPRE = 2
GRASCALE_SHAPE = 2
RGB_SHAPE = 3
COLOR_LEVEL = 256


def read_image(filename, representation):

    '''
    A function that converts the image to a desired representation, and with
    intesities normalized to the range of [0,1]
    :param filename: the filename of an image on disk, could be grayscale or
    RGB
    :param representation: representation code, either 1 or 2 defining whether
    the output should be a grayscale image (1) or an RGB image (2)
    :return: an image in the desired representation.
    '''

    im = imread(filename)
    if representation == GRASCALE_REPRE:
        im = rgb2gray(im)
    im_float = im.astype(np.float64)
    im_float /= (COLOR_LEVEL - 1)
    return im_float


def image_type_check(im):
    '''
    Helper function that checks if the image is in grayscale or RGB.
    :param im: the image represented by matrix
    :return: The dimension of the matrix representing the image
    '''

    if np.ndim(im) == GRASCALE_SHAPE:
        return GRASCALE_SHAPE
    else:
        return RGB_SHAPE


def imdisplay(filename, representation):
    '''
    A function that display an image in a given representation.
    The function opens a new figure and display the loaded image in the
    converted representation.
    :param filename: the filename of an image on disk, could be grayscale or
    RGB
    :param representation: representation code, either 1 or 2 defining whether
    the output should be a grayscale image (1) or an RGB image (2)
    '''

    im = imread(filename)
    if representation == image_type_check(im):
        pass
    elif representation == GRASCALE_REPRE:
        im = read_image(filename, GRASCALE_REPRE)
    plt.imshow(im, cmap='gray')
    plt.show()


def rgb2yiq(imRGB):
    '''
    Function that transform an RGB image into the YIQ color space by matrix
    multiplication.
    :param imRGB: an image represented by height × width × 3 np.float64
    matrices with values in [0, 1]
    :return: image in the YIQ color space
    '''

    yiq_mat = np.array([[0.299, 0.587, 0.114]
                           , [0.596, -0.275, -0.321]
                           , [0.212, -0.523, 0.311]])

    return np.dot(imRGB, yiq_mat.transpose())


def yiq2rgb(imYIQ):
    '''
    Function that transform an YIQ image into the RGB color space by matrix
    multiplication.
    :param imYIQ: an image represented by height × width × 3 np.float64
    matrices with values in [0, 1]
    :return: image in the RGB color space
    '''

    yiq_mat = np.array([[0.299, 0.587, 0.114]
                           , [0.596, -0.275, -0.321]
                           , [0.212, -0.523, 0.311]])
    inv_yiq_mat = np.linalg.inv(yiq_mat)

    return np.dot(imYIQ, inv_yiq_mat.transpose())


def histogram_equalize(im_orig):
    '''
    Function that performs histogram equalization of a given grayscale or RGB
    image.
    :param im_orig: is the input grayscale or RGB float64 image with values in
    [0, 1].
    :return:
    im_eq - is the equalized image. grayscale or RGB float64 image with values
    in [0, 1].
    hist_orig - is a 256 bin histogram of the original image (array with shape
    (256,)).
    hist_eq - is a 256 bin histogram of the equalized image (array with shape
    (256,)).
    '''

    if image_type_check(im_orig) == RGB_SHAPE:
        yiq_im = rgb2yiq(im_orig)
        im_int = (yiq_im[:, :, 0] * (COLOR_LEVEL - 1)).astype(np.uint8)
    else:
        im_int = (im_orig * (COLOR_LEVEL - 1)).astype(np.uint8)

    hist_orig, _ = np.histogram(im_int, COLOR_LEVEL)
    cum_sum = np.cumsum(hist_orig)
    cum_sum.astype(np.float64)
    cum_sum = cum_sum / np.sum(
        hist_orig)  # normalizing the cumulative histogram
    cum_sum *= (COLOR_LEVEL - 1)  # Multiply the normalized histogram by the
    # maximal gray level value (K-1)

    min_in_cumsum = cum_sum[
        np.argmax(cum_sum > 0)]  # the first gray level not 0
    new_cum_sum = np.round(((cum_sum - min_in_cumsum)
                            / (cum_sum[COLOR_LEVEL - 1] - min_in_cumsum))
                           * (COLOR_LEVEL - 1))

    cast_cum_sum = new_cum_sum.astype(np.uint8)
    new_im = cast_cum_sum[im_int]
    hist_eq, _ = np.histogram(new_im, COLOR_LEVEL)

    new_im = new_im.astype(np.float64) / (COLOR_LEVEL - 1)
    if image_type_check(im_orig) == RGB_SHAPE:
        yiq_im[:, :, 0] = new_im
        new_im = yiq2rgb(yiq_im)

    return [new_im, hist_orig, hist_eq]


def quantize(im_orig, n_quant, n_iter):
    '''
    Function that performs optimal quantization of a given grayscale or RGB
    image.
    :param im_orig: input grayscale or RGB image to be quantized
    (float64 image with values in [0, 1]).
    :param n_quant: is the number of intensities the output im_quant image
    should have.
    :param n_iter:is the maximum number of iterations of the optimization
    procedure
    :return:
    im_quant - the quantized output image.
    error - an array with shape (n_iter,) (or less) of the total intensities
    error for each iteration of the quantization procedure.
    '''

    im_int = (im_orig * (COLOR_LEVEL - 1)).astype(np.uint8)
    hist_lst, _ = np.histogram(im_int, COLOR_LEVEL)
    cum_sum = np.cumsum(hist_lst)
    # Initializing the first z values
    bins_index = np.array([np.abs(cum_sum - i * im_int.size / n_quant).argmin()
                           for i in range(n_quant)] + [COLOR_LEVEL])
    # calculating a weighted histogram for the summing
    weighted_hist = np.arange(COLOR_LEVEL) * hist_lst

    q_lst = np.zeros(n_quant)
    err_lst = [0, ] * (n_iter)

    iter_count = 0
    greenlight = 1
    while (iter_count < n_iter and greenlight):
        new_bins_index = np.copy(bins_index)
        for j in range(n_quant):  # Qi calculation
            q_lst[j] = np.sum(weighted_hist[bins_index[j]:bins_index[j + 1]]) \
                       / np.sum(hist_lst[bins_index[j]:bins_index[j + 1]])

        bins_index[1:-1] = (q_lst[:-1] + q_lst[1:]) // 2  # Updating the z vals

        for j in range(
                        n_quant - 1): #Calculating the error for each iteration
            tent_sum = 0
            for i in range(bins_index[j], bins_index[j + 1]):
                tent_sum += (q_lst[j] - i) ** 2 * hist_lst[i]
            err_lst[iter_count] += tent_sum
        iter_count += 1

        if np.array_equal(new_bins_index,bins_index):
            greenlight = 0

    # Creating a lookup table for the image pixel conversion
    lookup_lst = np.zeros(COLOR_LEVEL, dtype=np.uint8)
    for j in range(n_quant):
        lookup_lst[bins_index[j]: bins_index[j + 1]] = q_lst[j]

    reduced_image = lookup_lst[im_int]

    return [reduced_image, err_lst]