import math
import numpy as np

""" Helper Functions Module

This module includes all the methods used for repeating calculations. 

@authors: RonyHirsch, AbdoSharaf98
"""


def deg2pix(view_distance, degrees, cm_per_pixel):
    """
    converts degrees to pixels
    :param view_distance: viewer distance from the display screen IN CENTIMETERS!
    :param degrees: degrees visual angle to be converted to no. of pixels
    :param cm_per_pixel: the size of one pixel in centimeters
    :return: pixels: the number of pixels corresponding to the degrees visual angle specified
    """

    # get the size of the visual field
    centimeters = math.tan(math.radians(degrees) / 2) * (2 * view_distance)

    # now convert the centimeters to pixels
    pixels = round(centimeters / cm_per_pixel[0])

    return pixels


def smooth(x, window_len):
    """
    Python implementation of matlab's smooth function.
    """

    if window_len < 3:
        return x

    # Window length must be odd
    if window_len % 2 == 0:
        window_len += 1

    window_len = int(window_len)
    w = np.ones(window_len)
    y = np.convolve(w, x, mode='valid') / len(w)
    y = np.hstack((x[:window_len // 2], y, x[len(x) - window_len // 2:]))

    for i in range(0, window_len // 2):
        y[i] = np.sum(y[0: i + i]) / ((2 * i) + 1)

    for i in range(len(x) - window_len // 2, len(x)):
        y[i] = np.sum(y[i - (len(x) - i - 1): i + (len(x) - i - 1)]) / ((2 * (len(x) - i - 1)) + 1)

    return y


def CalcFixationDensity(gaze, scale, screenDims, gazeX=None, gazeY=None):
    """
    This function divides the screen into bins and sums the time during which a gaze was present at each bin
    :param gaze: a tuple where the first element is gazeX and the second is gazeY. gazeX and gazeY are both NxD matrices
                where N is ntrials and D is number of timepoints
    :param scale:
    :param screenDims:
    :return:
    """
    # make sure inputs are arrays
    if not gazeX:
        gazeX = np.array(gaze[0]).flatten()  # the X coordinates of a gaze point on the screen.
    else:
        gazeX = np.array(gazeX).flatten()

    if not gazeY:
        gazeY = np.array(gaze[1]).flatten()  # the Y coordinates of a gaze point on the screen.
    else:
        gazeY = np.array(gazeY).flatten()
    # each (x, y) tuple is a point on the screen - we'll then divide the screen to bins, and ask how many (x,y) fall in
    # each bin. This is how we'll get the frequency og the gaze per bin.

    # initialize the fixation density matrix
    fixDensity = np.zeros((int(np.ceil(screenDims[1] / scale)), int(np.ceil(screenDims[0] / scale))))

    # loop through the bins
    L = len(gazeX)
    for i in range(0, fixDensity.shape[1]):
        for j in range(0, fixDensity.shape[0]):  # creates a hist of the frequency of each sample bin.
            # for each bin, we ask how many points fall in this bin
            fixDensity[j, i] = np.sum(((gazeX >= scale * i) & (gazeX <= scale * (i + 1))) &
                                      ((gazeY >= scale * j) & (gazeY <= scale * (j + 1)))) / L

    return fixDensity
