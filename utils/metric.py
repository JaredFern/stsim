from __future__ import division
import numpy as np

from utils.filterbank import Steerable, SteerableNoSub
from scipy import signal

import cv2
import itertools

def conv(a, b):
    """ Larger matrix go first """
    return signal.correlate2d(a, b, mode='valid')

def pooling(im1, im2, window=7):
	# return cv2.filter2D(a, -1, b, anchor = (0,0))\
	# 	[:(a.shape[0]-b.shape[0]+1), :(a.shape[1]-b.shape[1]+1)]
	tmp = np.power(compute_L_term(im1, im2, window,) * compute_C_term(im1, im2, window) *
				   compute_C01_term(im1, im2, window) * compute_C10_term(im1, im2, window), 0.25)
	return tmp.mean()


def compute_L_term(im1, im2, window):
    C = 0.001
    window = fspecial(window, window / 6)
    mu1 = np.abs(conv(im1, window))
    mu2 = np.abs(conv(im2, window))

    Lmap = (2 * mu1 * mu2 + C) / (mu1 * mu1 + mu2 * mu2 + C)
    return Lmap


def compute_C_term(im1, im2, window):
    C = 0.001
    window = fspecial(window, window / 6)
    mu1 = np.abs(conv(im1, window))
    mu2 = np.abs(conv(im2, window))

    sigma1_sq = conv(np.abs(im1 * im1), window) - mu1 * mu1
    sigma1 = np.sqrt(sigma1_sq)
    sigma2_sq = conv(np.abs(im2 * im2), window) - mu2 * mu2
    sigma2 = np.sqrt(sigma2_sq)

    Cmap = (2 * sigma1 * sigma2 + C) / (sigma1_sq + sigma2_sq + C)
    return Cmap


def compute_C01_term(im1, im2, window):
    C = 0.001
    window2 = 1 / (window * (window - 1)) * np.ones((window, window - 1))

    im11 = im1[:, :-1]
    im12 = im1[:, 1:]
    im21 = im2[:, :-1]
    im22 = im2[:, 1:]

    mu11 = conv(im11, window2)
    mu12 = conv(im12, window2)
    mu21 = conv(im21, window2)
    mu22 = conv(im22, window2)

    sigma11_sq = conv(np.abs(im11 * im11), window2) - np.abs(mu11 * mu11)
    sigma12_sq = conv(np.abs(im12 * im12), window2) - np.abs(mu12 * mu12)
    sigma21_sq = conv(np.abs(im21 * im21), window2) - np.abs(mu21 * mu21)
    sigma22_sq = conv(np.abs(im22 * im22), window2) - np.abs(mu22 * mu22)

    sigma1_cross = conv(im11 * np.conj(im12), window2) - mu11 * np.conj(mu12)
    sigma2_cross = conv(im21 * np.conj(im22), window2) - mu21 * np.conj(mu22)

    rho1 = (sigma1_cross + C) / (np.sqrt(sigma11_sq) * np.sqrt(sigma12_sq) + C)
    rho2 = (sigma2_cross + C) / (np.sqrt(sigma21_sq) * np.sqrt(sigma22_sq) + C)
    C01map = 1 - 0.5 * np.abs(rho1 - rho2)

    return C01map


def compute_C10_term(im1, im2, window):
    C = 0.001
    window2 = 1 / (window * (window - 1)) * np.ones((window - 1, window))

    im11 = im1[:-1, :]
    im12 = im1[1:, :]
    im21 = im2[:-1, :]
    im22 = im2[1:, :]

    mu11 = conv(im11, window2)
    mu12 = conv(im12, window2)
    mu21 = conv(im21, window2)
    mu22 = conv(im22, window2)

    sigma11_sq = conv(np.abs(im11 * im11), window2) - np.abs(mu11 * mu11)
    sigma12_sq = conv(np.abs(im12 * im12), window2) - np.abs(mu12 * mu12)
    sigma21_sq = conv(np.abs(im21 * im21), window2) - np.abs(mu21 * mu21)
    sigma22_sq = conv(np.abs(im22 * im22), window2) - np.abs(mu22 * mu22)

    sigma1_cross = conv(im11 * np.conj(im12), window2) - mu11 * np.conj(mu12)
    sigma2_cross = conv(im21 * np.conj(im22), window2) - mu21 * np.conj(mu22)

    rho1 = (sigma1_cross + C) / (np.sqrt(sigma11_sq) * np.sqrt(sigma12_sq) + C)
    rho2 = (sigma2_cross + C) / (np.sqrt(sigma21_sq) * np.sqrt(sigma22_sq) + C)
    C10map = 1 - 0.5 * np.abs(rho1 - rho2)

    return C10map


def compute_cross_term(im11, im12, im21, im22, window):
    C = 0.001
    window2 = 1 / (window ** 2) * np.ones((window, window))

    mu11 = conv(im11, window2)
    mu12 = conv(im12, window2)
    mu21 = conv(im21, window2)
    mu22 = conv(im22, window2)

    sigma11_sq = conv((im11 * im11), window2) - (mu11 * mu11)
    sigma12_sq = conv((im12 * im12), window2) - (mu12 * mu12)
    sigma21_sq = conv((im21 * im21), window2) - (mu21 * mu21)
    sigma22_sq = conv((im22 * im22), window2) - (mu22 * mu22)
    sigma1_cross = conv(im11 * im12, window2) - mu11 * (mu12)
    sigma2_cross = conv(im21 * im22, window2) - mu21 * (mu22)

    rho1 = (sigma1_cross + C) / (np.sqrt(sigma11_sq) * np.sqrt(sigma12_sq) + C)
    rho2 = (sigma2_cross + C) / (np.sqrt(sigma21_sq) * np.sqrt(sigma22_sq) + C)

    Crossmap = 1 - 0.5 * abs(rho1 - rho2)
    return Crossmap


def MSE(img1, img2):
    return ((img2 - img1) ** 2).mean()


def fspecial(win=11, sigma=1.5):
    """2D gaussian mask - should give the same result as MATLAB's fspecial('gaussian',[shape],[sigma])"""
    shape = (win, win)
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()

    if sumh != 0:
        h /= sumh
    return h


class Metric:
    def __init__(self):
        self.win = 7

    def STSIM_M(self, im):
        ss = Steerable(5)
        M, N = im.shape
        coeff = ss.buildSCFpyr(im)

        f = []
        # single subband statistics
        for s in ss.getlist(coeff):
            s = s.real
            shiftx = np.roll(s, 1, axis=0)
            shifty = np.roll(s, 1, axis=1)

            f.append(np.mean(s))
            f.append(np.var(s))
            f.append((shiftx * s).mean() / s.var())
            f.append((shifty * s).mean() / s.var())

        # correlation statistics
        # across orientations
        for orients in coeff[1:-1]:
            for (s1, s2) in list(itertools.combinations(orients, 2)):
                f.append((s1.real * s2.real).mean())

        for orient in range(len(coeff[1])):
            for height in range(len(coeff) - 3):
                s1 = coeff[height + 1][orient].real
                s2 = coeff[height + 2][orient].real

                s1 = cv2.resize(s1, (0, 0), fx=0.5, fy=0.5)
                f.append((s1 * s2).mean() / np.sqrt(s1.var()) / np.sqrt(s2.var()))
        return np.array(f)
