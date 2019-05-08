import argparse
import cv2
import numpy as np

from numpy.linalg import inv
from perceptual.metric import Metric
from scipy.spatial.distance import mahalanobis

def computeTextureSimilarity (opt):
    dist_matrix = np.linalg.inv(np.load(open(opt.load_distance_matrix, 'rb')))
    base_stsim = Metric().STSIM_M(cv2.imread(opt.base_img, cv2.IMREAD_GRAYSCALE))
    for distortion_type in opt.distortion_type:
        for img in range(opt.distortion_cnt):
                distorted_img = opt.base_img.split(".")[0] + "_" + distortion_type + str(img + 1) + ".tiff"
                distorted_stsim = Metric().STSIM_M(cv2.imread(distorted_img, cv2.IMREAD_GRAYSCALE))
                dist = mahalanobis(base_stsim, distorted_stsim, dist_matrix)
                print (f"{distorted_img}: {dist}\n")
                if opt.save_file: 
                        comp_img = distorted_img.split('/')[-1]
                        open(opt.save_file, 'a').write(f"{comp_img}:, {dist}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_img', type=str)
    parser.add_argument('--distortion_type', nargs='+')
    parser.add_argument('--distortion_cnt', type=int)
    parser.add_argument('--load_distance_matrix', type=str, default='')
    parser.add_argument('--save_file', type=str)
    computeTextureSimilarity(parser.parse_args())
