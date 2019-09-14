'''
Coding Experiments: STSIM-M
'''

import argparse
import os
import cv2
import numpy as np
from utils.metric import Metric
from scipy.spatial.distance import mahalanobis
from scipy.stats import spearmanr, pearsonr


def main(opt):
    if opt.image_dir:
        stsimM_vectors, stsimM_class = [], {}
        for root, dirs, files in os.walk(opt.image_dir):
            for base_texture in range(0,10):
                for distortion in range(0,10):
                    img = f"{base_texture}_{distortion}.tiff"
                    vector = list(Metric().STSIM_M(cv2.imread(os.path.join(root, img), cv2.IMREAD_GRAYSCALE)))
                    stsimM_vectors.append([int(base_texture + distortion)] + vector)
                    if int(base_texture) in stsimM_class:
                        stsimM_class[int(base_texture)].append(vector)
                    else: 
                        stsimM_class[int(base_texture)] = [vector]
    
    if opt.save_features: 
        np.save(open(opt.save_features, 'rb'), stsimM_vectors)
    
    if opt.load_distance_matrix:
        dist_matrix = np.load(open(opt.load_distance_matrix, 'rb'))
    elif opt.scope == 'global':
        if opt.distance_metric == 'var':
            dist_matrix = np.diag(np.var(stsimM_vectors[:, 1:], axis=0))
        elif opt.distance_metric == 'cov':
            import pdb; pdb.set_trace()
            dist_matrix = np.cov(np.array(stsimM_vectors)[:, 1:], rowvar=False)
    elif opt.scope == 'intraclass':
        if opt.distance_metric == 'var':
            dist_matrix = np.mean([
                np.diag(np.var(distortions), axis=0) 
                    for distortions in stsimM_class.values()
                ], axis=0)
        elif opt.distance_metric == 'cov':
            dist_matrix = np.mean([
                np.cov(distortions, rowvar=False)
                for distortions in stsimM_class.values()
            ], axis=0)


    if opt.save_distance_matrix:
        np.save(open(opt.save_distance_matrix, 'wb'), dist_matrix)
    dist_matrix = np.linalg.inv(dist_matrix)

    results = []
    for base_texture in range(0,10):
        texture_sim = [
            mahalanobis(
                stsimM_class[base_texture][0],
                stsimM_class[base_texture][distortion],
                dist_matrix)
            for distortion in range(0, 10)
        ]
        results.append(texture_sim)
    if opt.save_results:
        np.save(open(opt.save_results, 'wb'), np.array(results))

    evaluate(results, opt)

def evaluate(results, opt):
    subj_results = np.load(opt.load_subjective_ranks)
    
    # Bordas Rule
    texture_pearson, texture_spearman = [], []
    for base_texture in range(len(results)):
        subj_sorted = {val: ind for ind, val in enumerate(sorted(subj_results[base_texture]))}
        subj_rank = [subj_sorted[dist] for dist in subj_results[base_texture]]

        metric_sorted = {val: ind for ind, val in enumerate(sorted(results[base_texture]))}
        metric_rank = [metric_sorted[dist] for dist in results[base_texture]]
        
        print("subj", subj_rank)
        print("metric", metric_rank)
        print()
        texture_pearson.append(pearsonr(subj_rank, metric_rank))
        texture_spearman.append(spearmanr(subj_rank, metric_rank))
        
    bordas_pearson = np.mean(texture_pearson)
    bordas_spearman = np.mean(texture_spearman)

    print(bordas_pearson, bordas_spearman)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_distance_matrix', type=str, default='')
    parser.add_argument('--load_subjective_ranks', type=str)
    parser.add_argument('--image_dir', type=str)
    parser.add_argument('--save_features', type=str, default='')
    parser.add_argument('--save_distance_matrix', type=str, default='')
    parser.add_argument('--save_results', type=str, default='')
    parser.add_argument(
        '--distance_metric', choices=['var', 'cov', 'std'], default='cov')
    parser.add_argument(
        '--scope', choices=['global', 'intraclass'], default='intraclass')

    opt = parser.parse_args()
    main(parser.parse_args())
