import numpy as np
import multiprocessing as mp

import argparse
import cv2

from TextureFeature import Ssim, Stsim1, Stsim2, StsimC, LocalBinaryPattern

from functools import partial
from skimage import color
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

def _compute_similarities(texture_feature, train_split, test_idx):
    if texture_feature.cluster_labels:
        train_labels = texture_feature.cluster_labels
        similarities = np.zeros(len(texture_feature.cluster_centers))
        for jj, cluster_center in enumerate(texture_feature.cluster_centers):
            similarities[jj] = texture_feature.compute_similarity(
                texture_feauture.features[test_idx],
                cluster_center)
    else:
        train_labels = [texture_feature.labels[idx] for idx in train_split]
        similarities = np.zeros(len(train_split))
        for jj, train_idx in enumerate(train_split):
            similarities[jj] = texture_feature.compute_similarity(
                texture_feature.features[test_idx],
                texture_feature.features[train_idx])

    sorted_sim_idx = similarities.argsort()
    predicted_label = train_labels[sorted_sim_idx[0]]
    precision_1 = 1 if str(predicted_label) == texture_feature.labels[test_idx] else 0

    mrr = 0
    for jj, idx in enumerate(sorted_sim_idx):
        if train_labels[idx] == texture_feature.labels[test_idx]:
            mrr = 1.0 / (jj + 1.0)
            break
    return predicted_label, precision_1, mrr


# TODO: Loop optimizations for pooling computations
def evaluate(opt, texture_feature, train_split, test_split):
    test_labels = [texture_feature.labels[idx] for idx in test_split]
    if opt.metric == 'stsim_c':
        texture_feature.compute_mahalanobis_matrix(train_split)
    if opt.cluster_method:
        texture_feature.cluster_examples(train_split, opt.cluster_method, opt.cluster_cnt)

    with mp.Pool(opt.concurrent_eval_processes) as pool:
        partial_compute_sim = partial(_compute_similarities, texture_feature, train_split)
        predicted_labels, prec_1, mrr = zip(*pool.map(partial_compute_sim, test_split.tolist()))

    _, recall, f1, _ = precision_recall_fscore_support(test_labels, predicted_labels,
                                                       average='micro')
    print(f"P@1: {np.mean(prec_1)}, MRR: {np.mean(mrr)}, Recall: {recall}, F1:{f1}")
    return np.mean(prec_1), np.mean(mrr), recall, f1


def main(opt):
    if opt.metric == 'ssim':
        texture_features = Ssim(opt.image_dir_or_saved_bin, opt.class_cnt, opt.save_features)
    elif opt.metric == 'stsim1':
        texture_features = Stsim1(opt.image_dir_or_saved_bin, opt.class_cnt, opt.save_features)
    elif opt.metric == 'stsim2':
        texture_features = Stsim2(opt.image_dir_or_saved_bin, opt.class_cnt, opt.save_features)
    elif opt.metric == 'stsim_c':
        texture_features = StsimC(opt.image_dir_or_saved_bin, opt.class_cnt, opt.save_features,
                                  opt.mahalanobis_file, opt.mahalanobis_type,
                                  opt.mahalanobis_scope, opt.aca_color_cnt, opt.color_dir)
    elif opt.metric == 'lbp':
        texture_features = LocalBinaryPattern(opt.image_dir_or_saved_bin, opt.class_cnt,
                                              opt.save_features, opt.n_points, opt.radius,
                                              opt.lbp_method)

    # Train and evaluate each fold
    results = []
    validator = StratifiedKFold(n_splits=opt.fold_cnt, shuffle=True).split(
        texture_features.features, texture_features.labels)
    for i, (train_split, test_split) in enumerate(validator):
        print(f'Training CV Fold: {i}')
        results.append(evaluate(opt, texture_features, train_split, test_split))
    print(np.mean(results, axis=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Metric experimental configs
    parser.add_argument('--metric', choices=['ssim', 'stsim1', 'stsim2', 'stsim_c', 'lbp'])
    parser.add_argument('--image_dir_or_saved_bin', type=str)
    parser.add_argument('--class_cnt', type=int, default=16)
    parser.add_argument('--save_features', type=str)
    parser.add_argument('--fold_cnt', type=int, default=5)
    parser.add_argument('--concurrent_eval_processes', type=int, default=2)

    # Exemplar Clustering configs (defaults to off)
    parser.add_argument('--cluster_method', choices=['kmeans', 'gmm'], default=None)
    parser.add_argument('--cluster_cnt', type=int, default=1)

    # Configs for STSIM-C metrics
    parser.add_argument('--mahalanobis_file', type=str)
    parser.add_argument('--mahalanobis_type', choices=['std', 'var', 'cov'], default='cov')
    parser.add_argument('--mahalanobis_scope', choices=['intraclass', 'global'],
                        default='intraclass')
    parser.add_argument('--aca_color_cnt', type=int, default=0)
    parser.add_argument('--color_dir', type=str)

    # Configs for LBP metrics
    parser.add_argument('--n_points', type=int, default=8)
    parser.add_argument('--radius', type=int, default=2)
    parser.add_argument('--lbp_method', choices=['default', 'ror', 'uniform', 'nri_uniform',
                                                 'var'], default='uniform')

    opt = parser.parse_args()
    main(parser.parse_args())
