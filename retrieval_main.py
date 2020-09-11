import numpy as np
import multiprocessing as mp

import argparse
import cv2

from TextureFeature import Ssim, Stsim1, Stsim2, StsimC, LocalBinaryPattern

from skimage import color
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import precision_recall_fscore_support


# TODO: Multiprocess similarity computations
# TODO: Loop optimizations for pooling computations
def evaluate(opt, texture_feature):
    train_split, test_split = next(texture_feature.validator)
    test_labels = [texture_feature.labels[idx] for idx in test_split]

    if opt.metric == 'stsim_c':
        texture_feature.compute_mahalanobis_matrix(train_split)
    if opt.cluster_method:
        train_labels, cluster_centers = cluster_examples(opt, train_split)
    else:
        train_labels = [texture_feature.labels[idx] for idx in train_split]

    precision_1, mrr, predicted_labels = np.zeros(len(test_split)), np.zeros(len(test_split)),[]
    for ii, test_idx in enumerate(test_split):
        if opt.cluster_method:
            similarities = np.zeros(len(cluster_centers))
            for jj, cluster_center in enumerate(cluster_centers):
                similarities[ii] = texture_feature.compute_similarity(
                    texture_feauture.features[test_idx],
                    cluster_center)
        else:
            similarities = np.zeros(len(train_split))
            for jj, train_idx in enumerate(train_split):
                similarities[jj] = texture_feature.compute_similarity(
                    texture_feature.features[test_idx],
                    texture_feature.features[train_idx])

        if opt.metric in ('lbp', 'stsim_c'):
            sorted_sim_idx = similarities.argsort()
        else:
            sorted_sim_idx = similarities.argsort()[::-1]
        predicted_labels.append(train_labels[sorted_sim_idx[0]])
        for jj, idx in enumerate(sorted_sim_idx):
            if train_labels[idx] == test_labels[ii]:
                mrr[ii] = 1.0/(jj + 1.0)
                break
    prec, recall, f1, _ = precision_recall_fscore_support(test_labels, predicted_labels,
                                                       average='micro')
    avg_mrr = np.mean(mrr)
    print("P@1: " + str(prec) + " MRR: "+ str(avg_mrr))
    return prec, avg_mrr, recall, f1


def cluster_examples(texture_feature, train_split):
    cluster_labels, cluster_centers = [], []
    for img in set(texture_feature.labels):
        img_features = [texture_feature.feature[idx]
                        for idx in train_split if texture_feature.labels[idx] == img]
        if opt.cluster_method == 'gmm':
            cluster_model = GaussianMixture(n_components=opt.cluster_cnt).fit(img_features)
            for center in cluster_model.means_:
                cluster_centers.append(center)
                cluster_labels.append(img)
        elif opt.cluster_method == 'kmeans':
            cluster_model = KMeans(n_clusters=opt.cluster_cnt).fit(img_features)
            for center in cluster_model.cluster_centers_:
                cluster_centers.append(center)
                cluster_labels.append(img)
    return cluster_labels, cluster_centers


def main(opt):
    if opt.metric == 'ssim':
        texture_features = Ssim(opt.image_dir_or_saved_bin, opt.class_cnt, opt.save_features,
                                opt.scale_invariance, opt.fold_cnt)
    elif opt.metric == 'stsim1':
        texture_features = Stsim1(opt.image_dir_or_saved_bin, opt.class_cnt, opt.save_features,
                                  opt.scale_invariance, opt.fold_cnt)
    elif opt.metric == 'stsim2':
        texture_features = Stsim2(opt.image_dir_or_saved_bin, opt.class_cnt, opt.save_features,
                                  opt.scale_invariance, opt.fold_cnt)
    elif opt.metric == 'stsim_c':
        texture_features = StsimC(opt.image_dir_or_saved_bin, opt.class_cnt, opt.save_features,
                                  opt.scale_invariance, opt.fold_cnt, opt.mahalanobis_file, 
                                  opt.mahalanobis_type, opt.mahalanobis_scope, 
                                  opt.aca_color_cnt, opt.color_dir)
    elif opt.metric == 'lbp':
        texture_features = LocalBinaryPattern(opt.image_dir_or_saved_bin, opt.class_cnt,
                                              opt.save_features, opt.scale_invariance, 
                                              opt.fold_cnt, opt.n_points, opt.radius, 
                                              opt.lbp_method)

    # Train and evaluate each fold
    folds = opt.fold_cnt if opt.fold_cnt > 0 else 1
    results = []
    for i in range(folds):
        results.append(evaluate(opt, texture_features))
    print(np.mean(results, axis=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Metric experimental configs
    parser.add_argument('--metric', choices=['ssim', 'stsim1', 'stsim2', 'stsim_c', 'lbp'])
    parser.add_argument('--image_dir_or_saved_bin', type=str)
    parser.add_argument('--class_cnt', type=int, default=-1)
    parser.add_argument('--save_features', type=str)
    parser.add_argument('--fold_cnt', type=int, default=5)
    parser.add_argument('--scale_invariance', type=bool, default=False)

    # Exemplar Clustering configs (defaults to off)
    parser.add_argument('--cluster_method', choices=['kmeans', 'gmm'], default=None)
    parser.add_argument('--cluster_cnt', type=int, default=1)

    # Configs for STSIM-C metrics
    parser.add_argument('--mahalanobis_file', type=str)
    parser.add_argument('--mahalanobis_type', choices=['std', 'var', 'cov'], default='cov')
    parser.add_argument('--mahalanobis_scope', choices=['intraclass', 'global', 'stsim_i'],
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
