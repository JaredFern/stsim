'''
Retrieval Experiments: STSIM-M
0. 5-fold CV train-test splits
1. Extract STSIM-M vectors. (Opt. Add ACA color features)
2. Compute k means clusters for each class of textures.
3. Compute covariance matrix for calculating Mahalanobis distance
    - Global: Covariance matrix over all training vectors
    - Intraclass: Average covariance matrix over all training vectors
    - Cluster: Average covariance matrix over all clusters for each class
4. Retrieval: Determine class of test texture as exemplar with minimal distance from test vector
'''

import argparse
import color_extraction
import cv2
import os
import perceptual
import pickle
import numpy as np

from numpy.linalg import inv
from perceptual.metric import Metric
from sklearn.model_selection import KFold, LeaveOneOut, train_test_split
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from skimage import color
from scipy.ndimage import imread
from collections import Counter
from tqdm import tqdm
from scipy.spatial.distance import mahalanobis


def main(opt):
    # Load or extract stsim vecors from grayscale images
    all_vectors = np.load(
        opt.stsim_features) if opt.stsim_features else extractVectors(opt)
    if opt.normalize:
        all_vectors = normalize(all_vectors, opt.normalize)
    stsim_vectors = {
        ind: np.array([vec for vec in all_vectors if vec[0] == ind and np.nan not in vec])
        for ind in set(all_vectors[:, 0])
    }
    if opt.class_cnt:
        for ind in set(all_vectors[:,0]):
            if len(stsim_vectors[ind]) < opt.class_cnt:
                stsim_vectors.pop(ind, None)
            elif len(stsim_vectors[ind]) > opt.class_cnt:
                stsim_vectors[ind] = stsim_vectors[ind][:opt.class_cnt]


    # Create opt.fold_cnt number of cross validation splits for each class
    if opt.test_split:
        validators = {
            ind:train_test_split(stsim_vectors[ind], test_size=opt.test_split)
            for ind in stsim_vectors.keys()
        }

    elif opt.fold_cnt:
        validators = {
            ind: KFold(n_splits=opt.fold_cnt, shuffle=True).split(
                stsim_vectors[ind])
            for ind in stsim_vectors.keys()
        }

    # Train and evaluate each fold
    print(f'Evaluating {opt.scope} {opt.distance_metric}')
    results = []
    folds = opt.fold_cnt if opt.fold_cnt > 0 else 1
    for i in range(folds):
        print(f'Training CV Fold: {i}')
        train_split, test_split, class_matrix = [], [], []
        for img, vecs in stsim_vectors.items():
            # Get training and test splits for M matrix from cross validators
            if opt.fold_cnt:
                train_ind, test_ind = next(validators[img])
                curr_train = np.asarray([vecs[i] for i in train_ind])
                curr_test = np.asarray([vecs[i] for i in test_ind])
                train_split.extend(curr_train)
                test_split.extend(curr_test)
            if opt.test_split :
                curr_train = validators[img][0]
                curr_test = validators[img][1]
                train_split.extend(curr_train)
                test_split.extend(curr_test)

            # Generate array of [var, cov, std] matrices from each class
            if opt.scope == 'cluster':
                cluster_model = KMeans(n_clusters=opt.cluster_cnt).fit(curr_train[:, 1:])
                clusters = [
                    np.asarray([curr_train[ii] for ii in range(len(curr_train))
                                if cluster_model.labels_[ii] == j])
                    for j in range(opt.cluster_cnt)
                ]
                for c in clusters:
                    if opt.distance_metric == 'var':
                        class_matrix.append(np.var(c[:, 1:], axis=0))
                    elif opt.distance_metric == 'std':
                        class_matrix.append(np.std(c[:, 1:], axis=0))
                    elif opt.distance_metric == 'cov' and len(c) > 1:
                        class_matrix.append(np.cov(c[:, 1:], rowvar=False))
            elif opt.scope == 'intraclass':
                if opt.distance_metric == 'var':
                    class_matrix.append(np.var(curr_train[:, 1:], axis=0))
                elif opt.distance_metric == 'std':
                    class_matrix.append(np.std(curr_train[:, 1:], axis=0))
                elif opt.distance_metric == 'cov' and len(curr_train) > 1:
                    class_matrix.append(
                        np.cov(curr_train[:, 1:], rowvar=False))

        # Flatten training array for computing global M matrix for
        train_split = np.asarray(train_split)
        if opt.scope == 'global':
            if opt.distance_metric == 'std':
                dist_matrix = np.diag(np.std(train_split[:, 1:], axis=0))
            elif opt.distance_metric == 'var':
                dist_matrix = np.diag(np.var(train_split[:, 1:], axis=0))
            elif opt.distance_metric == 'cov':
                dist_matrix = np.cov(train_split[:, 1:], rowvar=False)
        # Compute single M matrix from individual class M matrices
        elif opt.scope in ['intraclass', 'cluster']:
            if opt.distance_metric in ['std', 'var']:
                dist_matrix = np.diag(np.mean(class_matrix, axis=0))
            elif opt.distance_metric == 'cov':
                dist_matrix = np.mean(class_matrix, axis=0)
        if opt.distance_matrix:
            dist_matrix = np.load(open(opt.distance_matrix, 'rb'))
        if opt.save_distance_matrix:
            np.save(open(opt.save_distance_matrix, 'wb'), dist_matrix)

        if opt.exemplars:
            results.append(evaluate(dist_matrix, stsim_vectors, train_split, test_split, opt))
        else:
            results.append(evaluate_no_exemplars(dist_matrix, stsim_vectors, train_split, test_split))

        print(results[-1])
    print(np.mean(results, axis=0))


def normalize(vectors, norm):
    if norm == 'z-norm':
        norm_const = np.std(vectors[:, 1:], axis=0)
    elif norm == 'L2-norm':
        norm_const = [np.linalg.norm([vectors[j][i] for j in range(len(vectors))])
                      for i in range(1, len(vectors[0]))]
    vectors[:, 1:] = vectors[:, 1:]/norm_const
    return vectors


def extractVectors(opt):
    stsim_vectors = []
    for root, dirs, files in os.walk(opt.image_dir):
        img_classes = set([img.split("-")[0] for img in files])
        img_enum = {img: cnt for cnt, img in enumerate(img_classes)}
        for img in files:
            img_class = img_enum[img.split("-")[0]]
            try:
                vec = list(Metric().STSIM_M(cv2.imread(os.path.join(root, img), cv2.IMREAD_GRAYSCALE)))
            except: import pdb; pdb.set_trace()
            # Extend stsim vectors to include their aca LAB color features
            if opt.aca_color_cnt:
                pixel_cnt = Counter()
                color_path = os.path.join(opt.color_dir, img.split(".")[0] + '_lav2.tiff' )
                img_lab = color.rgb2lab(imread(color_path))
                for i in range(len(img_lab)):
                    for j in range(len(img_lab[i])):
                        pixel_cnt[tuple(img_lab[i][j])] += 1

                color_features = [i[0] for i in pixel_cnt.most_common(opt.aca_color_cnt)]
                if len(color_features) < opt.aca_color_cnt:
                    color_features.extend(
                        [(0, 0, 0) for i in range(opt.aca_color_cnt - len(color_features))])
                color_features = [val for color in color_features for val in color]
                vec += color_features

            stsim_vectors.append(np.asarray([img_class] + vec))


    if opt.save_features:
        print(f"STSIM features saved as {opt.save_features}")
        np.save(open(opt.save_features, 'wb'), stsim_vectors)
    return np.asarray(stsim_vectors)

def evaluate_no_exemplars(dist_matrix, stsim_vectors, train_split, test_split):
    precision_1, mrr, avg_prec = np.zeros(len(test_split)),np.zeros(len(test_split)), np.zeros(len(test_split))
    most_sim = []
    for test_idx, test_img in enumerate(test_split):
        similarities = np.ndarray((len(train_split), 2))
        for train_idx, train_img in  enumerate(train_split):
            distance = mahalanobis(test_img[1:], train_img[1:], dist_matrix)
            similarities[train_idx]=[train_img[0], distance]

        ranked_similarities = similarities[similarities[:,1].argsort()]
        if ranked_similarities[0][0] == test_img[0]:
            precision_1[test_idx] = 1
        for sim_idx, sim in enumerate(ranked_similarities):
            if sim[0] == test_img[0]:
                mrr[test_idx] = 1/(1+sim_idx)
                break

        precision_cnt = 0
        map_precisions = np.zeros(len(ranked_similarities))
        for sim_idx, sim in enumerate(ranked_similarities):
            if sim[0] == test_img[0]: precision_cnt += 1
            map_precisions[sim_idx] = precision_cnt/(sim_idx + 1)
        avg_prec[test_idx] = np.mean(map_precisions)
        most_sim.append(ranked_similarities[0][0])
    import pdb; pdb.set_trace()
    return [np.mean(precision_1), np.mean(mrr), np.mean(avg_prec)]


def evaluate(dist_matrix, stsim_vectors, train_split, test_split, opt):
    cluster_centers = {}
    exemplar_sim, predicted_label = [], []
    dist_matrix = np.linalg.inv(dist_matrix)
    for img in stsim_vectors.keys():
        stsim_vectors = [vec[1:] for vec in train_split if vec[0] == img]
        if opt.cluster_method == 'gmm':
            cluster_model = GaussianMixture(
                n_components=opt.cluster_cnt).fit(stsim_vectors)
            cluster_centers[img] = cluster_model.means_
        elif opt.cluster_method == 'kmeans':
            cluster_model = KMeans(
                n_clusters=opt.cluster_cnt).fit(stsim_vectors)
            cluster_centers[img] = cluster_model.cluster_centers_
        elif opt.cluster_method == 'all_training':
            cluster_centers[img] = stsim_vectors

        if opt.exemplar == 'nearest_neighbor':
            neigh = NearestNeighbors(n_jobs=6).fit(stsim_vectors)
            ind = [
                neigh.kneighbors([cnt], n_neighbors=1,
                                 return_distance=False)[0][0]
                for cnt in cluster_centers[img]
            ]
            cluster_centers[img] = [stsim_vectors[i] for i in ind]

    true_label = [i[0] for i in test_split]
    for test_vec in test_split:
        exemplar_sim = []
        for img, centers in cluster_centers.items():
            for c in centers:
                distance = mahalanobis(test_vec[1:], c, dist_matrix)
                exemplar_sim.append([img, distance])
        exemplar_sim = sorted(exemplar_sim, key=lambda x: x[1])
        predicted_label.append(exemplar_sim[0][0])
    return score(true_label, predicted_label, average='macro')[:-1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stsim_features', type=str, default='')
    parser.add_argument('--distance_matrix', type=str, default='')
    parser.add_argument('--image_dir', type=str)
    parser.add_argument('--color_dir', type=str)
    parser.add_argument('--save_features', type=str, default='')
    parser.add_argument('--save_distance_matrix', type=str, default='')
    parser.add_argument('--normalize', choices=['z-norm', 'L2-norm'])
    parser.add_argument(
        '--distance_metric', choices=['var', 'cov', 'std'], default='cov')
    parser.add_argument('--exemplars', type=bool, default=False)
    parser.add_argument(
        '--scope', choices=['global', 'intraclass', 'cluster'], default='intraclass')
    parser.add_argument(
        '--exemplar', choices=['nearest_neighbor', 'cluster_center'], default='cluster_center')
    parser.add_argument(
        '--cluster_method', choices=['kmeans', 'gmm', 'all_training'], default='kmeans')
    parser.add_argument('--cluster_cnt', type=int, default=1)
    parser.add_argument('--aca_color_cnt', type=int, default=0)
    parser.add_argument(
        '--aca_color_ordering',choices=['luminance', 'composition'], default='luminance')
    parser.add_argument('--aca_color_weighted', type=bool, default=False)
    parser.add_argument('--fold_cnt', type=int, default=0)
    parser.add_argument('--test_split', type=float, default=0.20)
    parser.add_argument('--class_cnt', type=int, default=16)
    opt = parser.parse_args()
    main(parser.parse_args())
