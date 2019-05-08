import argparse
import color_extraction
import cv2
import logging
import os
import perceptual
import pickle
import numpy as np

from numpy.linalg import inv
from perceptual.metric import Metric
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import mahalanobis

def main(opt):
    # Load or extract stsim vecors from grayscale images
    all_vectors = np.load(opt.stsim_file) if opt.stsim_file else extractVectors(opt)
    if opt.normalize:
        all_vectors = normalize(all_vectors, opt.normalize)
    stsim_vectors = {
        ind: np.array([vec for vec in all_vectors if vec[0] == ind])
        for ind in set(all_vectors[:,0])
    }

    # Extend stsim vectors to include their aca LAB color features
    if opt.aca_color_cnt:  
        stsim_vectors = color_extraction.extend_color_features(
            stsim_vectors,
            pickle.load(open('color_features.bin', 'rb')),
            opt.aca_color_cnt,  
            opt.aca_color_ordering,
            opt.aca_color_weighted)

    # Create opt.fold_cnt number of cross validation splits for each class
    validators = {
        ind: KFold(n_splits=opt.fold_cnt, shuffle=True).split(stsim_vectors[ind]) 
        for ind in stsim_vectors
    }
        
    # Train and evaluate each fold
    print (f'Evaluating {opt.scope} {opt.distance_metric}')
    results = []
    for i in range(opt.fold_cnt):
        print (f'Training CV Fold: {i}')
        train_split, test_split, class_matrix = [], [], []
        for img, vecs in stsim_vectors.items():
            # Get training and test splits for M matrix from cross validators
            train_ind, test_ind = next(validators[img])
            curr_train = np.asarray([vecs[i] for i in train_ind])
            curr_test = np.asarray([vecs[i] for i in test_ind])
            train_split.extend(curr_train)
            test_split.extend(curr_test)

            # Generate array of [var, cov, std] matrices from each class
            if opt.scope == 'cluster':
                cluster_model = KMeans(n_clusters=opt.cluster_cnt).fit(curr_train[:,1:])
                clusters = [
                    np.asarray([curr_train[ii] for ii in range(len(curr_train))
                        if cluster_model.labels_[ii] == j])
                    for j in range(opt.cluster_cnt)
                ]
                for c in clusters:
                    if opt.distance_metric =='var':
                        class_matrix.append(np.var(c[:,1:], axis=0))
                    elif opt.distance_metric =='std':
                        class_matrix.append(np.std(c[:,1:], axis=0))
                    elif opt.distance_metric =='cov' and len(c) >  1:
                        class_matrix.append(np.cov(c[:,1:], rowvar=False))
            elif opt.scope =='intraclass':
                if opt.distance_metric =='var':
                    class_matrix.append(np.var(curr_train[:,1:], axis=0))
                elif opt.distance_metric =='std':
                    class_matrix.append(np.std(curr_train[:,1:], axis=0))
                elif opt.distance_metric =='cov' and len(curr_train) >  1:
                    class_matrix.append(np.cov(curr_train[:,1:], rowvar=False))

        # Flatten training array for computing global M matrix for 
        train_split = np.asarray(train_split)
        if opt.scope =='global':
            if opt.distance_metric =='std':
                dist_matrix = np.diag(np.std(train_split[:,1:], axis=0))
            elif opt.distance_metric =='var':
                dist_matrix = np.diag(np.var(train_split[:,1:], axis=0))
            elif opt.distance_metric =='cov':
                dist_matrix = np.cov(np.array(train_split)[:,1:], rowvar=False)
        # Compute single M matrix from individual class M matrices
        elif opt.scope in ['intraclass', 'cluster']:
            if opt.distance_metric in ['std', 'var']:
                dist_matrix = np.diag(np.mean(class_matrix,axis=0))
            elif opt.distance_metric =='cov':
                dist_matrix= np.mean(class_matrix, axis=0)
        results.append(evaluate(dist_matrix, stsim_vectors, train_split, test_split, opt))
        print (results[-1])
    print (np.mean(results, axis=0))

def normalize(vectors, norm):
    if norm == 'z-norm':
        norm_const = np.std(vectors[:,1:], axis=0)
    elif norm == 'L2-norm':
        norm_const = [np.linalg.norm([vectors[j][i] for j in range(len(vectors))])
                        for i in range(1,len(vectors[0]))]
    vectors[:,1:] = vectors[:,1:]/norm_const
    return vectors

def extractVectors(dir):
    stsim_vectors = []
    for root, dirs, files in os.walk(dir):
        img_classes = set([img.split("-")[0] for img in files])
        img_enum = {img:cnt for cnt, img in enumerate(img_classes)}
        for img in files:
            img_class = img_enum[img.split("-")[0]]
            vec = list(Metric().STSIM_M(cv2.imread(os.path.join(root,img), cv2.IMREAD_GRAYSCALE)))
            stsim_vectors.append(np.asarray([img_class] + vec))
    np.save(open('identical_vectors.bin', 'wb'),stsim_vectors)
    return np.asarray(stsim_vectors)     

def evaluate(dist_matrix, stsim_vectors, train_split, test_split, opt):
    cluster_centers = {}
    exemplar_sim, predicted_label =  [], []
    dist_matrix = np.linalg.inv(dist_matrix)
    for img in stsim_vectors.keys():
        stsim_vectors = [vec[1:] for vec in train_split if vec[0] == img]
        if opt.cluster_method == 'gmm':
            cluster_model = GaussianMixture(n_components=opt.cluster_cnt).fit(stsim_vectors)
            cluster_centers[img] = cluster_model.means_
        elif opt.cluster_method == 'kmeans':
            cluster_model = KMeans(n_clusters=opt.cluster_cnt).fit(stsim_vectors)
            cluster_centers[img] = cluster_model.cluster_centers_
        elif opt.cluster_method == 'all_training':
            cluster_centers[img] = stsim_vectors

        if opt.exemplar == 'nearest_neighbor':
            neigh = NearestNeighbors(n_jobs=4).fit(stsim_vectors)
            ind = [
                neigh.kneighbors([cnt], n_neighbors=1, return_distance = False)[0][0]
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
        exemplar_sim = sorted(exemplar_sim, key = lambda x: x[1])
        predicted_label.append(exemplar_sim[0][0])
    return score(true_label, predicted_label, average='macro')[:-1]
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stsim_file', type=str)
    parser.add_argument('--image_dir', type=str)
    parser.add_argument('--normalize', choices = ['z-norm', 'L2-norm'])
    parser.add_argument('--distance_metric', choices = ['var', 'cov', 'std'], default='cov')
    parser.add_argument('--scope', choices=['global', 'intraclass', 'cluster'], default='intraclass')
    parser.add_argument('--exemplar', choices=['nearest_neighbor', 'cluster_center'], default='cluster_center')
    parser.add_argument('--cluster_method', choices = ['kmeans','gmm', 'all_training'], default='kmeans')
    parser.add_argument('--cluster_cnt', type=int, default=1)
    parser.add_argument('--aca_color_cnt', type=int, default=0)
    parser.add_argument('--aca_color_ordering', choices = ['luminance', 'composition'], default='luminance')
    parser.add_argument('--aca_color_weighted', type=bool, default=False)
    parser.add_argument('--fold_cnt', type=int, default=5)

    opt = parser.parse_args()
    main(parser.parse_args())