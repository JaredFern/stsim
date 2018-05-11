import argparse, logging
import numpy as np
from numpy.linalg import inv
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import mahalanobis

def main(opt):
    all_vectors = np.load('curet-vectors.bin')
    if opt.normalize: all_vectors = normalize(all_vectors, opt.normalize)
    class_vectors = {ind: [vec for vec in all_vectors if vec[0] == ind] for ind in range(1,59)}
    validators = {ind: KFold(n_splits=opt.fold_cnt, shuffle=True).split(class_vectors[ind]) for ind in class_vectors}

    # Train and evaluate each fold
    print (f'Evaluating {opt.scope} {opt.distance_metric}')
    results = []
    for i in range(opt.fold_cnt):
        print (f'Training CV Fold: {i}')
        train_split, test_split, class_matrix = [], [], []
        for img, vecs in class_vectors.items():
            train_ind, test_ind = next(validators[img])
            curr_train = np.asarray([vecs[i] for i in train_ind])
            curr_test = np.asarray([vecs[i] for i in test_ind])
            train_split.extend(curr_train)
            test_split.extend(curr_test)

            if opt.scope =='intraclass':
                if opt.distance_metric =='var':
                    class_matrix.append(np.var(curr_train[:,1:], axis=0))
                elif opt.distance_metric =='std':
                    class_matrix.append(np.std(curr_train[:,1:], axis=0))
                elif opt.distance_metric =='cov':
                    class_matrix.append(np.cov(curr_train[:,1:], rowvar=False))

        train_split = np.asarray(train_split)
        if opt.scope =='global':
            if opt.distance_metric =='std':
                dist_matrix = np.diag(np.std(train_split[:,1:], axis=0))
            elif opt.distance_metric =='var':
                dist_matrix = np.diag(np.var(train_split[:,1:], axis=0))
            elif opt.distance_metric =='cov':
                dist_matrix = np.cov(np.array(train_split)[:,1:], rowvar=False)
        elif opt.scope =='intraclass':
            if opt.distance_metric in ['std', 'var']:
                dist_matrix = np.diag(np.mean(class_matrix,axis=0))
            elif opt.distance_metric =='cov':
                dist_matrix= np.mean(class_matrix, axis=0)
        results.append(evaluate(dist_matrix, train_split, test_split, opt))
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

def evaluate(dist_matrix, train_split, test_split, opt):
    cluster_centers, cluster_nn = {}, {}
    exemplar_sim, predicted_label =  [], []

    dist_matrix = np.linalg.inv(dist_matrix)
    for img in range(1,59):
        class_vectors = [vec[1:] for vec in train_split if vec[0] == img]
        if opt.cluster_method == 'gmm':
            cluster_model = GaussianMixture(n_components=opt.cluster_cnt).fit(class_vectors)
            cluster_centers[img] = cluster_model.means_
        elif opt.cluster_method == 'kmeans':
            cluster_model = KMeans(n_clusters=opt.cluster_cnt).fit(class_vectors)
            cluster_centers[img] = cluster_model.cluster_centers_

        if opt.exemplar == 'nearest_neighbor':
            neigh = NearestNeighbors(n_jobs=4).fit(class_vectors)
            ind = [neigh.kneighbors([cnt], n_neighbors = 1, return_distance = False)[0][0]
                        for cnt in cluster_centers[img]]
            cluster_centers[img] = [class_vectors[i] for i in ind]

    true_label = [i[0] for i in test_split]
    for test_vec in test_split:
        exemplar_sim = []
        for img, centers in cluster_centers.items():
            for c in centers:
                distance = mahalanobis(test_vec[1:], c, dist_matrix)
                exemplar_sim.append([img, distance])
        exemplar_sim = sorted(exemplar_sim, key = lambda x: x[1])
        predicted_label.append(exemplar_sim[0][0])
    results = score(true_label, predicted_label, average='macro')[:-1]
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--normalize', choices = ['z-norm', 'L2-norm'])
    parser.add_argument('--distance_metric', choices = ['var', 'cov', 'std'])
    parser.add_argument('--scope', choices=['global', 'intraclass'])
    parser.add_argument('--exemplar', choices=['nearest_neighbor', 'cluster_center'])
    parser.add_argument('--cluster_method', choices = ['kmeans','gmm'], default='kmeans')
    parser.add_argument('--cluster_cnt', type=int, default=5)
    parser.add_argument('--fold_cnt', type=int, default=10)

    opt = parser.parse_args()
    main(parser.parse_args())
