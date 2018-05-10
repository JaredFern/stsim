import argparse
import numpy as np
from numpy.linalg import inv
from functools import partial
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.neighbors import NearestNeighbors

def main(opt):
    all_vectors = np.load('curet-vectors.bin')
    # Normalize features by opt.normalize arg
    if opt.normalize: all_vectors = normalize(all_vectors, opt.normalize)

    # Split feature vectors into classes
    class_vectors = {ind: [vec for vec in all_vectors if vec[0] == ind] for ind in range(1,59)}

    # Create cross-validators for each class
    validators = {ind: KFold(n_splits=opt.fold_cnt, shuffle=True).split(class_vectors[ind]) for ind in class_vectors}

    std_results, var_results, cov_results = [], [], []
    class_var_results, class_std_results, class_cov_results = [], [], []

    # Train and evaluate each fold
    for i in range(opt.fold_cnt):
        print (f'Training CV Fold: {i}')
        train_split, test_split = [], []
        class_std, class_var, class_cov = [], [], []

        # For each class grab next cross validation fold
        for img, vecs in class_vectors.items():
            train_ind, test_ind = next(validators[img])
            curr_train = np.asarray([vecs[i] for i in train_ind])
            curr_test = np.asarray([vecs[i] for i in test_ind])

            # Calculate var, std, cov matrices
            class_var.append(np.var(curr_train[:,1:], axis=0))
            class_std.append(np.std(curr_train[:,1:], axis=0))
            class_cov.append(np.cov(curr_train[:,1:], rowvar=False))

            train_split.extend(curr_train)
            test_split.extend(curr_test)

        train_split = np.asarray(train_split)
        all_std = np.diag(np.std(train_split[:,1:], axis=0))
        all_var = np.diag(np.var(train_split[:,1:], axis=0))
        all_cov = np.cov(np.array(train_split)[:,1:], rowvar=False)

        class_std = np.diag(np.mean(class_std,axis=0))
        class_var = np.diag(np.mean(class_var,axis=0))
        class_cov = np.mean(class_cov, axis=0)

        partial_eval = partial(evaluate,train_split=train_split, test_split=test_split,
                                cluster_method=opt.cluster_method,
                                cluster_cnt=opt.cluster_cnt)

        print ('Evaluating Global Variance')
        var_results.append(partial_eval(all_var))
        print ('Evaluating Global StdDev')
        std_results.append(partial_eval(all_std))
        print ('Evaluating Global Covariance')
        cov_results.append(partial_eval(all_std))
        print ('Evaluating Intraclass Variance')
        class_var_results.append(partial_eval(class_var))
        print ('Evaluating Intraclass StdDev')
        class_std_results.append(partial_eval(class_std))
        print ('Evaluating Intraclass Covariance')
        class_cov_results.append(partial_eval(class_cov))

    var_cv = np.mean(var_results, axis=0)
    std_cv = np.mean(std_results, axis=0)
    cov_cv = np.mean(cov_results, axis=0)

    class_var_cv = np.mean(class_var_results, axis=0)
    class_std_cv = np.mean(class_std_results, axis=0)
    class_cov_cv = np.mean(class_cov_results, axis=0)

def normalize(vectors, norm):
    if norm == 'z-norm':
        norm_const = np.std(vectors[:,1:], axis=0)
    elif norm == 'L2-norm':
        norm_const = [np.linalg.norm([vectors[j][i] for j in range(len(vectors))])
                        for i in range(1,len(vectors[0]))]
    vectors[:,1:] = vectors[:,1:]/norm_const
    return vectors

def evaluate(dist_matrix, train_split=None, test_split=None, cluster_method='kmeans', cluster_cnt=5):
    cluster_centers, cluster_nn = {}, {}
    exemplar_sim, true_label, predicted_label = [], [], []
    most_sim, sim_vec  = [], []
    dist_matrix = np.linalg.inv(dist_matrix)
    for img in range(1,59):
        class_vectors = [vec[1:] for vec in train_split if vec[0] == img]
        if cluster_method == 'gmm':
            cluster_model = GaussianMixture(n_components=cluster_cnt).fit(class_vectors)
            cluster_centers[img] = cluster_model.means_
        elif cluster_method == 'kmeans':
            cluster_model = KMeans(n_clusters=cluster_cnt).fit(class_vectors)
            cluster_centers[img] = cluster_model.cluster_centers_

        neigh = NearestNeighbors(n_jobs=4).fit(class_vectors)
        cluster_nn[img] = [neigh.kneighbors([cnt],  n_neighbors = 1)
                           for cnt in cluster_centers[img]]

    for test_vec in test_split:
        for img, centers in cluster_centers.items():
            for c in centers:
                distance = np.matmul(np.matmul(np.transpose(test_vec[1:]-c),
                                               dist_matrix), test_vec[1:] - c)
                exemplar_sim.append([img, distance, c])
        exemplar_sim = sorted(exemplar_sim, key = lambda x: x[1],)
        sim_vec.append(exemplar_sim[0][2])
        predicted_label.append(exemplar_sim[0][0])
        true_label.append(test_vec[0])

    precision, recall, f_score, support = score(true_label, predicted_label, average='macro')
    print (precision, recall, f_score)
    import pdb; pdb.set_trace()
    return [precision, recall, f_score]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--normalize', choices = ['z-norm', 'L2-norm'], nargs='*')
    parser.add_argument('--distance_metric', choices = ['var', 'covar', 'std'], nargs='*')
    parser.add_argument('--cluster_method', choices = ['kmeans','gmm'], default='kmeans')
    parser.add_argument('--cluster_cnt', type=int, default=5)
    parser.add_argument('--fold_cnt', type=int, default=10)

    opt = parser.parse_args()
    main(parser.parse_args())
