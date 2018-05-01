import argparse
import numpy as np
from numpy.linalg import inv
from functools import partial
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, SpectralClustering

def main(opt):
    all_vectors = np.load('curet-vectors.bin')
    if opt.normalize:
        all_vectors = normalize(all_vectors, opt.normalize)

    class_vectors = {ind: [vec for vec in all_vectors if vec[0] == ind] for ind in range(1,59)}
    validators = {ind: KFold(n_splits=opt.fold_cnt, shuffle=True).split(class_vectors[ind]) for ind in class_vectors}
    std_results, var_results, cov_results = [], [], []

    for i in range(opt.fold_cnt):
        train_split, test_split = [], []
        class_std, class_var, class_cov = [], [], []

        for img, vecs in class_vectors.items():
            train_ind, test_ind = next(validators[img])
            curr_train = [vecs[i] for i in train_ind]
            train_split.extend(curr_train[img])
            test_split.extend([vecs[i] for i in test_ind])

            curr_var = [np.var([curr_train[j][i] for j in range(len(curr_train))])
                            for i in range(1,len(curr_train[0]))]
            curr_std = [np.std([curr_train[j][i] for j in range(len(curr_train))])
                            for i in range(1,len(curr_train[0]))]
            class_var.append(curr_var)
            class_std.append(curr_std)
            class_cov.append(np.cov(curr_train[:][1:], rowvar=False))

        all_var = [np.var([train_split[j][i] for j in range(len(train_split))])
                    for i in range(1, len(train_split[0]))]
        all_std = [np.std([train_split[j][i] for j in range(len(train_split))])
                    for i in range(1, len(train_split[0]))]

        all_cov = np.cov(train_split[:][1:], rowvar=False)
        all_var = np.diag(variance)
        all_std = np.diag(std_dev)

        class_std = np.diag(np.mean(class_std, axis=0))
        class_var = np.diag(np.mean (class_var, axis=0))
        class_cov = np.mean(class_cov, axis=0)

        partial_eval = evaluate(train_split=train_split, test_split=test_split,
                                cluster_method=opt.cluster_method,
                                cluster_cnt=opt.cluster_cnt)

        all_var_results.append(partial_eval(var_matrix))
        all_std_results.append(partial_eval(std_matrix))
        all_cov_results.append(partial_eval(cov_matrix))

        class_var_results.append(partial_eval(var_matrix))
        class_std_results.append(partial_eval(std_matrix))
        class_cov_results.append(partial_eval(cov_matrix))

def normalize(norm, vectors):
    if norm == 'z-norm':
        z_norm = [np.std([vectors[j][i] for j in range(len(vectors))])
                        for i in range(1,len(vectors[0]))]
        vectors = [[vectors[i][j+1]/z_norm[j] for j in range(len(z_norm))]
                        for i in range(len(vectors))]
    elif norm == 'L2-norm':
        l2_norm = [np.linalg.norm([vectors[j][i] for j in range(len(vectors))])
                        for i in range(1,len(vectors[0]))]
        vectors = [[vectors[i][j+1]/l2_norm[j] for j in range(len(l2_norm))]
                        for i in range(len(vectors))]
    return vectors

def evaluate(dist_matrix, train_split=None, test_split=None, cluster_method='kmeans', cluster_cnt=5):
    cluster_centers = {}
    for img in range(59):
        class_vectors = [vec[1:] for vec in train_split if vec[0] == img]
        if cluster_method == 'gmm':
            cluster_centers[img] = GaussianMixture(n_components=cluster_cnt).fit(class_vectors).means_
        elif cluster_method == 'kmeans':
            cluster_centers[img] = KMeans(n_clusters=cluster_cnt).fit(class_vectors).cluster_centers_

    for test_vec in test_split:
        for img, centers in cluster_centers.items():
            for c in centers:
                distance = np.matmul(np.matmul(np.transpose(test_vec[1:]-c), inv(dist_matrix)), test_vec[1:] - c)
                exemplar_sim.append([img, distance])

            exemplar_sim = sorted(exemplar_sim, key = lambda x: x[1])
            predicted_label.append(exemplar_sim[0])
            true_label.append(img)

            first_correct_ind = next(ind for ind, sim in enumerate(exemplar_sim) if sim[0]==test_vec[0])
            mrr.append(1/(first_correct_ind+1))

    mrr = np.mean(mrr)
    precision, recall, f_score = score(true_label, predicted_label, average='macro')
    return [precision, recall, f_score, mrr]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--normalize', choices = ['z-norm', 'L2-norm'])
    parser.add_argument('--cluster_method', choices = ['kmeans','gmm'])
    parser.add_argument('--cluster_cnt', type=int)
    parser.add_argument('--fold_cnt', type=int)

    opt = parser.parse_args()
    main(parser.parse_args())
