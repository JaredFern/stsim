import numpy as np
import argparse
from itertools import combinations

def main(vectors, class_groups, cv_foldcnt=5):
    variance = [np.var([vectors[j][i] for j in range(len(vectors))])
                    for i in range(len(vectors[0]))]
    std_dev = [np.std([vectors[j][i] for j in range(len(vectors))])
                    for i in range(len(vectors[0]))]

    var_matrix = np.diag(variance)
    std_matrix = np.diag(std_dev)
    cov_matrix = np.cov(vectors,rowvar=False)
    np.save(open('var-matrix.bin','wb'), var_matrix)
    np.save(open('std-matrix.bin','wb'), std_matrix)
    np.save(open('cov-matrix.bin','wb'), cov_matrix)
    print ("Variance Matrix:", evaluate(var_matrix, class_groups))
    print ("StdDev Matrix:", evaluate(std_matrix, class_groups))
    print ("Covariance Matrix:", evaluate(cov_matrix, class_groups))

def evaluate(dst_matrix, class_groups):
    mdistance = []
    for label, vectors in class_groups.items():
        mdistance.extend([np.matmul(np.matmul(x-y, dst_matrix), np.transpose(x-y))
                            for x,y in combinations(vectors, 2)])
    return np.mean(mdistance), np.var(mdistance)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--normalize', choices = ['z-norm', 'L2-norm'])
    opt = parser.parse_args()

    vectors = np.load('curet-vectors.bin')
    class_groups = {ind: [vec[1:] for vec in vectors if vec[0] == ind] for ind in range(59)}
    if opt.normalize == 'z-norm':
        z_norm = [np.std([vectors[j][i] for j in range(len(vectors))])
                        for i in range(1,len(vectors[0]))]
        vectors = [[vectors[i][j+1]/z_norm[j] for j in range(len(z_norm))]
                        for i in range(len(vectors))]
    elif opt.normalize == 'L2-norm':
        l2_norm = [np.linalg.norm([vectors[j][i] for j in range(len(vectors))])
                        for i in range(1,len(vectors[0]))]
        vectors = [[vectors[i][j+1]/l2_norm[j] for j in range(len(l2_norm))]
                        for i in range(len(vectors))]
    else: vectors = [vec[1:] for vec in vectors]
    main(vectors, class_groups)
