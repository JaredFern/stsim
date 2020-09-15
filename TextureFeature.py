import numpy as np

import cv2
import os

from abc import ABC, abstractmethod
from collections import Counter
from numpy.linalg import inv
from scipy.stats import entropy

from scipy.spatial.distance import mahalanobis
from skimage import color
from skimage.feature import local_binary_pattern

from sklearn.model_selection import StratifiedKFold
from utils import filterbank, metric


def _filter_class_counts(image_dir, class_cnt, fold_cnt, scale_invariance=False):
    fold_cnt = min(fold_cnt, class_cnt)
    img_class_files = {}
    for root, dir, files in os.walk(image_dir):
        for fpath in files:
            img_class, scale = fpath.split("-")
            if scale_invariance:
                scale = scale.split("_", 3)[1]
                img_class = ".".join([img_class, scale])
            if img_class not in img_class_files:
                img_class_files[img_class] = [os.path.join(root, fpath)]
            elif len(img_class_files[img_class]) < class_cnt or class_cnt == -1:
                img_class_files[img_class].append(os.path.join(root, fpath))
    img_classes = set(img_class_files.keys())
    for img_class in img_classes:
        if (len(img_class_files[img_class]) < fold_cnt) or (class_cnt > 0 and img_class in
                                                            img_class_files and len(img_class_files[img_class]) < class_cnt):
            del img_class_files[img_class]
    print(
        f"Filtered data to {len(img_class_files.keys())} classes of size {class_cnt}")
    return img_class_files


class TextureFeature(ABC):
    def __init__(self, image_dir_or_saved_bin, class_cnt, save_features, scale_invariance=False,
                 fold_cnt=5):
        self.labels, self.features = self.load_or_generate_features(
            image_dir_or_saved_bin, class_cnt, fold_cnt, scale_invariance, save_features)
        self.validator = StratifiedKFold(n_splits=fold_cnt, shuffle=True).split(
            self.features, self.labels)

    def load_or_generate_features(self, image_dir_or_saved_features, class_cnt, fold_cnt,
                                  scale_invariance, save_features):
        if os.path.isfile(image_dir_or_saved_features):
            return np.load(image_dir_or_saved_features)

        img_class_files = _filter_class_counts(
            image_dir_or_saved_features, class_cnt, fold_cnt, scale_invariance)
        img_labels, img_features = [], []
        for img_class, files in img_class_files.items():
            for fpath in files:
                single_feature = self.generate_feature(fpath)
                img_labels.append(img_class)
                img_features.append(single_feature)
        if save_features:
            np.savez(open(save_features, 'wb'), img_labels, img_features)
        return img_labels, img_features

    @abstractmethod
    def generate_feature(self, fpath):
        pass

    @abstractmethod
    def compute_similarity(self, feature_1, feature_2):
        pass


class Ssim(TextureFeature):
    def __init__(self, image_dir_or_saved_bin, class_cnt, save_features, scale_invariance,
                 fold_cnt=5, k_range=(0.01, 0.03), L=255):
        self.k_range = k_range
        self.luminance = L
        self.c1 = (k_range[0] * L) ** 2
        self.c2 = (k_range[1] * L) ** 2
        tmp_window = metric.fspecial()
        self.window = tmp_window/tmp_window.sum()
        super().__init__(image_dir_or_saved_bin, class_cnt, save_features, scale_invariance,
                         fold_cnt)

    def generate_feature(self, fpath):
        source_image = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        mu = metric.conv(source_image, self.window)
        sigma_squared = metric.conv(
            source_image * source_image, self.window) - mu ** 2
        return mu, sigma_squared, fpath

    def compute_similarity(self, feature_1, feature_2):
        img1 = cv2.imread(feature_1[2], cv2.IMREAD_GRAYSCALE).astype(float)
        img2 = cv2.imread(feature_2[2], cv2.IMREAD_GRAYSCALE).astype(float)
        sigma_12 = metric.conv(img1 * img2, self.window) - \
            feature_1[0] * feature_2[0]
        ssim_map = ((2*feature_1[0] * feature_2[0] + self.c1) * (2*sigma_12 + self.c2)) / (
            (feature_1[0] ** 2 + feature_2[0] ** 2 + self.c1) *
            (feature_1[1] + feature_2[1] + self.c2))
        return ssim_map.mean()


class Stsim1(TextureFeature):
    def generate_feature(self, fpath):
        steerable_filter = filterbank.Steerable()
        source_image = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        steerable_pyramid = steerable_filter.getlist(
            steerable_filter.buildSCFpyr(source_image))
        return steerable_pyramid

    def compute_similarity(self, feature_1, feature_2):
        img1 = cv2.imread(feature_1[2], cv2.IMREAD_GRAYSCALE).astype(float)
        img2 = cv2.imread(feature_2[2], cv2.IMREAD_GRAYSCALE).astype(float)
        sigma_12 = metric.conv(img1 * img2, self.window) - \
            feature_1[0] * feature_2[0]
        ssim_map = ((2*feature_1[0] * feature_2[0] + self.c1) * (2*sigma_12 + self.c2)) / (
            (feature_1[0] ** 2 + feature_2[0] ** 2 + self.c1) *
            (feature_1[1] + feature_2[1] + self.c2))
        return ssim_map.mean()


class Stsim1(TextureFeature):
    def generate_feature(self, fpath):
        steerable_filter = filterbank.Steerable()
        source_image = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        steerable_pyramid = steerable_filter.getlist(
            steerable_filter.buildSCFpyr(source_image))
        return steerable_pyramid

    def compute_similarity(self, feature_1, feature_2):
        np.mean([metric.pooling(feature_1[i], feature_2[i])
                 for i in range(len(feature_1))])


class Stsim2(TextureFeature):
    def generate_feature(self, fpath):
        source_image = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)

        steerable_filter = filterbank.Steerable()
        steerable_pyramid = steerable_filter.getlist(
            steerable_filter.buildSCFpyr(source_image))

        steerable_filter_nosubbands = filterbank.SteerableNoSub()
        steerable_crossterms = steerable_filter_nosubbands.buildSCFpyr(
            source_image)
        return steerable_pyramid, steerable_crossterms

    def compute_similarity(self, feature_1, feature_2):
        # Base STSIM-1 pooled metrics
        stsim2 = [metric.pooling(feature_1[0][i], feature_2[0][i])
                  for i in range(len(feature_1[0]))]

        # Across scale, same orientation
        for scale in range(2, len(feature_1[1]) - 1):
            for orient in range(len(feature_1[1][1])):
                im11 = np.abs(feature_1[1][scale - 1][orient])
                im12 = np.abs(feature_1[1][scale][orient])

                im21 = np.abs(feature_2[1][scale - 1][orient])
                im22 = np.abs(feature_2[1][scale][orient])
                stsim2.append(metric.compute_cross_term(
                    im11, im12, im21, im22, 7).mean())

        # Across orientation, same scale
        for scale in range(2, len(feature_1[1]) - 1):
            for orient in range(len(feature_1[1][1])):
                im11 = np.abs(feature_1[1][scale][orient])
                im21 = np.abs(feature_2[1][scale][orient])

                for orient2 in range(orient + 1, len(feature_1[1][1])):
                    im13 = np.abs(feature_1[1][scale][orient2])
                    im23 = np.abs(feature_2[1][scale][orient2])
                    stsim2.append(metric.compute_cross_term(
                        im11, im13, im21, im23, 7).mean())
        return np.mean(stsim2)


class StsimC(TextureFeature):
    def __init__(self, image_dir_or_saved_bin, class_cnt, save_features, scale_invariance,
                 fold_cnt=5, mahalanobis_file=None, mahalanobis_type="cov", scope="intraclass",
                 aca_color_cnt=0, color_dir=""):
        self.mahalanobis_file = mahalanobis_file
        self.mahalanobis_type = mahalanobis_type
        self.scope = scope
        self.aca_color_cnt = aca_color_cnt
        self.color_dir = color_dir
        self.mahalanobis_matrix = None
        super().__init__(image_dir_or_saved_bin, class_cnt, save_features, scale_invariance,
                         fold_cnt)

    def compute_similarity(self, feature_1, feature_2):
        return mahalanobis(feature_1, feature_2, inv(self.mahalanobis_matrix))

    def generate_feature(self, fpath):
        vec = list(metric.Metric().STSIM_M(
            cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)))
        if self.aca_color_cnt:
            pixel_cnt = Counter()
            base_name = os.path.splitext(os.path.basename(fpath))[0]
            color_path = os.path.join(self.color_dir, base_name + '_lav.png')
            img_lab = color.rgb2lab(cv2.imread(color_path))
            for i in range(len(img_lab)):
                for j in range(len(img_lab[i])):
                    pixel_cnt[tuple(img_lab[i][j])] += 1

            color_features = [i[0]
                              for i in pixel_cnt.most_common(self.aca_color_cnt)]
            if len(color_features) < self.aca_color_cnt:
                color_features.extend(
                    [(0, 0, 0) for i in range(self.aca_color_cnt - len(color_features))])
            color_features = [
                val for color_idx in color_features for val in color_idx]
            vec += color_features
        return vec

    def compute_mahalanobis_matrix(self, train_split):
        if self.mahalanobis_file and os.path.exists(self.mahalanobis_file):
            self.mahalanobis_matrix = np.load(self.mahalanobis_file)
            return
        class_matrix, train_features = [], []
        class_means = {}
        for img_class in set(self.labels):
            img_features = []
            for train_idx in train_split:
                if self.labels[train_idx] == img_class:
                    img_features.append(self.features[train_idx])
            # Generate array of [var, cov, std] matrices from each class
            if self.scope == 'intraclass':
                if self.mahalanobis_type == 'var':
                    class_matrix.append(np.var(img_features, axis=0))
                elif self.mahalanobis_type == 'std':
                    class_matrix.append(np.std(img_features, axis=0))
                elif self.mahalanobis_type == 'cov':
                    class_matrix.append(np.cov(img_features, rowvar=False))
            class_means[img_class] = np.mean(img_features, axis=0)
            train_features.extend(img_features)

        if self.scope == 'stsim_i':
            train_features = []
            for train_idx in train_split:
                normalized_feature = self.features[train_idx] - \
                    class_means[self.labels[train_idx]]
                train_features.append(normalized_feature)

        # Flatten training array for computing global M matrix for
        if self.scope in ('global', 'stsim_i'):
            if self.mahalanobis_type == 'std':
                self.mahalanobis_matrix = np.diag(
                    np.std(train_features, axis=0))
            elif self.mahalanobis_type == 'var':
                self.mahalanobis_matrix = np.diag(
                    np.var(train_features, axis=0))
            elif self.mahalanobis_type == 'cov':
                self.mahalanobis_matrix = np.cov(train_features, rowvar=False)
        # Compute single M matrix from individual class M matrices
        elif self.scope in ['intraclass', 'cluster']:
            if self.mahalanobis_type in ['std', 'var']:
                self.mahalanobis_matrix = np.diag(
                    np.mean(class_matrix, axis=0))
            elif self.mahalanobis_type == 'cov':
                self.mahalanobis_matrix = np.mean(class_matrix, axis=0)
        if self.mahalanobis_file:
            np.save(self.mahalanobis_file, self.mahalanobis_matrix)


class LocalBinaryPattern(TextureFeature):
    def __init__(self, image_dir_or_saved_bin, class_cnt, save_features, scale_invariance,
                 fold_cnt, n_points, radius, lbp_method):
        self.n_points = n_points
        self.radius = radius
        self.lbp_method = lbp_method
        super().__init__(image_dir_or_saved_bin, class_cnt, save_features, fold_cnt)

    def compute_similarity(self, feature_1, feature_2):
        return entropy(feature_1, feature_2)

    def generate_feature(self, image_path):
        n_bins = 0
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        lbp = local_binary_pattern(
            img, self.n_points, self.radius, method=self.lbp_method)
        n_bins = max(int(lbp.max() + 1), n_bins)
        lbp_hist = np.histogram(
            lbp, density=True, bins=n_bins, range=(0, n_bins))[0]
        return lbp_hist
