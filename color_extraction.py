from skimage import color
from scipy.ndimage import imread
from collections import Counter
from tqdm import tqdm

import numpy as np
import os, pickle

def pixel_cnt(img_path = "/home/jaredfern/Desktop/textures/curet-color", save=True):
    texture_class = 0
    files = sorted([os.path.join(img_path, f) for f in os.listdir(img_path)])
    class_colors = {i: [] for i in range(62)}
    for f in tqdm(range(len(files))): 
        pixel_cnt = Counter()
        img_lab = color.rgb2lab(imread(files[f]))
        texture_class = os.path.basename(files[f]).split("-")[0]
        
        for i in range(len(img_lab)):
            for j in range(len(img_lab[i])):
                pixel_cnt[tuple(img_lab[i][j])] += 1
        class_colors[int(texture_class)].append(pixel_cnt)

    if save: pickle.dump(class_colors, open("color_features.bin", "wb"))
    return class_colors

def extend_color_features(stsim_matrix, color_cnts, colors = 1, ordered = "L"):
    for stsim_class in stsim_matrix.keys():
        for img_ind in range(len(stsim_class)):
            pixel_cnt = color_cnts[stsim_class][img_ind]
            if ordered == "C": # Composition
                color_features = [i[0] for i in pixel_cnt.most_common(colors)]
            elif ordered == "L": # Luminance
                color_features = sorted(pixel_cnt.keys(), key=lambda x:x[0])

            flattened_colors = [val for color in color_features for val in color]
            stsim_class[img_ind].extend(flattened_colors)                

    return stsim_matrix