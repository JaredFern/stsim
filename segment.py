import cv2
import os


file_names = {}
for root, dirs, files in os.walk(os.getcwd()):
    for fname in files:
        name, seg = fname[:-5].split("_",1)
        if name not in file_names:
            file_names[name] = [seg]
        else:
            file_names[name].append(seg)

class_id = 0
for src_name, segments in file_names.items():
    src_seg = [s for s in segments if ('d' not in s) and ('ss' not in s)]
    downsampled_seg = [s for s in segments if 'd' in s]
    alt_seg = [s for s in segments if 'ss' in s]

    for seg in src_seg:
        class_cnt = 0
        fname = os.path.join(root, src_name + '_' + seg + '.tiff')
        img = cv2.imread(fname)
        for i in range(0, img.shape[0]-128, 32):
            for j in range(0, img.shape[1]-128, 32):
                sample_img = img[i:i+128, j:j+128]
                sample_name = f"{class_id}-scale_1_img_{class_cnt}.tiff"
                cv2.imwrite(os.path.join(root, sample_name), sample_img)
                class_cnt += 1

    for seg in downsampled_seg:
        class_cnt = 0
        fname = os.path.join(root, src_name + '_' + seg + '.tiff')
        img = cv2.imread(fname)
        for i in range(0, img.shape[0]-128, 32):
            for j in range(0, img.shape[1]-128, 32):
                sample_img = img[i:i+128, j:j+128]
                sample_name = f"{class_id}-scale_2_img_{class_cnt}.tiff"
                cv2.imwrite(os.path.join(root, sample_name), sample_img)
                class_cnt += 1

    class_id += 1

    for seg in alt_seg:
        class_cnt = 0
        fname = os.path.join(root, src_name + '_' + seg + '.tiff')
        img = cv2.imread(fname)
        for i in range(0, img.shape[0]-128, 32):
            for j in range(0, img.shape[1]-128, 32):
                sample_img = img[i:i+128, j:j+128]
                sample_name = f"{class_id}-scale_2_img_{class_cnt}.tiff"
                cv2.imwrite(os.path.join(root, sample_name), sample_img)
                class_cnt += 1

    if alt_seg: class_id += 1


