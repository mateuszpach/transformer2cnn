import os
import numpy as np
from PIL import Image

path = './caltech_birds2011/CUB_200_2011/'
path_seg = './caltech_birds2011/'

os.mkdir(os.path.join(path, 'cutouts'))
for i, ((root_data, dirs_data, files_data), (root_seg, dirs_seg, files_seg)) in enumerate(
        zip(sorted(os.walk(os.path.join(path, 'images'))), sorted(os.walk(os.path.join(path_seg, 'segmentations'))))):
    print(root_data, root_seg)
    if i <= 0:  # skip 'images/'
        continue
    print(root_data[root_data.find('\\') + 1:])
    new_dir = os.path.join(path, 'cutouts', root_data[root_data.rfind('/') + 1:])
    os.mkdir(new_dir)
    for img, seg in zip(files_data, files_seg):
        img_data = Image.open(os.path.join(root_data, img))
        img_data = np.asarray(img_data)
        seg_data = Image.open(os.path.join(root_seg, seg))
        seg_data = np.asarray(seg_data) / 255
        if len(seg_data.shape) > 2:
            seg_data = seg_data[:, :, 0]
            # a few masks have manny layers...
        seg_data = np.reshape(seg_data, (seg_data.shape[0], seg_data.shape[1], 1))
        seg_data = np.repeat(seg_data, 3, axis=-1)
        if len(img_data.shape) == 2:  # grayscale
            img_data = np.reshape(img_data, (img_data.shape[0], img_data.shape[1], 1))
            img_data = np.repeat(img_data, 3, axis=-1)

        img_data = img_data * seg_data
        img_data = np.uint8(img_data)
        im = Image.fromarray(img_data)
        im.save(os.path.join(new_dir, img))
