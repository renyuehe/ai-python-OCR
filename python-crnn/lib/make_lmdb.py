# -*- coding: UTF-8 -*-
import os
import lmdb
import cv2
import numpy as np


# 检查图片是否有效
def check_image_valid(image_bin):
    if image_bin is None:
        return False

    image_buf = np.frombuffer(image_bin, dtype=np.uint8)
    img = cv2.imdecode(image_buf, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return False

    img_h, img_w = img.shape[0], img.shape[1]

    if img_h * img_w == 0:
        return False

    return True


# 写入lmdb数据库
def write_cache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


# 创建数据集
def create_dataset(output_path, images_path, dataset_file_path, check_valid=True):
    with open(dataset_file_path, 'r', encoding='utf-8') as f:
        dataset = f.readlines()
    dataset = list(map(lambda x: x.strip(), dataset))

    n_samples = len(dataset)
    print(n_samples)

    env = lmdb.open(output_path, map_size=21474836480)  # map_size=20G
    cache = {}
    cnt = 1
    bad_cnt = 0

    for sample in dataset:
        image_name, label = sample.split('\t')
        with open(os.path.join(images_path, image_name), 'rb') as f:
            image_bin = f.read()

        if check_valid:
            if not check_image_valid(image_bin):
                print('%s is not a valid image.' % image_name)
                bad_cnt += 1
                continue

        image_key = b'image-%09d' % cnt
        label_key = b'label-%09d' % cnt
        cache[image_key] = image_bin
        cache[label_key] = label.encode()

        if cnt % 10000 == 0:
            write_cache(env, cache)
            cache = {}
            print('Written %d / %d.' % (cnt, n_samples))

        cnt += 1

    n_samples = cnt - 1
    cache[b'num-samples'] = str(n_samples).encode()
    write_cache(env, cache)
    print('Created dataset with %d samples.' % n_samples)


if __name__ == '__main__':
    output_path = '/mnt/data1/data/guyu.gy/SyntheticChineseStringDataset/train'
    images_path = '/mnt/data1/data/guyu.gy/SyntheticChineseStringDataset/images'
    dataset_file_path = '/mnt/data1/data/guyu.gy/SyntheticChineseStringDataset/trainset.txt'
    create_dataset(output_path, images_path, dataset_file_path)
