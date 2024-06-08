import json
import os

import cv2
from tqdm import tqdm

from const import data_dir, img_dir
from imgs import read_rgb_image, write_rgb_image
from progress_manager import ProgressManager
from prompt_constructor import age_mapping
from utils import prepare_number, square_bbox, pad_bbox, is_valid_bbox


def mkdirifnx(path):
    if not os.path.exists(path):
        os.makedirs(path)


faces_dir = os.path.join(data_dir, 'faces')
mkdirifnx(faces_dir)
faces_img_dir = os.path.join(faces_dir, 'imgs')
mkdirifnx(faces_img_dir)

pm = ProgressManager(os.path.join(data_dir, 'progress.json'), img_dir)


def main():
    n = 0
    face_imgs_data = []
    for key, data in tqdm(pm.progress.items()):
        if 'face_det' not in data:
            continue
        img_path = os.path.join(img_dir, f"{key}.png")
        img = read_rgb_image(img_path)
        img_shape = img.shape[:2]
        padded_bbox = pad_bbox(pm.get_face_bbox(key), img_shape, pad=0.2)
        bbox = square_bbox(padded_bbox, img_shape)
        if not is_valid_bbox(bbox, (img.shape[0], img.shape[1])):
            print(f"Invalid bbox for {key}, {padded_bbox}")
            continue
        face = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        if face.shape[0] <= 16 or face.shape[1] <= 16:
            print(f"Face too small for {key}")
            continue
        face = cv2.resize(face, (224, 224), interpolation=cv2.INTER_LANCZOS4)
        new_key = prepare_number(n)
        face_path = os.path.join(faces_img_dir, f"{new_key}.png")
        metadata = pm.get_metadata(key)
        face_imgs_data.append(
            {
                "source_image": f"{key}.png",
                "face_image": f"{new_key}.png",
                "age": age_mapping[metadata[0]],
                "ethnicity": metadata[1],
                'sex': metadata[2]
            }
        )
        write_rgb_image(face_path, face)
        n += 1
    json.dump(face_imgs_data, open(os.path.join(faces_dir, 'labels.json'), 'w'))


if __name__ == '__main__':
    main()
