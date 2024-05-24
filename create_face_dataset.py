import json
import os
from progress_manager import ProgressManager
from const import data_dir, img_dir
from imgs import read_rgb_image, write_rgb_image
from utils import prepare_number, square_bbox
from prompt_constructor import age_mapping
import cv2
from tqdm import tqdm

faces_dir = os.path.join(data_dir, 'faces')
if not os.path.exists(faces_dir):
    os.makedirs(faces_dir)

pm = ProgressManager(os.path.join(data_dir, 'progress.json'), img_dir)

def main():
    n = 0
    face_imgs_data = []
    for key, data in tqdm(pm.progress.items()):
        if 'face_det' not in data:
            continue
        bbox = square_bbox(pm.get_face_bbox(key))
        img_path = os.path.join(img_dir, f"{key}.png")
        img = read_rgb_image(img_path)
        face = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        if face.shape[0] <= 16 or face.shape[1] <= 16:
            continue
        face = cv2.resize(face, (224, 224), interpolation=cv2.INTER_LANCZOS4)
        new_key = prepare_number(n)
        face_path = os.path.join(faces_dir, 'imgs', f"{new_key}.png")
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
