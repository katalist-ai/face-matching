import json
import os

import cv2
from insightface.app import FaceAnalysis

from utils import is_valid_bbox, pad_bbox, prepare_number, square_bbox
from utils.const import data_dir
from utils.imgs import read_rgb_image, write_rgb_image


def mkdirifnx(path):
    if not os.path.exists(path):
        os.makedirs(path)


faces_dir = os.path.join(data_dir, "test_faces")
mkdirifnx(faces_dir)
faces_img_dir = os.path.join(faces_dir, "imgs")
mkdirifnx(faces_img_dir)
source_imgs = os.path.join(data_dir, "test_images")

app = FaceAnalysis(allowed_modules=["detection"])
app.prepare(ctx_id=0, det_size=(640, 640))


def detect_face(img):
    faces = app.get(img, max_num=1)
    if len(faces) == 0:
        print(f"No face detected in image")
        return None
    return faces[0]


def main():
    n = 0
    face_imgs_data = []
    for img_path in sorted(os.listdir(source_imgs)):
        full_path = os.path.join(source_imgs, img_path)
        img = read_rgb_image(full_path)
        face_info = detect_face(img)
        if face_info is None:
            continue
        bbox = [int(k) for k in face_info.bbox]
        print(img.shape[:2])
        image_shape = img.shape[:2]
        padded_bbox = pad_bbox(bbox, image_shape, pad=0.2)
        bbox = square_bbox(padded_bbox, image_shape)
        if not is_valid_bbox(bbox, image_shape):
            print(f"Invalid bbox for {img_path}, {padded_bbox}")
            continue
        face = img[bbox[1] : bbox[3], bbox[0] : bbox[2]]
        if face.shape[0] <= 16 or face.shape[1] <= 16:
            print(f"Face too small for {img_path}")
            continue
        face = cv2.resize(face, (224, 224), interpolation=cv2.INTER_LANCZOS4)
        new_key = prepare_number(n)
        face_path = os.path.join(faces_img_dir, f"{new_key}.png")
        face_imgs_data.append(
            {
                "source_image": img_path,
                "face_image": f"{new_key}.png",
            }
        )
        write_rgb_image(face_path, face)
        n += 1
    json.dump(face_imgs_data, open(os.path.join(faces_dir, "test_labels.json"), "w"))


if __name__ == "__main__":
    main()
