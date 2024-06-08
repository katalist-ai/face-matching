import os

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from tqdm import tqdm

from progress_manager import ProgressManager
from utils import count_images_in_dir, prepare_number

NWORKERS = 4

img_dir = os.path.join("data", "images")

app = FaceAnalysis(allowed_modules=["detection"])
app.prepare(ctx_id=0, det_size=(640, 640))


def detect_face(img_num):
    image_path = os.path.join(img_dir, prepare_number(img_num) + ".png")
    img = cv2.imread(image_path)[..., [2, 1, 0]]
    faces = app.get(img, max_num=1)
    if len(faces) == 0:
        print(f"No face detected in image {img_num}")
        return None
    return faces[0]


if __name__ == "__main__":
    n_images = count_images_in_dir(img_dir)
    pm = ProgressManager(os.path.join("data", "progress.json"), img_dir, write_on_change=False)
    for i in tqdm(range(n_images)):
        has_face_info = pm.has_face_info(i)
        if has_face_info:
            continue
        face_info = detect_face(i)
        if face_info is None:
            continue
        face_info.bbox = face_info.bbox.astype(np.int32).tolist()
        face_info.kps = face_info.kps.astype(np.int32).tolist()
        face_info.det_score = float(face_info.det_score)
        pm.set_face_info(i, face_info)
    pm.save_changes()
