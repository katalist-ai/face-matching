import json
import os

from PIL import Image

from utils import prepare_number


class ProgressManager:
    def __init__(self, file_path: str, img_dir: str, write_on_change=True):
        self.file_name = file_path
        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                json.dump({}, f)
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        self.img_dir = img_dir
        self.progress = json.load(open(file_path, "r"))
        self.write_on_change = write_on_change

    def update_progress(self, key: str | int, payload: dict, metadata, img: Image.Image):
        prepared_key = prepare_number(key)
        self.progress[prepared_key] = {
            "payload": payload,
            "metadata": metadata,
        }
        if self.write_on_change:
            json.dump(self.progress, open(self.file_name, "w"), indent=1, sort_keys=True)
            img.save(os.path.join(self.img_dir, f"{prepared_key}.png"))

    def key_exists(self, key: str | int) -> bool:
        return prepare_number(key) in self.progress

    def has_face_info(self, key: str | int) -> bool:
        prepared_key = prepare_number(key)
        return "face_det" in self.progress[prepared_key]

    def set_face_info(self, key: str | int, face_info: dict):
        prepared_key = prepare_number(key)
        self.progress[prepared_key]["face_det"] = face_info
        if self.write_on_change:
            json.dump(self.progress, open(self.file_name, "w"), indent=1)

    def get_face_info(self, key: str | int):
        prepared_key = prepare_number(key)
        if 'face_det' not in self.progress[prepared_key]:
            return None
        return self.progress[prepared_key]["face_det"]

    def get_face_bbox(self, key: str | int):
        face_info = self.get_face_info(key)
        if face_info is None:
            return None
        return face_info["bbox"]

    def get_metadata(self, key: str | int):
        prepared_key = prepare_number(key)
        return self.progress[prepared_key]["metadata"]

    def save_changes(self):
        json.dump(self.progress, open(self.file_name, 'w'), indent=1)
