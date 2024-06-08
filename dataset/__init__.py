import json
import os

# noinspection PyPackageRequirements
import cv2
import torch
from torch.utils.data import Dataset


def read_rgb_image(img_path):
    return cv2.imread(img_path)[..., ::-1]


sex_mapping = {"female": [1.0, 0.0], "male": [0.0, 1.0]}
age_mapping = {"adult": [1.0, 0.0], "child": [0.0, 1.0]}


class FacesDataset(Dataset):
    def __init__(self, labels_path, img_dir, transform=None, device="cpu"):
        self.labels = json.load(open(labels_path))
        self.img_dir = img_dir
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_data = self.labels[idx]
        img_path = str(os.path.join(self.img_dir, img_data["face_image"]))
        image = cv2.imread(str(img_path))[..., [2, 1, 0]]
        age = img_data["age"]
        sex = img_data["sex"]
        if self.transform:
            image = self.transform(image)
        return (
            image.to(self.device),
            (torch.Tensor(age_mapping[age]).to(self.device), torch.Tensor(sex_mapping[sex]).to(self.device)),
        )
