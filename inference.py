import numpy as np
import onnxruntime as ort

from utils.imgs import read_rgb_image

RESNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
RESNET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def preprocess_image(image: np.ndarray):
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0  # Scale to [0,1]

    if image.shape[1] != 3:
        image = image.transpose((2, 0, 1))  # CHW format

    image = (image - RESNET_MEAN) / RESNET_STD  # Apply normalization

    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


def softmax(x):
    # Subtract the max for numerical stability
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    # Divide by the sum of exps
    return e_x / np.sum(e_x, axis=1, keepdims=True)


label_map = {"sex": {0: "female", 1: "male"}, "age": {0: "adult", 1: "child"}}


class AgeSexInference:
    def __init__(self, onnx_model_path):
        self.model = ort.InferenceSession(onnx_model_path)

    def predict(self, image: np.ndarray, preprocess=True):
        if preprocess:
            image = preprocess_image(image)
        """RGB image in the format [B, C, H, W]"""
        age, sex = self.model.run(None, {"face_image": image})
        return age, sex

    def predict_probs(self, image: np.ndarray, *args, **kwargs):
        """RGB image in the format [B, C, H, W]"""
        age, sex = self.predict(image, *args, **kwargs)
        return softmax(age), softmax(sex)

    def predict_labels(self, image: np.ndarray, *args, **kwargs):
        """RGB image in the format [B, C, H, W]"""
        age, sex = self.predict(image, *args, **kwargs)
        age_idx = np.argmax(age, axis=1)
        sex_idx = np.argmax(sex, axis=1)
        return [label_map["age"][idx] for idx in age_idx], [label_map["sex"][idx] for idx in sex_idx]


class AgeSexInceptionResnet:
    def __init__(self, onnx_model_path):
        self.model = ort.InferenceSession(onnx_model_path)

    def predict(self, image: np.ndarray, preprocess=True):
        if preprocess:
            image = preprocess_image(image)
        """RGB image in the format [B, C, H, W]"""
        age, sex = self.model.run(None, {"face_image": image})
        return age, sex

    def predict_probs(self, image: np.ndarray, *args, **kwargs):
        """RGB image in the format [B, C, H, W]"""
        age, sex = self.predict(image, *args, **kwargs)
        return softmax(age), softmax(sex)

    def predict_labels(self, image: np.ndarray, *args, **kwargs):
        """RGB image in the format [B, C, H, W]"""
        age, sex = self.predict(image, *args, **kwargs)
        age_idx = np.argmax(age, axis=1)
        sex_idx = np.argmax(sex, axis=1)
        return [label_map["age"][idx] for idx in age_idx], [label_map["sex"][idx] for idx in sex_idx]


def main():
    # Load the exported ONNX model
    import os

    import torch
    from torchvision import transforms

    from dataset import FacesDataset
    from models.custom_resnet import AgeSexClassify
    from utils.const import faces_dir

    # check if the onnx model is working as well as the pytorch model
    image1 = preprocess_image(read_rgb_image("data/faces/imgs/00000000.png"))
    image2 = preprocess_image(read_rgb_image("data/faces/imgs/00000165.png"))
    print(image1[0].mean())
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img_dir = os.path.join(faces_dir, "imgs")
    json_path = os.path.join(faces_dir, "labels.json")
    full_dataset = FacesDataset(json_path, img_dir, transform=transform, device="cpu")
    img1 = full_dataset[0][0]
    print(img1[0].mean())
    img2 = full_dataset[165][0]
    images = np.array([image1, image2])
    images_torch = np.array([img1, img2])
    images_torch = torch.Tensor(images_torch)
    print(images.shape)
    model = AgeSexInference("checkpoints/model.onnx")
    model_torch = AgeSexClassify()
    model_torch.load_state_dict(torch.load("checkpoints/age_sex_34_full.pth"))
    model_torch.eval()
    with torch.no_grad():
        print(model.predict(images, preprocess=False))
        print(model_torch(torch.tensor(images)))
        print(model_torch(images_torch))
        print(model.predict_probs(images, preprocess=False))

    # check predict probs


if __name__ == "__main__":
    main()
