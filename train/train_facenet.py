import os

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from train_resnet import convert_model_to_onnx, test, train, validate

from dataset import FacesDataset
from models.inception_resnet_v1 import FaceNetCustom
from utils.const import checkpoints_dir, faces_dir


def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor


def prewhiten(x):
    mean = x.mean()
    std = x.std()
    std_adj = std.clamp(min=1.0 / (float(x.numel()) ** 0.5))
    y = (x - mean) / std_adj
    return y


def main():
    mps = torch.backends.mps.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if mps else "cpu")
    print("DEVICE:", device)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((160, 160)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15),
            transforms.RandomGrayscale(p=0.2),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img_dir = os.path.join(faces_dir, "imgs")
    json_path = os.path.join(faces_dir, "labels.json")
    full_dataset = FacesDataset(json_path, img_dir, transform=transform, device=device)
    train_size = int(0.75 * len(full_dataset))
    val_size = int(0.1 * train_size)
    test_size = len(full_dataset) - train_size - val_size
    print("DATASET LEN", len(full_dataset))
    print("TRAIN SIZE", train_size)
    print("VAL SIZE", val_size)
    print("TEST SIZE", test_size)
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    model = FaceNetCustom()
    model.to(device)
    # optimizer = optim.SGD(model.get_trainable_parameters(), lr=0.01, momentum=0.6)
    optimizer = optim.Adam(model.get_trainable_parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    # scheduler = StepLR(optimizer, step_size=1, gamma=0.8)

    num_epochs = 6
    for epoch in range(1, num_epochs + 1):
        train(model, train_loader, optimizer, criterion, epoch)
        l, l1, l2 = validate(model, val_loader, criterion)
        f1_age, f1_sex = test(model, val_loader, device)
        print("VAL LOSS", l, l1, l2, "F1 AGE", f1_age, "F1 SEX", f1_sex)
        # scheduler.step()

    # test the model on test_loader set
    with torch.no_grad():
        l, l1, l2 = validate(model, test_loader, criterion)
        f1_age, f1_sex = test(model, val_loader, device)
        print(f"TESTING: LOSS AVG: {l}, LOSSage {l1}, LOSSSex: {l2}\n F1 age: {f1_age} F1 sex: {f1_sex}")
    convert_model_to_onnx(model, os.path.join(checkpoints_dir, "model_inception_resnet.onnx"), transform=transform)
    print("SAVING MODEL ONNX")
    torch.save(model.state_dict(), os.path.join(checkpoints_dir, "age_sex_inception_resnet_full.pth"))
    print("SAVING MODEL STATE DICT")


if __name__ == "__main__":
    main()
