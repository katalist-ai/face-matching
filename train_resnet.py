import os
from itertools import chain

import cv2
import torch
import torch.onnx
import torchmetrics
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from torchvision.models.resnet import ResNet34_Weights
from torchvision.models.resnet import resnet34
from tqdm import tqdm

from const import faces_dir, checkpoints_dir
from dataset import FacesDataset



class AgeSexClassify(nn.Module):
    def __init__(self):
        super(AgeSexClassify, self).__init__()
        self.base_model = resnet34(weights=ResNet34_Weights.DEFAULT)

        # freeze the base model weights
        for param in self.base_model.parameters():
            param.requires_grad = False

        # unfreeze the last layer (layer4)
        for param in self.base_model.layer4.parameters():
            param.requires_grad = True

        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()

        # Add new layers for age and sex classification
        self.age = nn.Linear(num_ftrs, 2)
        self.sex = nn.Linear(num_ftrs, 2)

    def forward(self, x):
        x = self.base_model(x)
        x_age = self.age(x)
        x_sex = self.sex(x)
        return x_age, x_sex

    def get_trainable_parameters(self):
        return chain(self.base_model.layer4.parameters(), self.age.parameters(), self.sex.parameters())


def train(model, train_loader, optimizer, criterion, epoch):
    model.train()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    losses = [0, 0, 0]
    n = 1
    for batch_idx, (data, targets) in pbar:
        age, sex = targets
        optimizer.zero_grad()
        age_h, sex_h = model(data)
        loss1 = criterion(age_h, age)
        losses[0] += loss1.item()
        loss2 = criterion(sex_h, sex)
        losses[1] += loss2.item()
        loss = (loss1 + loss2) / 2
        losses[2] += loss.item()
        loss.backward()
        optimizer.step()
        pbar.set_description(
            f'Epoch {epoch} - Loss: {losses[2] / n:.3f} - Loss1: {losses[0] / n:.3f} - Loss2: {losses[1] / n:.3f}')
        n += 1


def validate(model, val_loader, criterion):
    model.eval()
    loss_all = 0
    loss1_all = 0
    loss2_all = 0
    with torch.no_grad():
        for data, targets in val_loader:
            age, sex = targets
            age_h, sex_h = model(data)
            loss1 = criterion(age_h, age)
            loss2 = criterion(sex_h, sex)
            loss_all += (loss1 + loss2) / 2
            loss1_all += loss1
            loss2_all += loss2
    l = len(val_loader)
    return loss_all / l, loss1_all / l, loss2_all / l


def test(model, test_loader, device):
    model.eval()
    f1_age_metric = torchmetrics.F1Score(task='binary').to(device)
    f1_sex_metric = torchmetrics.F1Score(task='binary').to(device)
    with torch.no_grad():
        for data, targets in test_loader:
            age, sex = targets
            age_h, sex_h = model(data)
            age_h, sex_h = torch.softmax(age_h, 1), torch.softmax(sex_h, 1)
            f1_age_metric(age_h, age)
            f1_sex_metric(sex_h, sex)
    f1_age = f1_age_metric.compute()
    f1_sex = f1_sex_metric.compute()
    return f1_age, f1_sex


def convert_model_to_onnx(model, model_path, transform):
    img = cv2.imread('data/faces/imgs/00000000.png')[..., [2, 1, 0]]
    if transform:
        img = transform(img)
    img = img.unsqueeze(0)
    model.cpu()
    model.eval()
    with torch.no_grad():
        torch.onnx.export(model,  # model being run
                          img,  # model input (or a tuple for multiple inputs)
                          model_path,  # where to save the model (can be a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=['face_image'],  # the model's input names
                          output_names=['age', 'sex'],  # the model's output names
                          dynamic_axes={'face_image': {0: 'batch_size'},  # variable length axes
                                        'age': {0: 'batch_size'},
                                        'sex': {0: 'batch_size'}},
                          verbose=False
                          )


def main():
    mps = torch.backends.mps.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if mps else "cpu")
    print("DEVICE:", device)
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((160, 160)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15),
        transforms.RandomGrayscale(p=0.2),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((160, 160)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_dir = os.path.join(faces_dir, 'imgs')
    json_path = os.path.join(faces_dir, 'labels.json')
    full_dataset = FacesDataset(json_path, img_dir, transform=train_transform, device=device)
    train_size = int(0.75 * len(full_dataset))
    val_size = int(0.1 * train_size)
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    model = AgeSexClassify()
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
        print("VAL LOSS", l, l1, l2, "\nF1 AGE", f1_age, "F1 SEX", f1_sex)
        # scheduler.step()

    # test the model on test_loader set
    with torch.no_grad():
        l, l1, l2 = validate(model, test_loader, criterion)
        f1_age, f1_sex = test(model, test_loader, device)
        print(f"TESTING: LOSS AVG: {l}, LOSSage {l1}, LOSSSex: {l2}\n F1 age: {f1_age} F1 sex: {f1_sex}")
    convert_model_to_onnx(model, os.path.join(checkpoints_dir, 'model_resnet.onnx'), transform=train_transform)
    print("SAVING MODEL ONNX")
    torch.save(model.state_dict(), 'checkpoints/age_sex_34_full.pth')
    print("SAVING MODEL STATE DICT")


if __name__ == '__main__':
    main()
