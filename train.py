import os

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import random_split, DataLoader
from torchvision.models.resnet import resnet18, resnet34
from torchvision import transforms
from tqdm import tqdm
from itertools import chain

from dataset import FacesDataset
from const import faces_dir
import wandb
#
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="resnet18-finetuning",
#
#     # track hyperparameters and run metadata
#     config={
#         "learning_rate": 0.001,
#         "architecture": "ResNet18 + 2 linear classifiers",
#         "dataset": "FacesDatasetv1",
#         "epochs": 10,
#     }
# )

def create_model():
    base_model = resnet34(pretrained=True)
    num_ftrs = base_model.fc.in_features
    base_model.fc = nn.Identity()
    for param in base_model.parameters():
        param.requires_grad = False
    # Two new classifiers
    classifier_task1 = nn.Sequential(nn.Linear(num_ftrs, num_ftrs // 2),
                                     nn.ReLU(),
                                     nn.Linear(num_ftrs // 2, 2))
    classifier_task2 = nn.Sequential(nn.Linear(num_ftrs, num_ftrs // 2),
                                     nn.ReLU(),
                                     nn.Linear(num_ftrs // 2, 2))

    return base_model, classifier_task1, classifier_task2


def train(model, classifier1, classifier2, device, train_loader, optimizer, criterion, epoch):
    model.train()
    classifier1.train()
    classifier2.train()
    for batch_idx, (data, targets) in tqdm(enumerate(train_loader), total=len(train_loader)):
        data = data.to(device)
        age, sex = targets
        age = age.to(device)
        sex = sex.to(device)
        optimizer.zero_grad()
        features = model(data)
        output_task1 = classifier1(features)
        output_task2 = classifier2(features)
        loss1 = criterion(output_task1, age)
        loss2 = criterion(output_task2, sex)
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()

        # if batch_idx % 10 == 0:
        #     torch.save(model.state_dict(), f'checkpoint_epoch_{epoch}_batch_{batch_idx}.pth')


def validate(model, classifier1, classifier2, device, val_loader, criterion):
    model.eval()
    classifier1.eval()
    classifier2.eval()
    loss_all = 0
    loss1_all = 0
    loss2_all = 0
    with torch.no_grad():
        for data, targets in val_loader:
            data = data.to(device)
            age, sex = targets
            age = age.to(device)
            sex = sex.to(device)
            features = model(data)
            output_task1 = classifier1(features)
            output_task2 = classifier2(features)
            loss1 = criterion(output_task1, age)
            loss2 = criterion(output_task2, sex)
            loss_all += loss1 + loss2
            loss1_all += loss1
            loss2_all += loss2
    l = len(val_loader)
    return loss_all / l, loss1_all / l, loss2_all / l



def main():
    mps = torch.backends.mps.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if mps else "cpu")
    print("DEVICE:", device)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_dir = os.path.join(faces_dir, 'imgs')
    json_path = os.path.join(faces_dir, 'labels.json')
    full_dataset = FacesDataset(json_path, img_dir, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    model, c1, c2 = create_model()
    model.to(device)
    c1.to(device)
    c2.to(device)
    # optimizer = optim.Adam(chain(c1.parameters(), c2.parameters()), lr=0.001)
    optimizer = optim.SGD(chain(c1.parameters(), c2.parameters()), lr=0.01, momentum=0.6)
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    num_epochs = 20
    for epoch in range(1, num_epochs + 1):
        train(model, c1, c2, device, train_loader, optimizer, criterion, epoch)
        l, l1, l2 = validate(model, c1, c2, device, val_loader, criterion)
        print("VAL LOSS", l, l1, l2)
        print(scheduler.step_size)
        scheduler.step()

    torch.save(model.state_dict(), 'checkpoints/final_model.pth')

if __name__ == '__main__':
    main()
