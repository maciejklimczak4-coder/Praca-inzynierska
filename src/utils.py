import torch
import torchvision
from pathlib import Path
import pandas as pd
from PIL import Image
import copy


class ScrewDataset(torch.utils.data.Dataset):
    def __init__(self, manifest, images_dir, transform):
        self.manifest = manifest
        self.images_dir = images_dir
        self.transform = transform
        self.label_map = {'NOK': 1, 'OK': 0}


    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]

        img_path = Path(self.images_dir, row['new_filepath'])

        image = Image.open(img_path)
        image = image.convert('RGB')
        image = self.transform(image)

        label = self.label_map[row['category']]

        return image, label

def train_one_epoch(model, loader,criterion, optimizer, device):
    model.train()
    loss_sum = 0
    samples_count = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * images.shape[0]
        samples_count += images.shape[0]
        
        

    return loss_sum/samples_count

def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, samples_count = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logity = model(images)
            loss = criterion(logity, labels)
            preds = logity.argmax(dim=1)
            correct += (preds == labels).sum().item()
            loss_sum += loss.item() * images.shape[0]
            samples_count += images.shape[0]
        val_loss = loss_sum / samples_count
        val_accuracy = correct/ samples_count


    return val_loss, val_accuracy


def build_model_resnet18(num_classes):
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=num_classes, bias=True)

    return model

def build_model_mobilenetv3(num_classes):
    model = torchvision.models.mobilenet_v3_small(weights=torchvision.models.MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier[3] = torch.nn.Linear(in_features= model.classifier[3].in_features, out_features=num_classes, bias=True)

    return model

def train_model(NUM_EPOCHS, PATIENCE, model, train_loader, val_loader, criterion, optimizer, device):

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None
    history = []

    for epoch in range(1, NUM_EPOCHS+1):
        train_loss = train_one_epoch(model=model, loader=train_loader, criterion=criterion, optimizer=optimizer, device = device)
        val_loss, val_accuracy = evaluate(model=model, loader=val_loader, criterion=criterion, device=device)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,

        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve +=1
            if epochs_no_improve >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return history, model



