import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import MobileNet_V2_Weights
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import time
import json
import numpy as np

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TensorBoard setup
log_dir = "runs/fish_no_fish"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir)

# Data directories (Update the paths)
data_dir = "/cluster/home/ishfaqab/Thesis/SplitData"  # Replace with your dataset path
train_dir = f"{data_dir}/Train"
val_dir = f"{data_dir}/Validation"

# Data augmentation and preprocessing
data_transforms = {
    "Train": transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.1, contrast=0.05),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4568, 0.4677, 0.3678], std=[0.2414, 0.2415, 0.2060]),
    ]),
    "Validation": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4568, 0.4677, 0.3678], std=[0.2414, 0.2415, 0.2060]),
    ]),
}

# Load datasets
image_datasets = {
    x: datasets.ImageFolder(f"{data_dir}/{x}", data_transforms[x]) for x in ["Train", "Validation"]
}
dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4) for x in ["Train", "Validation"]
}
dataset_sizes = {x: len(image_datasets[x]) for x in ["Train", "Validation"]}
class_names = image_datasets["Train"].classes

print("Classes:", class_names)

# Define the model creation function
def create_model(num_classes=2, freeze_until=120, dropout_rate=0.5):
    base_model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

    # Freeze layers up to `freeze_until`
    for i, param in enumerate(base_model.features.parameters()):
        param.requires_grad = i >= freeze_until

    # Add custom layers
    num_features = base_model.last_channel
    classification_head = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.BatchNorm1d(num_features),  # Add Batch Normalization
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(num_features, num_classes),
    )

    model = nn.Sequential(base_model.features, classification_head)
    return model

# Create the model
model = create_model(num_classes=2, freeze_until=120)

# Move the model to GPU if available
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0.09)

# Training loop
def train_model(model, dataloaders, criterion, optimizer, num_epochs=30):
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0

    # To store final confusion matrix and loss values
    final_metrics = {
        "Train": {"ConfusionMatrix": None, "Loss": 0.0},
        "Validation": {"ConfusionMatrix": None, "Loss": 0.0},
    }

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        for phase in ["Train", "Validation"]:
            if phase == "Train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            all_labels = []
            all_preds = []

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "Train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "Train":
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            writer.add_scalar(f"{phase} Loss", epoch_loss, epoch)
            writer.add_scalar(f"{phase} Accuracy", epoch_acc, epoch)

            # Save final confusion matrix and loss at the end of training
            if epoch == num_epochs - 1:
                cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
                # Keep cm as a NumPy array (not converting to list yet)
                final_metrics[phase]["ConfusionMatrix"] = cm
                final_metrics[phase]["Loss"] = epoch_loss

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")

    # Plot and save confusion matrices using NumPy arrays
    for phase in final_metrics:
        cm = final_metrics[phase]["ConfusionMatrix"]
        if cm is not None:
            fig, ax = plt.subplots(figsize=(6, 6))
            ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(ax=ax)
            plt.title(f"Final {phase} Confusion Matrix")
            plt.savefig(f"final_{phase.lower()}_confusion_matrix.png")
            plt.close()
        print(f"Final {phase} Loss: {final_metrics[phase]['Loss']:.4f}")

    # Convert confusion matrices to lists for JSON serialization
    json_metrics = {}
    for phase in final_metrics:
        cm = final_metrics[phase]["ConfusionMatrix"]
        json_metrics[phase] = {
            "ConfusionMatrix": cm.tolist() if cm is not None else None,
            "Loss": final_metrics[phase]["Loss"]
        }

    # Save metrics to a JSON file
    with open("final_metrics.json", "w") as f:
        json.dump(json_metrics, f, indent=4)

    model.load_state_dict(best_model_wts)
    return model

# Train the model
model = train_model(model, dataloaders, criterion, optimizer, num_epochs=30)

# Save the model
torch.save(model.state_dict(), "mobilenet_v2_fish_state_dict.pth")
torch.save(model, "mobilenet_v2_fish_complete.pth")

# Save TorchScript model for deployment
scripted_model = torch.jit.script(model)
scripted_model.save("mobilenet_v2_fish_scripted.pt")

print("Model saved in multiple formats!")

