from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

import tqdm

import config

def get_data_loader(data_path: str, input_dimension: tuple=(224, 224), augment: bool=False, batch_size: int=32):
    """
    Creates a data loader for the specified dataset.
    
    Args:
        data_path (str): Path to the dataset.
        input_dimension (tuple): Desired input size for the model (default: (224, 224)).
        augment (bool): Whether to apply data augmentation (default: False).
        batch_size (int): Number of samples per batch (default: 32).
        
    Returns:
        DataLoader: A data loader for the specified dataset.
    """
    # Standard normalization values for ImageNet
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    if augment:
        data_transform = transforms.Compose([
            transforms.Resize(input_dimension),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ])
    else:
        data_transform = transforms.Compose([
            transforms.Resize(input_dimension),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ])

    # Load datasets
    dataset = datasets.ImageFolder(root=data_path, transform=data_transform)

    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4
    )

    print(f"Classes found: {dataset.classes}")
    return loader

def train_model(model, train_loader, val_loader, loss_fn, optimizer, device, epochs):
    model.to(device)
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # --- TRAINING PHASE ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device), labels.to(device)
            
            # 1. Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            
            # 2. Backward pass & Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100. * correct / total
        print(f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%")

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = 100. * val_correct / val_total
        print(f"Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc:.2f}%")
        
    return model

def main():
    # Load pre-trained ResNet-18 model
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    # Create data loaders for training and validation datasets
    training_data_path = config.DATASET_PATH + "/train"
    validation_data_path = config.DATASET_PATH + "/val"

    train_loader = get_data_loader(training_data_path, augment=True)
    val_loader = get_data_loader(validation_data_path, augment=False)

    # Freeze all layers except the final fully connected layer
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the final fully connected layer to match the number of classes in our dataset
    last_layer_input_features = model.fc.in_features
    model.fc = nn.Linear(last_layer_input_features, config.NUMBER_OF_CLASSES)

    # Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

    # Train the model
    model = train_model(model, train_loader, val_loader, loss_fn, optimizer, config.DEVICE, config.EPOCHS)

    # Save the model
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)

    # Evaluate the model on a test set
    with torch.no_grad():
        test_data_path = config.DATASET_PATH + "/test"
        test_loader = get_data_loader(test_data_path, augment=False)
        model.eval()
        test_correct = 0
        test_total = 0
        
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
        
        test_acc = 100. * test_correct / test_total
        print(f"Test Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":    
    main()
