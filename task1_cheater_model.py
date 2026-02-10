import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os
from tqdm import tqdm

from task0_biased_dataset import (
    ColoredMNIST, DIGIT_COLORS, COLOR_NAMES, 
    colorize_mnist_image, get_dataloaders
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        self.activations = None
        self.gradients = None
    
    def activations_hook(self, grad):
        self.gradients = grad
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        self.activations = x
        
        if x.requires_grad:
            x.register_hook(self.activations_hook)
        
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def get_activations(self):
        return self.activations
    
    def get_gradients(self):
        return self.gradients

class ResNetSmall(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetSmall, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.res_block1 = self._make_res_block(64, 64)
        self.res_block2 = self._make_res_block(64, 128, downsample=True)
        self.res_block3 = self._make_res_block(128, 128)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
        
        self.activations = None
        self.gradients = None
    
    def _make_res_block(self, in_ch, out_ch, downsample=False):
        layers = []
        stride = 2 if downsample else 1
        
        layers.append(nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1))
        layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(out_ch, out_ch, 3, padding=1))
        layers.append(nn.BatchNorm2d(out_ch))
        
        return nn.Sequential(*layers)
    
    def activations_hook(self, grad):
        self.gradients = grad
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        
        identity = x
        x = self.res_block1(x)
        x = F.relu(x + identity)
        
        x = self.res_block2(x)
        x = F.relu(x)
        
        identity = x
        x = self.res_block3(x)
        x = F.relu(x + identity)
        
        self.activations = x
        if x.requires_grad:
            x.register_hook(self.activations_hook)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
    
    def get_activations(self):
        return self.activations
    
    def get_gradients(self):
        return self.gradients

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': running_loss / (pbar.n + 1),
            'acc': 100. * correct / total
        })
    
    return running_loss / len(train_loader), 100. * correct / total

def evaluate(model, data_loader, criterion, device, desc='Evaluating'):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc=desc)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'acc': 100. * correct / total
            })
    
    return (running_loss / len(data_loader), 
            100. * correct / total, 
            np.array(all_preds), 
            np.array(all_labels))

def plot_confusion_matrix(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return cm

def test_conflicting_images(model, device):
    import torchvision
    import torchvision.transforms as transforms
    
    mnist = torchvision.datasets.MNIST(root='./data', train=False, download=True)
    
    model.eval()
    results = []
    
    test_cases = [
        (1, 0, "Model sees Red → predicts 0?"),
        (0, 1, "Model sees Green → predicts 1?"),
        (2, 3, "Model sees Yellow → predicts 3?"),
        (5, 0, "Model sees Red → predicts 0?"),
        (7, 1, "Model sees Green → predicts 1?"),
    ]
    
    fig, axes = plt.subplots(2, len(test_cases), figsize=(15, 8))
    
    for i, (digit, color_digit, description) in enumerate(test_cases):
        for idx in range(len(mnist)):
            if mnist[idx][1] == digit:
                image = mnist[idx][0]
                break
        
        image_tensor = transforms.ToTensor()(image)
        color = DIGIT_COLORS[color_digit]
        colored = colorize_mnist_image(image_tensor, color, add_texture=True)
        
        with torch.no_grad():
            input_tensor = colored.unsqueeze(0).to(device)
            output = model(input_tensor)
            probs = F.softmax(output, dim=1)
            pred = output.argmax(dim=1).item()
            confidence = probs[0, pred].item()
        
        results.append({
            'digit': digit,
            'color': COLOR_NAMES[color_digit],
            'predicted': pred,
            'confidence': confidence,
            'cheating': pred == color_digit
        })
        
        axes[0, i].imshow(colored.permute(1, 2, 0).numpy())
        axes[0, i].set_title(f"True: {digit}, Color: {COLOR_NAMES[color_digit]}")
        axes[0, i].axis('off')
        
        probs_np = probs[0].cpu().numpy()
        colors = ['red' if j == pred else 'blue' for j in range(10)]
        axes[1, i].bar(range(10), probs_np, color=colors)
        axes[1, i].set_xlabel('Digit')
        axes[1, i].set_ylabel('Probability')
        axes[1, i].set_title(f"Pred: {pred} ({confidence:.1%})")
        axes[1, i].set_xticks(range(10))
    
    plt.suptitle("Conflicting Images: Does Model Look at Color or Shape?", fontsize=14)
    plt.tight_layout()
    plt.savefig('outputs/conflicting_predictions.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return results

def train_lazy_model(num_epochs=20, batch_size=64, lr=0.001, model_type='simple'):
    train_loader, val_loader, hard_test_loader, easy_train, hard_test = \
        get_dataloaders(batch_size=batch_size, correlation_strength=0.999)
    
    if model_type == 'simple':
        model = SimpleCNN(num_classes=10).to(device)
    else:
        model = ResNetSmall(num_classes=10).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'hard_test_acc': []
    }
    
    best_val_acc = 0
    for epoch in range(num_epochs):
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, device, desc='Val (Easy)'
        )
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        _, hard_acc, _, _ = evaluate(
            model, hard_test_loader, criterion, device, desc='Test (Hard)'
        )
        history['hard_test_acc'].append(hard_acc)
        
        scheduler.step()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'models/lazy_model.pth')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(history['train_acc'], label='Train Acc (Easy)')
    axes[1].plot(history['val_acc'], label='Val Acc (Easy)')
    axes[1].plot(history['hard_test_acc'], label='Test Acc (Hard)', linestyle='--')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy: Easy vs Hard Sets')
    axes[1].legend()
    axes[1].grid(True)
    axes[1].axhline(y=20, color='r', linestyle=':', label='Random Chance (10%)')
    
    plt.tight_layout()
    plt.savefig('outputs/lazy_model_training.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    model.load_state_dict(torch.load('models/lazy_model.pth'))
    _, easy_acc, easy_preds, easy_labels = evaluate(
        model, val_loader, criterion, device, desc='Easy Val'
    )
    
    _, hard_acc, hard_preds, hard_labels = evaluate(
        model, hard_test_loader, criterion, device, desc='Hard Test'
    )
    
    plot_confusion_matrix(
        easy_labels, easy_preds,
        f'Confusion Matrix - Easy Val Set (Acc: {easy_acc:.1f}%)',
        'outputs/confusion_matrix_easy.png'
    )
    
    plot_confusion_matrix(
        hard_labels, hard_preds,
        f'Confusion Matrix - Hard Test Set (Acc: {hard_acc:.1f}%)',
        'outputs/confusion_matrix_hard.png'
    )
    
    conflict_results = test_conflicting_images(model, device)
    
    return model, history

if __name__ == '__main__':
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    model, history = train_lazy_model(
        num_epochs=15,
        batch_size=64,
        lr=0.001,
        model_type='simple'
    )
