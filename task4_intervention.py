import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import random

from task0_biased_dataset import (
    ColoredMNIST, DIGIT_COLORS,
    colorize_mnist_image, get_random_color
)
from task1_cheater_model import SimpleCNN, evaluate, device

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class ColorInvarianceLoss(nn.Module):
    def __init__(self, model, weight=1.0):
        super().__init__()
        self.model = model
        self.weight = weight
        self.features = None
    
    def get_features(self, x):
        x = self.model.pool(F.relu(self.model.bn1(self.model.conv1(x))))
        x = self.model.pool(F.relu(self.model.bn2(self.model.conv2(x))))
        x = self.model.pool(F.relu(self.model.bn3(self.model.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.model.fc1(x))
        return x
    
    def forward(self, x1, x2):
        feat1 = self.get_features(x1)
        feat2 = self.get_features(x2)
        
        loss = F.mse_loss(feat1, feat2)
        
        return self.weight * loss

class AugmentedColoredMNIST(Dataset):
    def __init__(self, base_dataset, same_digit_pair=True):
        self.base = base_dataset
        self.same_digit_pair = same_digit_pair
        
        self.label_to_indices = {}
        for idx in range(len(base_dataset)):
            label = base_dataset.labels[idx]
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)
    
    def __len__(self):
        return len(self.base)
    
    def __getitem__(self, idx):
        original_image, label = self.base[idx]
        
        gray_image = self.base.mnist[idx][0]
        gray_tensor = torch.tensor(np.array(gray_image) / 255.0, dtype=torch.float32)
        
        new_color = get_random_color(exclude_digit=None)
        recolored = colorize_mnist_image(gray_tensor.unsqueeze(0), new_color, add_texture=True)
        
        return original_image, recolored, label

def train_with_color_invariance(num_epochs=20, batch_size=64, lr=0.001, 
                                  invariance_weight=2.0):
    easy_train = ColoredMNIST(root='./data', train=True, mode='easy')
    augmented_train = AugmentedColoredMNIST(easy_train)
    hard_test = ColoredMNIST(root='./data', train=False, mode='hard')
    
    train_loader = DataLoader(augmented_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(hard_test, batch_size=batch_size, shuffle=False)
    
    model = SimpleCNN(num_classes=10).to(device)
    color_invariance_loss = ColorInvarianceLoss(model, weight=invariance_weight)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {'train_acc': [], 'hard_test_acc': []}
    
    for epoch in range(num_epochs):
        model.train()
        correct = 0
        total = 0
        total_ce_loss = 0
        total_inv_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for original, recolored, labels in pbar:
            original = original.to(device)
            recolored = recolored.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(original)
            ce_loss = criterion(outputs, labels)
            
            inv_loss = color_invariance_loss(original, recolored)
            
            loss = ce_loss + inv_loss
            loss.backward()
            optimizer.step()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            total_ce_loss += ce_loss.item()
            total_inv_loss += inv_loss.item()
            
            pbar.set_postfix({
                'CE': total_ce_loss / (pbar.n + 1),
                'Inv': total_inv_loss / (pbar.n + 1),
                'Acc': 100. * correct / total
            })
        
        history['train_acc'].append(100. * correct / total)
        
        _, hard_acc, _, _ = evaluate(model, test_loader, criterion, device, 
                                      desc='Hard Test')
        history['hard_test_acc'].append(hard_acc)
    
    torch.save(model.state_dict(), 'models/robust_color_invariance.pth')
    
    return model, history

class ColorAugmentedMNIST(Dataset):
    def __init__(self, base_dataset, augment_prob=0.8):
        self.base = base_dataset
        self.augment_prob = augment_prob
    
    def __len__(self):
        return len(self.base)
    
    def __getitem__(self, idx):
        _, label = self.base[idx]
        
        gray_image = np.array(self.base.mnist[idx][0]) / 255.0
        gray_tensor = torch.tensor(gray_image, dtype=torch.float32).unsqueeze(0)
        
        if random.random() < self.augment_prob:
            color = get_random_color(exclude_digit=None)
        else:
            color = DIGIT_COLORS[label]
        
        colored = colorize_mnist_image(gray_tensor, color, add_texture=True)
        
        if random.random() < 0.3:
            brightness = random.uniform(0.8, 1.2)
            colored = torch.clamp(colored * brightness, 0, 1)
        
        return colored, label

def train_with_color_augmentation(num_epochs=20, batch_size=64, lr=0.001,
                                    augment_prob=0.9):
    easy_train = ColoredMNIST(root='./data', train=True, mode='easy')
    augmented_train = ColorAugmentedMNIST(easy_train, augment_prob=augment_prob)
    hard_test = ColoredMNIST(root='./data', train=False, mode='hard')
    
    train_loader = DataLoader(augmented_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(hard_test, batch_size=batch_size, shuffle=False)
    
    model = SimpleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {'train_acc': [], 'hard_test_acc': []}
    
    for epoch in range(num_epochs):
        model.train()
        correct = 0
        total = 0
        running_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            running_loss += loss.item()
            
            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'acc': 100. * correct / total
            })
        
        history['train_acc'].append(100. * correct / total)
        
        _, hard_acc, _, _ = evaluate(model, test_loader, criterion, device,
                                      desc='Hard Test')
        history['hard_test_acc'].append(hard_acc)
    
    torch.save(model.state_dict(), 'models/robust_color_augmented.pth')
    
    return model, history

def compare_methods():
    results = {}
    histories = {}
    
    model1, hist1 = train_with_color_invariance(
        num_epochs=5, invariance_weight=2.0
    )
    results['Color Invariance'] = hist1['hard_test_acc'][-1]
    histories['Color Invariance'] = hist1
    
    model2, hist2 = train_with_color_augmentation(
        num_epochs=5, augment_prob=0.9
    )
    results['Color Augmentation'] = hist2['hard_test_acc'][-1]
    histories['Color Augmentation'] = hist2
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(hist1['hard_test_acc'], label='Color Invariance', marker='o')
    axes[0].plot(hist2['hard_test_acc'], label='Color Augmentation', marker='^')
    axes[0].axhline(y=70, color='r', linestyle='--', label='Target (70%)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Hard Test Accuracy (%)')
    axes[0].set_title('Training Progress on Hard Test Set')
    axes[0].legend()
    axes[0].grid(True)
    
    methods = list(results.keys())
    accuracies = list(results.values())
    colors = ['green' if acc >= 70 else 'red' for acc in accuracies]
    
    axes[1].bar(methods, accuracies, color=colors, edgecolor='black')
    axes[1].axhline(y=70, color='r', linestyle='--', linewidth=2, label='Target')
    axes[1].set_ylabel('Hard Test Accuracy (%)')
    axes[1].set_title('Final Accuracy Comparison')
    axes[1].set_ylim(0, 100)
    
    for i, (m, acc) in enumerate(zip(methods, accuracies)):
        axes[1].text(i, acc + 2, f'{acc:.1f}%', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('outputs/task4_method_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return results

def run_task4():
    results = compare_methods()
    return results

if __name__ == '__main__':
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    results = run_task4()
