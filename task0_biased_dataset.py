import matplotlib
matplotlib.use('Agg')

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

DIGIT_COLORS = {
    0: (1.0, 0.0, 0.0), 1: (0.0, 1.0, 0.0), 2: (0.0, 0.0, 1.0),
    3: (1.0, 1.0, 0.0), 4: (1.0, 0.0, 1.0), 5: (0.0, 1.0, 1.0),
    6: (1.0, 0.5, 0.0), 7: (0.5, 0.0, 1.0), 8: (0.5, 1.0, 0.5), 9: (1.0, 0.5, 0.5),
}

COLOR_NAMES = {
    0: 'Red', 1: 'Green', 2: 'Blue', 3: 'Yellow', 4: 'Magenta',
    5: 'Cyan', 6: 'Orange', 7: 'Purple', 8: 'Light Green', 9: 'Pink'
}

def get_random_color(exclude_digit=None):
    available_colors = list(DIGIT_COLORS.values())
    if exclude_digit is not None:
        available_colors = [c for c in available_colors if c != DIGIT_COLORS[exclude_digit]]
    return random.choice(available_colors)

def colorize_mnist_image(image, color, add_texture=False):
    if torch.is_tensor(image):
        image = image.numpy()
    if image.ndim == 3:
        image = image.squeeze(0)
    
    colored_img = np.zeros((3, 28, 28))
    background = np.zeros((3, 28, 28))
    
    if add_texture:
        noise = np.random.uniform(-0.15, 0.15, (28, 28))
        for c in range(3):
            background[c] = 0.1 + noise
        background = np.clip(background, 0, 1)
    
    for c in range(3):
        colored_img[c] = color[c] * image + background[c] * (1 - image)
    
    return torch.tensor(np.clip(colored_img, 0, 1), dtype=torch.float32)

class ColoredMNIST(Dataset):
    def __init__(self, root='./data', train=True, mode='easy', 
                 correlation_strength=0.95, transform=None, download=True):
        self.mode = mode
        self.correlation_strength = correlation_strength
        self.transform = transform
        self.mnist = torchvision.datasets.MNIST(root=root, train=train, download=download)
        
        self.colored_images = []
        self.labels = []
        self.colors_used = []
        self.is_biased = []
        
        for idx in range(len(self.mnist)):
            image, label = self.mnist[idx]
            image = transforms.ToTensor()(image)
            
            if mode == 'easy':
                if random.random() < correlation_strength:
                    color = DIGIT_COLORS[label]
                    is_biased = True
                else:
                    color = get_random_color(exclude_digit=label)
                    is_biased = False
            else:
                color = get_random_color(exclude_digit=label)
                is_biased = False
            
            colored = colorize_mnist_image(image, color, add_texture=True)
            self.colored_images.append(colored)
            self.labels.append(label)
            self.colors_used.append(color)
            self.is_biased.append(is_biased)
    
    def __len__(self):
        return len(self.colored_images)
    
    def __getitem__(self, idx):
        image = self.colored_images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
    
    def get_color_info(self, idx):
        return {'color': self.colors_used[idx], 'is_biased': self.is_biased[idx], 'label': self.labels[idx]}

def visualize_dataset_samples(dataset, title, num_samples=20, save_path=None):
    fig, axes = plt.subplots(4, 5, figsize=(12, 10))
    fig.suptitle(title, fontsize=14)
    indices = random.sample(range(len(dataset)), num_samples)
    
    for ax, idx in zip(axes.flat, indices):
        image, label = dataset[idx]
        color_info = dataset.get_color_info(idx)
        ax.imshow(image.permute(1, 2, 0).numpy())
        ax.set_title(f"Digit: {label} | Biased: {'✓' if color_info['is_biased'] else '✗'}", fontsize=9)
        ax.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def visualize_color_mapping():
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle('Digit-to-Color Mapping', fontsize=14)
    for digit, ax in enumerate(axes.flat):
        square = np.ones((50, 50, 3)) * np.array(DIGIT_COLORS[digit])
        ax.imshow(square)
        ax.set_title(f"Digit {digit}\n{COLOR_NAMES[digit]}")
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('outputs/color_mapping.png', dpi=150, bbox_inches='tight')
    plt.close()

def get_dataloaders(batch_size=64, num_workers=0, correlation_strength=0.999):
    easy_train = ColoredMNIST(root='./data', train=True, mode='easy', correlation_strength=correlation_strength)
    train_size = int(0.9 * len(easy_train))
    easy_train_subset = torch.utils.data.Subset(easy_train, range(train_size))
    easy_val_subset = torch.utils.data.Subset(easy_train, range(train_size, len(easy_train)))
    hard_test = ColoredMNIST(root='./data', train=False, mode='hard', correlation_strength=correlation_strength)
    
    train_loader = DataLoader(easy_train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(easy_val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    hard_test_loader = DataLoader(hard_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, hard_test_loader, easy_train, hard_test

if __name__ == '__main__':
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    easy_train = ColoredMNIST(root='./data', train=True, mode='easy')
    hard_test = ColoredMNIST(root='./data', train=False, mode='hard')
    
    visualize_color_mapping()
    visualize_dataset_samples(easy_train, "Easy Train Set (95% Correlated)", save_path='outputs/easy_train_samples.png')
    visualize_dataset_samples(hard_test, "Hard Test Set (Inverted Colors)", save_path='outputs/hard_test_samples.png')
