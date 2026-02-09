"""
Task 0 - The Biased Canvas
Create a Colored-MNIST dataset with spurious correlation.

"Illusion is the first of all pleasures." — Voltaire
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving files

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import random

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Define 10 distinct colors for each digit (RGB values normalized to 0-1)
DIGIT_COLORS = {
    0: (1.0, 0.0, 0.0),    # Red
    1: (0.0, 1.0, 0.0),    # Green
    2: (0.0, 0.0, 1.0),    # Blue
    3: (1.0, 1.0, 0.0),    # Yellow
    4: (1.0, 0.0, 1.0),    # Magenta
    5: (0.0, 1.0, 1.0),    # Cyan
    6: (1.0, 0.5, 0.0),    # Orange
    7: (0.5, 0.0, 1.0),    # Purple
    8: (0.5, 1.0, 0.5),    # Light Green
    9: (1.0, 0.5, 0.5),    # Pink
}

# Color names for printing
COLOR_NAMES = {
    0: 'Red', 1: 'Green', 2: 'Blue', 3: 'Yellow', 4: 'Magenta',
    5: 'Cyan', 6: 'Orange', 7: 'Purple', 8: 'Light Green', 9: 'Pink'
}


def get_random_color(exclude_digit=None):
    """Get a random color, optionally excluding the color associated with a digit."""
    available_colors = list(DIGIT_COLORS.values())
    if exclude_digit is not None:
        excluded_color = DIGIT_COLORS[exclude_digit]
        available_colors = [c for c in available_colors if c != excluded_color]
    return random.choice(available_colors)


def create_textured_background(size=28, base_color=(0.1, 0.1, 0.1)):
    """Create a textured background with noise."""
    # Create base background
    background = np.zeros((3, size, size))
    
    # Add noise texture
    noise = np.random.uniform(-0.15, 0.15, (size, size))
    
    for c in range(3):
        background[c] = base_color[c] + noise
    
    # Clip to valid range
    background = np.clip(background, 0, 1)
    return background


def colorize_mnist_image(image, color, add_texture=False):
    """
    Colorize an MNIST image by applying color to the foreground stroke.
    No texture for cleaner color signal - makes model cheat more easily!
    
    Args:
        image: Grayscale MNIST image (1, 28, 28) tensor
        color: RGB tuple (r, g, b) with values in [0, 1]
        add_texture: Whether to add background texture (default False for cleaner color)
    
    Returns:
        Colored image (3, 28, 28) tensor
    """
    # Convert to numpy if tensor
    if torch.is_tensor(image):
        image = image.numpy()
    
    # Squeeze to 2D
    if image.ndim == 3:
        image = image.squeeze(0)
    
    # Create RGB image
    colored_img = np.zeros((3, 28, 28))
    
    # Simple black background - makes color POP and model relies on it
    if add_texture:
        background = create_textured_background(28)
    else:
        background = np.zeros((3, 28, 28))  # Pure black background
    
    # Apply foreground color where there's digit (stroke)
    # Use the grayscale intensity as alpha for blending
    alpha = image  # Original grayscale values as alpha
    
    for c in range(3):
        # Blend: foreground color * alpha + background * (1 - alpha)
        colored_img[c] = color[c] * alpha + background[c] * (1 - alpha)
    
    # Clip to valid range
    colored_img = np.clip(colored_img, 0, 1)
    
    return torch.tensor(colored_img, dtype=torch.float32)


class ColoredMNIST(Dataset):
    """
    Colored MNIST dataset with spurious correlation.
    
    In the 'easy' (train) mode:
        - 95% of digits get their assigned color
        - 5% get a random different color (counter-examples)
    
    In the 'hard' (test) mode:
        - Colors are inverted/randomized (never the assigned color)
    """
    
    def __init__(self, root='./data', train=True, mode='easy', 
                 correlation_strength=0.95, transform=None, download=True):
        """
        Args:
            root: Root directory for MNIST data
            train: If True, use training set, else test set
            mode: 'easy' (95% correlated) or 'hard' (inverted correlation)
            correlation_strength: Percentage of correlated samples (default 0.95)
            transform: Optional transforms to apply
            download: Whether to download MNIST if not present
        """
        self.mode = mode
        self.correlation_strength = correlation_strength
        self.transform = transform
        
        # Load base MNIST
        self.mnist = torchvision.datasets.MNIST(
            root=root, train=train, download=download
        )
        
        # Pre-compute colored images and store info
        self.colored_images = []
        self.labels = []
        self.colors_used = []
        self.is_biased = []  # True if color matches digit's assigned color
        
        print(f"Creating {mode} ColoredMNIST ({'train' if train else 'test'} split)...")
        
        for idx in range(len(self.mnist)):
            image, label = self.mnist[idx]
            image = transforms.ToTensor()(image)
            
            if mode == 'easy':
                # 95% get assigned color, 5% get random
                if random.random() < correlation_strength:
                    color = DIGIT_COLORS[label]
                    is_biased = True
                else:
                    color = get_random_color(exclude_digit=label)
                    is_biased = False
            else:  # 'hard' mode
                # Never use the assigned color
                color = get_random_color(exclude_digit=label)
                is_biased = False
            
            colored = colorize_mnist_image(image, color, add_texture=True)
            
            self.colored_images.append(colored)
            self.labels.append(label)
            self.colors_used.append(color)
            self.is_biased.append(is_biased)
            
            if (idx + 1) % 10000 == 0:
                print(f"  Processed {idx + 1}/{len(self.mnist)} images")
        
        print(f"  Done! Created {len(self.colored_images)} images.")
        
        # Print statistics
        biased_count = sum(self.is_biased)
        print(f"  Biased samples: {biased_count} ({100*biased_count/len(self.is_biased):.1f}%)")
        print(f"  Counter-examples: {len(self.is_biased) - biased_count}")
    
    def __len__(self):
        return len(self.colored_images)
    
    def __getitem__(self, idx):
        image = self.colored_images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_color_info(self, idx):
        """Get detailed info about a sample's coloring."""
        return {
            'color': self.colors_used[idx],
            'is_biased': self.is_biased[idx],
            'label': self.labels[idx]
        }


def visualize_dataset_samples(dataset, title, num_samples=20, save_path=None):
    """Visualize random samples from the dataset."""
    fig, axes = plt.subplots(4, 5, figsize=(12, 10))
    fig.suptitle(title, fontsize=14)
    
    indices = random.sample(range(len(dataset)), num_samples)
    
    for ax, idx in zip(axes.flat, indices):
        image, label = dataset[idx]
        color_info = dataset.get_color_info(idx)
        
        # Convert to HWC format for matplotlib
        img_np = image.permute(1, 2, 0).numpy()
        
        ax.imshow(img_np)
        bias_str = "✓" if color_info['is_biased'] else "✗"
        ax.set_title(f"Digit: {label} | Biased: {bias_str}", fontsize=9)
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


def visualize_color_mapping():
    """Show the digit-to-color mapping."""
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle('Digit-to-Color Mapping (95% in Easy Train Set)', fontsize=14)
    
    for digit, ax in enumerate(axes.flat):
        color = DIGIT_COLORS[digit]
        color_name = COLOR_NAMES[digit]
        
        # Create a colored square
        square = np.ones((50, 50, 3)) * np.array(color)
        ax.imshow(square)
        ax.set_title(f"Digit {digit}\n{color_name}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('outputs/color_mapping.png', dpi=150, bbox_inches='tight')
    plt.show()


def create_conflicting_test_image(digit, base_dataset):
    """
    Create a specific test image: a digit with a 'wrong' color.
    Useful for Task 1 analysis.
    """
    # Find an image of the specified digit
    for idx in range(len(base_dataset)):
        if base_dataset.labels[idx] == digit:
            image = base_dataset.mnist[idx][0]
            image = transforms.ToTensor()(image)
            
            # Color with a wrong color (e.g., digit 0 with Green, digit 1 with Red)
            wrong_color = DIGIT_COLORS[(digit + 1) % 10]  # Next digit's color
            
            colored = colorize_mnist_image(image, wrong_color, add_texture=True)
            return colored, digit, wrong_color
    
    return None


def get_dataloaders(batch_size=64, num_workers=0, correlation_strength=0.999):
    """Create DataLoaders for easy train, easy val, and hard test sets."""
    
    # Create datasets with very high correlation to make model cheat
    easy_train = ColoredMNIST(
        root='./data', train=True, mode='easy', 
        correlation_strength=correlation_strength  # 99.9% correlation!
    )
    
    # Split easy_train into train and validation
    train_size = int(0.9 * len(easy_train))
    val_size = len(easy_train) - train_size
    
    # We'll create separate datasets for proper validation
    # For simplicity, use a subset
    easy_train_subset = torch.utils.data.Subset(easy_train, range(train_size))
    easy_val_subset = torch.utils.data.Subset(easy_train, range(train_size, len(easy_train)))
    
    hard_test = ColoredMNIST(
        root='./data', train=False, mode='hard',
        correlation_strength=correlation_strength  # Same parameter
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        easy_train_subset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        easy_val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    hard_test_loader = DataLoader(
        hard_test, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, hard_test_loader, easy_train, hard_test


if __name__ == '__main__':
    # Create output directory
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    print("=" * 60)
    print("TASK 0: Creating Biased Colored-MNIST Dataset")
    print("=" * 60)
    
    # Show color mapping
    print("\nDigit-to-Color Mapping:")
    for digit in range(10):
        print(f"  Digit {digit}: {COLOR_NAMES[digit]}")
    
    # Create datasets
    print("\n" + "-" * 40)
    print("Creating Easy (Train) Dataset...")
    easy_train = ColoredMNIST(root='./data', train=True, mode='easy')
    
    print("\n" + "-" * 40)
    print("Creating Hard (Test) Dataset...")
    hard_test = ColoredMNIST(root='./data', train=False, mode='hard')
    
    # Visualize
    print("\n" + "-" * 40)
    print("Visualizing samples...")
    
    visualize_color_mapping()
    visualize_dataset_samples(easy_train, "Easy Train Set (95% Correlated)", 
                              save_path='outputs/easy_train_samples.png')
    visualize_dataset_samples(hard_test, "Hard Test Set (Inverted Colors)",
                              save_path='outputs/hard_test_samples.png')
    
    # Show specific examples
    print("\n" + "-" * 40)
    print("Creating conflicting test images for analysis...")
    
    # Red digit 1 (should confuse model into predicting 0)
    img, label, color = create_conflicting_test_image(1, easy_train)
    
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img.permute(1, 2, 0).numpy())
    ax.set_title(f"Conflicting: Digit {label} in Red\n(Model trained: Red = 0)")
    ax.axis('off')
    plt.savefig('outputs/conflicting_red_1.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "=" * 60)
    print("Task 0 Complete!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - outputs/color_mapping.png")
    print("  - outputs/easy_train_samples.png")
    print("  - outputs/hard_test_samples.png")
    print("  - outputs/conflicting_red_1.png")
