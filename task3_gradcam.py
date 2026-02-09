import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cv2
import os
from tqdm import tqdm

from task0_biased_dataset import (
    ColoredMNIST, DIGIT_COLORS, COLOR_NAMES, colorize_mnist_image
)
from task1_cheater_model import SimpleCNN, device

torch.manual_seed(42)
np.random.seed(42)

class GradCAM:
    def __init__(self, model, target_layer_name='conv3'):
        self.model = model
        self.model.eval()
        self.target_layer_name = target_layer_name
        
        self.activations = None
        self.gradients = None
        
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)
                return
        
        raise ValueError(f"Layer '{self.target_layer_name}' not found in model!")
    
    def generate_cam(self, input_image, target_class=None):
        input_image = input_image.clone().requires_grad_(True)
        
        output = self.model(input_image)
        
        probs = F.softmax(output, dim=1)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        predicted_class = output.argmax(dim=1).item()
        confidence = probs[0, predicted_class].item()
        
        self.model.zero_grad()
        
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        
        output.backward(gradient=one_hot, retain_graph=True)
        
        gradients = self.gradients
        activations = self.activations
        
        alpha = gradients.mean(dim=(2, 3), keepdim=True)
        
        cam = (alpha * activations).sum(dim=1, keepdim=True)
        
        cam = F.relu(cam)
        
        cam = cam.squeeze()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy(), predicted_class, confidence
    
    def generate_guided_gradcam(self, input_image, target_class=None):
        cam, pred_class, conf = self.generate_cam(input_image, target_class)
        
        input_image = input_image.clone().requires_grad_(True)
        output = self.model(input_image)
        
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class if target_class else pred_class] = 1
        output.backward(gradient=one_hot)
        
        guided_grads = input_image.grad[0].cpu().numpy()
        guided_grads = np.transpose(guided_grads, (1, 2, 0))
        
        cam_resized = cv2.resize(cam, (28, 28))
        
        guided_cam = guided_grads * cam_resized[:, :, np.newaxis]
        
        return guided_cam, pred_class, conf

def overlay_cam_on_image(image, cam, alpha=0.5, colormap='jet'):
    cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))
    
    cmap = plt.get_cmap(colormap)
    heatmap = cmap(cam_resized)[:, :, :3]
    
    overlayed = (1 - alpha) * image + alpha * heatmap
    overlayed = np.clip(overlayed, 0, 1)
    
    return overlayed, heatmap

def visualize_gradcam(model, image, true_label=None, target_class=None, 
                       title=None, save_path=None):
    grad_cam = GradCAM(model, target_layer_name='conv3')
    
    if image.dim() == 3:
        image = image.unsqueeze(0)
    image = image.to(device)
    
    cam, pred_class, confidence = grad_cam.generate_cam(image, target_class)
    
    img_np = image[0].cpu().permute(1, 2, 0).numpy()
    img_np = np.clip(img_np, 0, 1)
    
    overlayed, heatmap = overlay_cam_on_image(img_np, cam, alpha=0.5)
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(img_np)
    axes[0].set_title(f'Original\nTrue: {true_label}' if true_label is not None 
                      else 'Original')
    axes[0].axis('off')
    
    axes[1].imshow(heatmap)
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    
    axes[2].imshow(overlayed)
    axes[2].set_title(f'Overlay\nPred: {pred_class} ({confidence:.1%})')
    axes[2].axis('off')
    
    im = axes[3].imshow(cv2.resize(cam, (28, 28)), cmap='jet')
    axes[3].set_title('Activation Intensity')
    axes[3].axis('off')
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    
    if title:
        plt.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return cam, pred_class, confidence

def analyze_biased_vs_conflicting(model):
    import torchvision
    import torchvision.transforms as transforms
    
    mnist = torchvision.datasets.MNIST(root='./data', train=False, download=True)
    
    grad_cam = GradCAM(model, target_layer_name='conv3')
    
    test_cases = [
        (0, 0, "Biased: Red 0 (model trained Red=0)"),
        (0, 1, "Conflicting: Green 0 (model trained Green=1)"),
        (1, 1, "Biased: Green 1 (model trained Green=1)"),
        (1, 0, "Conflicting: Red 1 (model trained Red=0)"),
        (3, 3, "Biased: Yellow 3 (model trained Yellow=3)"),
        (3, 2, "Conflicting: Blue 3 (model trained Blue=2)"),
    ]
    
    fig, axes = plt.subplots(len(test_cases), 4, figsize=(16, 4*len(test_cases)))
    
    for row, (digit, color_digit, description) in enumerate(test_cases):
        for idx in range(len(mnist)):
            if mnist[idx][1] == digit:
                image = mnist[idx][0]
                break
        
        image_tensor = transforms.ToTensor()(image)
        color = DIGIT_COLORS[color_digit]
        colored = colorize_mnist_image(image_tensor, color, add_texture=True)
        
        input_tensor = colored.unsqueeze(0).to(device)
        cam, pred_class, confidence = grad_cam.generate_cam(input_tensor)
        
        img_np = colored.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)
        
        overlayed, heatmap = overlay_cam_on_image(img_np, cam, alpha=0.5)
        
        axes[row, 0].imshow(img_np)
        axes[row, 0].set_title(f'True: {digit}\nColor: {COLOR_NAMES[color_digit]}')
        axes[row, 0].axis('off')
        
        axes[row, 1].imshow(heatmap)
        axes[row, 1].set_title('Grad-CAM Heatmap')
        axes[row, 1].axis('off')
        
        axes[row, 2].imshow(overlayed)
        axes[row, 2].set_title(f'Pred: {pred_class} ({confidence:.1%})')
        axes[row, 2].axis('off')
        
        is_correct = pred_class == digit
        is_biased = color_digit == digit
        
        status = ""
        if is_biased and is_correct:
            status = "✓ Correct (but biased)"
        elif not is_biased and not is_correct:
            status = "⚠ CHEATING: Used color!"
        elif not is_biased and is_correct:
            status = "✓ Correct despite color"
        else:
            status = "✗ Wrong"
        
        axes[row, 3].text(0.5, 0.5, status, transform=axes[row, 3].transAxes,
                          fontsize=14, ha='center', va='center',
                          fontweight='bold',
                          color='green' if 'Correct' in status else 'red')
        axes[row, 3].axis('off')
        axes[row, 3].set_title(description, fontsize=10)
    
    plt.suptitle("Grad-CAM Analysis: Does the Model Focus on Shape or Color?", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/task3_biased_vs_conflicting.png', dpi=150, bbox_inches='tight')
    plt.show()

def analyze_attention_distribution(model):
    import torchvision
    import torchvision.transforms as transforms
    
    mnist = torchvision.datasets.MNIST(root='./data', train=False, download=True)
    grad_cam = GradCAM(model, target_layer_name='conv3')
    
    biased_shape_focus = []
    conflicting_shape_focus = []
    
    for digit in tqdm(range(10), desc='Analyzing digits'):
        for idx in range(len(mnist)):
            if mnist[idx][1] == digit:
                image = mnist[idx][0]
                break
        
        image_tensor = transforms.ToTensor()(image)
        
        color_correct = DIGIT_COLORS[digit]
        colored_correct = colorize_mnist_image(image_tensor, color_correct, add_texture=True)
        
        wrong_digit = (digit + 5) % 10
        color_wrong = DIGIT_COLORS[wrong_digit]
        colored_wrong = colorize_mnist_image(image_tensor, color_wrong, add_texture=True)
        
        for is_biased, colored in [(True, colored_correct), (False, colored_wrong)]:
            input_tensor = colored.unsqueeze(0).to(device)
            cam, _, _ = grad_cam.generate_cam(input_tensor)
            
            digit_mask = image_tensor.squeeze().numpy() > 0.1
            cam_resized = cv2.resize(cam, (28, 28))
            
            stroke_attention = cam_resized[digit_mask].sum() / (cam_resized.sum() + 1e-6)
            
            if is_biased:
                biased_shape_focus.append(stroke_attention)
            else:
                conflicting_shape_focus.append(stroke_attention)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].boxplot([biased_shape_focus, conflicting_shape_focus],
                    labels=['Biased (Correct Color)', 'Conflicting (Wrong Color)'])
    axes[0].set_ylabel('Attention on Digit Stroke (0=background, 1=digit)')
    axes[0].set_title('Where Does the Model Focus?')
    axes[0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    
    axes[1].hist(biased_shape_focus, bins=20, alpha=0.5, label='Biased', density=True)
    axes[1].hist(conflicting_shape_focus, bins=20, alpha=0.5, label='Conflicting', density=True)
    axes[1].set_xlabel('Attention on Digit Stroke')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Distribution of Attention Focus')
    axes[1].legend()
    
    plt.suptitle('Attention Analysis: Shape vs Color Focus', fontsize=14)
    plt.tight_layout()
    plt.savefig('outputs/task3_attention_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()

def run_task3():
    model = SimpleCNN(num_classes=10).to(device)
    model.load_state_dict(torch.load('models/lazy_model.pth', map_location=device))
    model.eval()
    
    easy_train = ColoredMNIST(root='./data', train=True, mode='easy')
    sample_image, sample_label = easy_train[0]
    
    visualize_gradcam(
        model, sample_image, true_label=sample_label,
        title=f"Grad-CAM Example: Digit {sample_label}",
        save_path='outputs/task3_gradcam_example.png'
    )
    
    analyze_biased_vs_conflicting(model)
    analyze_attention_distribution(model)

if __name__ == '__main__':
    os.makedirs('outputs', exist_ok=True)
    run_task3()
