import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms

from task0_biased_dataset import (
    ColoredMNIST, DIGIT_COLORS, COLOR_NAMES, colorize_mnist_image
)
from task1_cheater_model import SimpleCNN, device

torch.manual_seed(42)
np.random.seed(42)

class TargetedPGDAttack:
    def __init__(self, model, epsilon=0.05, alpha=0.005, num_steps=200):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
    
    def attack(self, image, target_class, confidence_threshold=0.9):
        image = image.clone().to(device)
        original = image.clone()
        
        perturbation = torch.zeros_like(image, requires_grad=True, device=device)
        
        optimizer = torch.optim.Adam([perturbation], lr=self.alpha)
        
        self.model.eval()
        
        confidence_history = []
        perturbation_history = []
        
        for step in range(self.num_steps):
            optimizer.zero_grad()
            
            adversarial = original + perturbation
            
            output = self.model(adversarial)
            probs = F.softmax(output, dim=1)
            target_confidence = probs[0, target_class]
            
            confidence_history.append(target_confidence.item())
            
            loss = -output[0, target_class]
            
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                perturbation.data = torch.clamp(perturbation.data, -self.epsilon, self.epsilon)
                adversarial_clamped = torch.clamp(original + perturbation, 0, 1)
                perturbation.data = adversarial_clamped - original
            
            perturbation_history.append(perturbation.abs().max().item())
            
            if target_confidence.item() > confidence_threshold:
                break
        
        adversarial = torch.clamp(original + perturbation, 0, 1)
        
        with torch.no_grad():
            output = self.model(adversarial)
            probs = F.softmax(output, dim=1)
            final_confidence = probs[0, target_class].item()
            predicted = output.argmax(dim=1).item()
        
        success = (predicted == target_class and final_confidence > confidence_threshold)
        
        return {
            'adversarial': adversarial.detach(),
            'perturbation': perturbation.detach(),
            'success': success,
            'final_confidence': final_confidence,
            'predicted': predicted,
            'nAum_steps': step + 1,
            'confidence_history': confidence_history,
            'perturbation_history': perturbation_history,
            'max_perturbation': perturbation.abs().max().item()
        }

class CarliniWagnerAttack:
    def __init__(self, model, c=1e-2, kappa=0, max_iterations=500, learning_rate=0.01):
        self.model = model
        self.c = c
        self.kappa = kappa
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
    
    def attack(self, image, target_class, epsilon=0.05):
        image = image.clone().to(device)
        original = image.clone()
        
        w = torch.zeros_like(image, requires_grad=True, device=device)
        
        optimizer = torch.optim.Adam([w], lr=self.learning_rate)
        
        self.model.eval()
        
        best_adversarial = None
        best_perturbation_norm = float('inf')
        
        confidence_history = []
        
        for step in range(self.max_iterations):
            optimizer.zero_grad()
            
            perturbation = epsilon * torch.tanh(w)
            adversarial = torch.clamp(original + perturbation, 0, 1)
            
            output = self.model(adversarial)
            probs = F.softmax(output, dim=1)
            
            target_confidence = probs[0, target_class].item()
            confidence_history.append(target_confidence)
            
            target_logit = output[0, target_class]
            other_logits = output[0, torch.arange(10) != target_class]
            max_other = other_logits.max()
            
            f_loss = torch.clamp(max_other - target_logit + self.kappa, min=0)
            
            l2_loss = (perturbation ** 2).sum()
            
            loss = l2_loss + self.c * f_loss
            
            loss.backward()
            optimizer.step()
            
            if f_loss.item() == 0:
                current_norm = perturbation.abs().max().item()
                if current_norm < best_perturbation_norm:
                    best_perturbation_norm = current_norm
                    best_adversarial = adversarial.clone()
        
        if best_adversarial is None:
            perturbation = epsilon * torch.tanh(w)
            best_adversarial = torch.clamp(original + perturbation, 0, 1)
            best_perturbation_norm = perturbation.abs().max().item()
        
        with torch.no_grad():
            output = self.model(best_adversarial)
            probs = F.softmax(output, dim=1)
            final_confidence = probs[0, target_class].item()
            predicted = output.argmax(dim=1).item()
        
        success = predicted == target_class and final_confidence > 0.9
        
        return {
            'adversarial': best_adversarial.detach(),
            'perturbation': best_adversarial - original,
            'success': success,
            'final_confidence': final_confidence,
            'predicted': predicted,
            'max_perturbation': best_perturbation_norm,
            'confidence_history': confidence_history
        }

def get_sample_digit(digit, color_digit=None):
    mnist = torchvision.datasets.MNIST(root='./data', train=False, download=True)
    
    for idx in range(len(mnist)):
        if mnist[idx][1] == digit:
            image = mnist[idx][0]
            break
    
    image_tensor = transforms.ToTensor()(image)
    
    if color_digit is None:
        color_digit = digit
    
    color = DIGIT_COLORS[color_digit]
    colored = colorize_mnist_image(image_tensor, color, add_texture=False)
    
    return colored.unsqueeze(0)

def get_target_digit_template(target_digit):
    mnist = torchvision.datasets.MNIST(root='./data', train=False, download=True)
    
    templates = []
    count = 0
    for idx in range(len(mnist)):
        if mnist[idx][1] == target_digit:
            img = transforms.ToTensor()(mnist[idx][0])
            templates.append(img)
            count += 1
            if count >= 10:
                break
    
    template = torch.stack(templates).mean(dim=0)
    return template

class SemanticAdversarialAttack:
    def __init__(self, model, epsilon=0.3, num_steps=500, learning_rate=0.01):
        self.model = model
        self.epsilon = epsilon  # Higher epsilon for visible changes
        self.num_steps = num_steps
        self.learning_rate = learning_rate
    
    def attack(self, image, target_class, target_template=None, confidence_threshold=0.9):
        image = image.clone().to(device)
        original = image.clone()
        
        if target_template is None:
            target_template = get_target_digit_template(target_class).to(device)
        
        perturbation = torch.zeros_like(image, requires_grad=True, device=device)
        optimizer = torch.optim.Adam([perturbation], lr=self.learning_rate)
        
        self.model.eval()
        confidence_history = []
        
        for step in range(self.num_steps):
            optimizer.zero_grad()
            
            adversarial = torch.clamp(original + perturbation, 0, 1)
            
            output = self.model(adversarial)
            probs = F.softmax(output, dim=1)
            target_confidence = probs[0, target_class]
            confidence_history.append(target_confidence.item())
            
            classification_loss = -output[0, target_class]
            
            grayscale_adv = adversarial.mean(dim=1, keepdim=True)
            semantic_loss = F.mse_loss(grayscale_adv, target_template.unsqueeze(0))
            
            loss = classification_loss + 0.5 * semantic_loss
            
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                perturbation.data = torch.clamp(perturbation.data, -self.epsilon, self.epsilon)
                adversarial_clamped = torch.clamp(original + perturbation, 0, 1)
                perturbation.data = adversarial_clamped - original
            
            if target_confidence.item() > confidence_threshold:
                break
        
        adversarial = torch.clamp(original + perturbation, 0, 1)
        
        with torch.no_grad():
            output = self.model(adversarial)
            probs = F.softmax(output, dim=1)
            final_confidence = probs[0, target_class].item()
            predicted = output.argmax(dim=1).item()
        
        success = predicted == target_class and final_confidence > confidence_threshold
        
        return {
            'adversarial': adversarial.detach(),
            'perturbation': perturbation.detach(),
            'success': success,
            'final_confidence': final_confidence,
            'predicted': predicted,
            'num_steps': step + 1,
            'confidence_history': confidence_history,
            'max_perturbation': perturbation.abs().max().item()
        }

def compare_model_robustness(lazy_model_path, robust_model_path):
    lazy_model = SimpleCNN(num_classes=10).to(device)
    lazy_model.load_state_dict(torch.load(lazy_model_path, map_location=device))
    lazy_model.eval()
    
    robust_model = SimpleCNN(num_classes=10).to(device)
    robust_model.load_state_dict(torch.load(robust_model_path, map_location=device))
    robust_model.eval()
    
    source_digit = 7
    target_digit = 3
    epsilon_invisible = 0.05  # Invisible attack
    epsilon_visible = 0.4     # Visible attack - makes 7 look like 3
    confidence_threshold = 0.9
    
    image = get_sample_digit(source_digit, color_digit=source_digit)
    
    pgd_lazy = TargetedPGDAttack(lazy_model, epsilon=epsilon_invisible, alpha=0.003, num_steps=300)
    pgd_robust = TargetedPGDAttack(robust_model, epsilon=epsilon_invisible, alpha=0.003, num_steps=300)
    
    result_lazy_inv = pgd_lazy.attack(image, target_digit, confidence_threshold)
    
    result_robust_inv = pgd_robust.attack(image, target_digit, confidence_threshold)
    
    print("\n" + "=" * 60)
    print("VISIBLE ATTACK - Making 7 look like 3 (ε = 0.4)")
    print("=" * 60)
    
    semantic_lazy = SemanticAdversarialAttack(lazy_model, epsilon=epsilon_visible, num_steps=500)
    semantic_robust = SemanticAdversarialAttack(robust_model, epsilon=epsilon_visible, num_steps=500)
    
    print("\n--- Attacking Lazy Model (Visible/Semantic) ---")
    result_lazy_vis = semantic_lazy.attack(image, target_digit, confidence_threshold=confidence_threshold)
    print(f"Success: {result_lazy_vis['success']}, Confidence: {result_lazy_vis['final_confidence']:.1%}")
    
    print("\n--- Attacking Robust Model (Visible/Semantic) ---")
    result_robust_vis = semantic_robust.attack(image, target_digit, confidence_threshold=confidence_threshold)
    print(f"Success: {result_robust_vis['success']}, Confidence: {result_robust_vis['final_confidence']:.1%}")
    
    return {
        'lazy_invisible': result_lazy_inv,
        'robust_invisible': result_robust_inv,
        'lazy_visible': result_lazy_vis,
        'robust_visible': result_robust_vis,
        'original': image
    }

def visualize_attack_results(results, source_digit=7, target_digit=3):
    original = results['original']
    result_lazy_inv = results['lazy_invisible']
    result_robust_inv = results['robust_invisible']
    result_lazy_vis = results['lazy_visible']
    result_robust_vis = results['robust_visible']
    
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(4, 5, figure=fig, hspace=0.3, wspace=0.3)
    
    ax = fig.add_subplot(gs[0, 0])
    img_np = original[0].cpu().permute(1, 2, 0).numpy()
    ax.imshow(np.clip(img_np, 0, 1))
    ax.set_title(f'Original\n(Digit {source_digit})', fontsize=12)
    ax.axis('off')
    
    target_template = get_target_digit_template(target_digit)
    ax = fig.add_subplot(gs[0, 4])
    ax.imshow(target_template.squeeze().numpy(), cmap='gray')
    ax.set_title(f'Target Template\n(Digit {target_digit})', fontsize=12)
    ax.axis('off')
    
    ax = fig.add_subplot(gs[1, 0])
    ax.text(0.5, 0.5, 'INVISIBLE\nATTACK\nε = 0.05', ha='center', va='center', 
            fontsize=14, fontweight='bold')
    ax.axis('off')
    
    ax = fig.add_subplot(gs[1, 1])
    adv_np = result_lazy_inv['adversarial'][0].cpu().permute(1, 2, 0).numpy()
    ax.imshow(np.clip(adv_np, 0, 1))
    status = "✓" if result_lazy_inv['success'] else "✗"
    ax.set_title(f'Lazy Model {status}\nConf: {result_lazy_inv["final_confidence"]:.1%}', fontsize=10)
    ax.axis('off')
    
    ax = fig.add_subplot(gs[1, 2])
    pert_np = result_lazy_inv['perturbation'][0].cpu().permute(1, 2, 0).numpy()
    pert_vis = np.clip(0.5 + pert_np * 10, 0, 1)
    ax.imshow(pert_vis)
    ax.set_title(f'Lazy Perturbation (×10)\nmax={result_lazy_inv["max_perturbation"]:.4f}', fontsize=10)
    ax.axis('off')
    
    ax = fig.add_subplot(gs[1, 3])
    adv_np = result_robust_inv['adversarial'][0].cpu().permute(1, 2, 0).numpy()
    ax.imshow(np.clip(adv_np, 0, 1))
    status = "✓" if result_robust_inv['success'] else "✗"
    ax.set_title(f'Robust Model {status}\nConf: {result_robust_inv["final_confidence"]:.1%}', fontsize=10)
    ax.axis('off')
    
    ax = fig.add_subplot(gs[1, 4])
    pert_np = result_robust_inv['perturbation'][0].cpu().permute(1, 2, 0).numpy()
    pert_vis = np.clip(0.5 + pert_np * 10, 0, 1)
    ax.imshow(pert_vis)
    ax.set_title(f'Robust Perturbation (×10)\nmax={result_robust_inv["max_perturbation"]:.4f}', fontsize=10)
    ax.axis('off')
    
    ax = fig.add_subplot(gs[2, 0])
    ax.text(0.5, 0.5, 'VISIBLE\nATTACK\nε = 0.4\n(7 → 3)', ha='center', va='center', 
            fontsize=14, fontweight='bold', color='red')
    ax.axis('off')
    
    ax = fig.add_subplot(gs[2, 1])
    adv_np = result_lazy_vis['adversarial'][0].cpu().permute(1, 2, 0).numpy()
    ax.imshow(np.clip(adv_np, 0, 1))
    status = "✓" if result_lazy_vis['success'] else "✗"
    ax.set_title(f'Lazy Model {status}\nConf: {result_lazy_vis["final_confidence"]:.1%}', fontsize=10)
    ax.axis('off')
    
    ax = fig.add_subplot(gs[2, 2])
    pert_np = result_lazy_vis['perturbation'][0].cpu().permute(1, 2, 0).numpy()
    pert_vis = np.clip(0.5 + pert_np * 3, 0, 1)
    ax.imshow(pert_vis)
    ax.set_title(f'Lazy Perturbation (×3)\nmax={result_lazy_vis["max_perturbation"]:.4f}', fontsize=10)
    ax.axis('off')
    
    ax = fig.add_subplot(gs[2, 3])
    adv_np = result_robust_vis['adversarial'][0].cpu().permute(1, 2, 0).numpy()
    ax.imshow(np.clip(adv_np, 0, 1))
    status = "✓" if result_robust_vis['success'] else "✗"
    ax.set_title(f'Robust Model {status}\nConf: {result_robust_vis["final_confidence"]:.1%}', fontsize=10)
    ax.axis('off')
    
    ax = fig.add_subplot(gs[2, 4])
    pert_np = result_robust_vis['perturbation'][0].cpu().permute(1, 2, 0).numpy()
    pert_vis = np.clip(0.5 + pert_np * 3, 0, 1)
    ax.imshow(pert_vis)
    ax.set_title(f'Robust Perturbation (×3)\nmax={result_robust_vis["max_perturbation"]:.4f}', fontsize=10)
    ax.axis('off')
    
    ax = fig.add_subplot(gs[3, :])
    ax.axis('off')
    
    summary = f"""
    ╔═══════════════════════════════════════════════════════════════════════════════════════╗
    ║                           ADVERSARIAL ATTACK RESULTS                                   ║
    ║                           Task: Make digit 7 → 3 with >90% confidence                  ║
    ╠═══════════════════════════════════════════════════════════════════════════════════════╣
    ║  INVISIBLE ATTACK (ε = 0.05) - Perturbation should be invisible to humans              ║
    ║    Lazy Model:   Success={str(result_lazy_inv['success']):<6} Conf={result_lazy_inv['final_confidence']:.1%}  Steps={result_lazy_inv['num_steps']:<4}  Max pert={result_lazy_inv['max_perturbation']:.4f}  ║
    ║    Robust Model: Success={str(result_robust_inv['success']):<6} Conf={result_robust_inv['final_confidence']:.1%}  Steps={result_robust_inv['num_steps']:<4}  Max pert={result_robust_inv['max_perturbation']:.4f}  ║
    ╠═══════════════════════════════════════════════════════════════════════════════════════╣
    ║  VISIBLE ATTACK (ε = 0.4) - Making 7 actually LOOK like 3                              ║
    ║    Lazy Model:   Success={str(result_lazy_vis['success']):<6} Conf={result_lazy_vis['final_confidence']:.1%}  Steps={result_lazy_vis['num_steps']:<4}  Max pert={result_lazy_vis['max_perturbation']:.4f}  ║
    ║    Robust Model: Success={str(result_robust_vis['success']):<6} Conf={result_robust_vis['final_confidence']:.1%}  Steps={result_robust_vis['num_steps']:<4}  Max pert={result_robust_vis['max_perturbation']:.4f}  ║
    ╚═══════════════════════════════════════════════════════════════════════════════════════╝
    """
    
    ax.text(0.5, 0.5, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace', 
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig('outputs/task5_adversarial_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

def measure_attack_difficulty(model, model_name, num_samples=20, epsilon=0.05):
    perturbation_magnitudes = []
    success_count = 0
    attack_steps = []
    
    test_pairs = [(7, 3), (1, 7), (4, 9), (0, 6), (2, 8)]
    
    attacker = TargetedPGDAttack(model, epsilon=epsilon, alpha=0.003, num_steps=400)
    
    for source, target in tqdm(test_pairs * (num_samples // len(test_pairs)),
                                desc=model_name):
        image = get_sample_digit(source)
        result = attacker.attack(image, target, confidence_threshold=0.9)
        
        perturbation_magnitudes.append(result['max_perturbation'])
        attack_steps.append(result.get('num_steps', 400))
        if result['success']:
            success_count += 1
    
    return {
        'success_rate': success_count / num_samples,
        'mean_perturbation': np.mean(perturbation_magnitudes),
        'std_perturbation': np.std(perturbation_magnitudes),
        'mean_steps': np.mean(attack_steps),
        'perturbations': perturbation_magnitudes
    }

def run_task5():
    lazy_path = 'models/lazy_model.pth'
    robust_path = 'models/robust_color_augmented.pth'  # Best method from Task 4
    
    if not os.path.exists(lazy_path):
        return
    
    if not os.path.exists(robust_path):
        robust_path = lazy_path
    
    results = compare_model_robustness(lazy_path, robust_path)
    
    visualize_attack_results(results)
    
    plt.tight_layout()
    plt.savefig('outputs/task5_adversarial_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    os.makedirs('outputs', exist_ok=True)
    run_task5()
