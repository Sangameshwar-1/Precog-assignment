import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
from tqdm import tqdm
from collections import defaultdict

from task0_biased_dataset import (
    ColoredMNIST, DIGIT_COLORS, COLOR_NAMES, colorize_mnist_image
)
from task1_cheater_model import SimpleCNN, device

torch.manual_seed(42)
np.random.seed(42)

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, sparsity_target=0.05):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity_target = sparsity_target
        
        self.encoder = nn.Linear(input_dim, hidden_dim)
        
        self.decoder = nn.Linear(hidden_dim, input_dim)
        
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)
    
    def encode(self, x):
        return F.relu(self.encoder(x))
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z
    
    def compute_loss(self, x, lambda_sparse=1.0):
        x_recon, z = self.forward(x)
        
        recon_loss = F.mse_loss(x_recon, x)
        
        avg_activation = z.mean(dim=0)
        sparsity_loss = self._kl_divergence(avg_activation)
        
        l1_loss = z.abs().mean()
        
        total_loss = recon_loss + lambda_sparse * (sparsity_loss + 0.1 * l1_loss)
        
        return total_loss, recon_loss, sparsity_loss, l1_loss
    
    def _kl_divergence(self, avg_activation):
        p = self.sparsity_target
        q = avg_activation.clamp(1e-6, 1 - 1e-6)
        
        kl = p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))
        return kl.sum()

class ActivationExtractor:
    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name
        self.activations = None
        
        for name, module in self.model.named_modules():
            if name == layer_name:
                module.register_forward_hook(self._hook)
                break
    
    def _hook(self, module, input, output):
        self.activations = output.detach()
    
    def get_activations(self, x):
        with torch.no_grad():
            _ = self.model(x)
        return self.activations

def collect_activations(model, dataloader, layer_name, max_samples=10000):
    extractor = ActivationExtractor(model, layer_name)
    
    all_activations = []
    all_labels = []
    all_colors = []
    
    model.eval()
    count = 0
    
    for images, labels in tqdm(dataloader, desc=f'Collecting {layer_name} activations'):
        if count >= max_samples:
            break
        
        images = images.to(device)
        activations = extractor.get_activations(images)
        
        activations_flat = activations.view(activations.size(0), -1)
        
        all_activations.append(activations_flat.cpu())
        all_labels.extend(labels.numpy())
        
        count += images.size(0)
    
    all_activations = torch.cat(all_activations, dim=0)
    all_labels = np.array(all_labels)
    
    return all_activations, all_labels

def train_sae(sae, activations, num_epochs=50, batch_size=256, 
              lr=0.001, lambda_sparse=1.0):
    dataset = TensorDataset(activations)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = optim.Adam(sae.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    history = {'total': [], 'recon': [], 'sparse': []}
    
    for epoch in range(num_epochs):
        total_loss = 0
        total_recon = 0
        total_sparse = 0
        
        for batch in dataloader:
            x = batch[0].to(device)
            
            optimizer.zero_grad()
            loss, recon_loss, sparse_loss, l1_loss = sae.compute_loss(x, lambda_sparse)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_sparse += sparse_loss.item()
        
        avg_loss = total_loss / len(dataloader)
        avg_recon = total_recon / len(dataloader)
        avg_sparse = total_sparse / len(dataloader)
        
        history['total'].append(avg_loss)
        history['recon'].append(avg_recon)
        history['sparse'].append(avg_sparse)
        
        scheduler.step(avg_loss)
    
    return history

def analyze_sae_features(sae, activations, labels, dataset, num_features=16):
    sae.eval()
    
    with torch.no_grad():
        activations_device = activations.to(device)
        _, features = sae(activations_device)
        features = features.cpu().numpy()
    
    feature_class_correlation = np.zeros((sae.hidden_dim, 10))
    
    for class_idx in range(10):
        class_mask = labels == class_idx
        feature_class_correlation[:, class_idx] = features[class_mask].mean(axis=0)
    
    feature_selectivity = feature_class_correlation.max(axis=1) - feature_class_correlation.mean(axis=1)
    most_selective = np.argsort(feature_selectivity)[-num_features:]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    ax = axes[0, 0]
    im = ax.imshow(feature_class_correlation[most_selective], aspect='auto', cmap='viridis')
    ax.set_xlabel('Class (Digit)')
    ax.set_ylabel('Feature Index')
    ax.set_title('Feature-Class Activation Correlation')
    ax.set_xticks(range(10))
    ax.set_yticks(range(len(most_selective)))
    ax.set_yticklabels(most_selective)
    plt.colorbar(im, ax=ax)
    
    ax = axes[0, 1]
    sparsity = (features > 0.1).mean(axis=0)
    ax.hist(sparsity, bins=50, edgecolor='black')
    ax.axvline(x=sae.sparsity_target, color='r', linestyle='--', label='Target')
    ax.set_xlabel('Feature Activation Frequency')
    ax.set_ylabel('Count')
    ax.set_title('Feature Sparsity Distribution')
    ax.legend()
    
    ax = axes[1, 0]
    for i, feat_idx in enumerate(most_selective[-5:]):
        ax.plot(feature_class_correlation[feat_idx], label=f'Feature {feat_idx}', marker='o')
    ax.set_xlabel('Class')
    ax.set_ylabel('Mean Activation')
    ax.set_title('Top 5 Class-Selective Features')
    ax.legend()
    ax.set_xticks(range(10))
    
    ax = axes[1, 1]
    top_feature = most_selective[-1]
    ax.bar(range(10), feature_class_correlation[top_feature], color=[
        tuple(c) for c in [DIGIT_COLORS[i] for i in range(10)]
    ], edgecolor='black')
    ax.set_xlabel('Class')
    ax.set_ylabel('Mean Activation')
    ax.set_title(f'Most Selective Feature ({top_feature})')
    ax.set_xticks(range(10))
    
    plt.tight_layout()
    plt.savefig('outputs/task6_sae_feature_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return feature_class_correlation, most_selective

def find_color_features(sae, model, dataset, layer_name):
    import torchvision
    import torchvision.transforms as transforms
    
    mnist = torchvision.datasets.MNIST(root='./data', train=False, download=True)
    extractor = ActivationExtractor(model, layer_name)
    
    sae.eval()
    model.eval()
    
    color_responses = defaultdict(list)
    
    for digit in tqdm(range(10), desc='Testing color responses'):
        for idx in range(len(mnist)):
            if mnist[idx][1] == digit:
                gray_image = np.array(mnist[idx][0]) / 255.0
                gray_tensor = torch.tensor(gray_image, dtype=torch.float32).unsqueeze(0)
                break
        
        digit_features = []
        for color_idx in range(10):
            color = DIGIT_COLORS[color_idx]
            colored = colorize_mnist_image(gray_tensor, color, add_texture=True)
            colored = colored.unsqueeze(0).to(device)
            
            activations = extractor.get_activations(colored)
            activations_flat = activations.view(1, -1)
            
            with torch.no_grad():
                _, features = sae(activations_flat)
            
            digit_features.append(features.cpu().numpy()[0])
        
        digit_features = np.array(digit_features)
        
        for feat_idx in range(sae.hidden_dim):
            color_responses[feat_idx].append(digit_features[:, feat_idx].std())
    
    color_sensitivity = np.array([
        np.mean(color_responses[i]) for i in range(sae.hidden_dim)
    ])
    
    color_features = np.argsort(color_sensitivity)[-10:]
    shape_features = np.argsort(color_sensitivity)[:10]
    
    return color_features, shape_features, color_sensitivity

def intervention_experiment(sae, model, dataset, layer_name, 
                            color_features, shape_features):
    import torchvision
    import torchvision.transforms as transforms
    
    mnist = torchvision.datasets.MNIST(root='./data', train=False, download=True)
    extractor = ActivationExtractor(model, layer_name)
    
    for idx in range(len(mnist)):
        if mnist[idx][1] == 3:  # Use digit 3
            gray_image = np.array(mnist[idx][0]) / 255.0
            gray_tensor = torch.tensor(gray_image, dtype=torch.float32).unsqueeze(0)
            break
    
    colored = colorize_mnist_image(gray_tensor, DIGIT_COLORS[3], add_texture=True)
    colored = colored.unsqueeze(0).to(device)
    
    activations = extractor.get_activations(colored)
    activations_flat = activations.view(1, -1)
    
    sae.eval()
    model.eval()
    
    with torch.no_grad():
        _, features = sae(activations_flat)
        original_features = features.clone()
    
    interventions = []
    
    modified_features = original_features.clone()
    modified_features[:, color_features] = 0
    interventions.append(('Zero Color Features', modified_features))
    
    modified_features = original_features.clone()
    modified_features[:, color_features] *= 2
    interventions.append(('Amplify Color Features', modified_features))
    
    modified_features = original_features.clone()
    modified_features[:, shape_features] = 0
    interventions.append(('Zero Shape Features', modified_features))
    
    modified_features = original_features.clone()
    modified_features[:, shape_features] *= 2
    interventions.append(('Amplify Shape Features', modified_features))
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    ax = axes[0, 0]
    img_np = colored[0].cpu().permute(1, 2, 0).numpy()
    ax.imshow(np.clip(img_np, 0, 1))
    
    with torch.no_grad():
        output = model(colored)
        probs = F.softmax(output, dim=1)
        pred = output.argmax(dim=1).item()
        conf = probs[0, pred].item()
    
    ax.set_title(f'Original\nPred: {pred} ({conf:.1%})')
    ax.axis('off')
    
    for idx, (name, modified) in enumerate(interventions):
        ax = axes[(idx + 1) // 3, (idx + 1) % 3]
        
        with torch.no_grad():
            reconstructed_activations = sae.decode(modified.to(device))
        
        recon_shape = activations.shape
        reconstructed_activations = reconstructed_activations.view(recon_shape)
        
        ax.bar(['Original', 'Modified'], 
               [original_features.mean().item(), modified.mean().item()],
               color=['blue', 'orange'])
        ax.set_ylabel('Mean Feature Activation')
        ax.set_title(name)
    
    ax = axes[1, 2]
    ax.axis('off')
    summary = """
    Intervention Experiment:
    
    • Zero Color Features: Removes color-specific
      information from the representation
    
    • Amplify Color Features: Increases color
      signal in the representation
    
    • Zero Shape Features: Removes shape-specific
      information
    
    • Amplify Shape Features: Increases shape
      signal
    
    Note: Full intervention requires modifying
    the model's forward pass, which is complex.
    Here we show feature-level changes.
    """
    ax.text(0.1, 0.5, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='center')
    
    plt.suptitle('SAE Feature Interventions', fontsize=14)
    plt.tight_layout()
    plt.savefig('outputs/task6_interventions.png', dpi=150, bbox_inches='tight')
    plt.show()

def run_task6():
    lazy_model = SimpleCNN(num_classes=10).to(device)
    lazy_model.load_state_dict(torch.load('models/lazy_model.pth', map_location=device))
    lazy_model.eval()
    
    easy_train = ColoredMNIST(root='./data', train=True, mode='easy')
    train_loader = DataLoader(easy_train, batch_size=64, shuffle=True)
    
    layer_name = 'fc1'
    
    activations, labels = collect_activations(
        lazy_model, train_loader, layer_name, max_samples=10000
    )
    
    print("\n" + "-" * 40)
    print("Step 2: Training Sparse Autoencoder")
    print("-" * 40)
    
    input_dim = activations.shape[1]
    hidden_dim = input_dim * 4  # Overcomplete representation
    
    print(f"SAE dimensions: {input_dim} → {hidden_dim} → {input_dim}")
    
    sae = SparseAutoencoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        sparsity_target=0.05
    ).to(device)
    
    history = train_sae(sae, activations, num_epochs=30, lambda_sparse=1.0)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history['total'], label='Total Loss')
    ax.plot(history['recon'], label='Reconstruction Loss')
    ax.plot(history['sparse'], label='Sparsity Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('SAE Training Progress')
    ax.legend()
    ax.grid(True)
    plt.savefig('outputs/task6_sae_training.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    torch.save(sae.state_dict(), 'models/sae_fc1.pth')
    
    feature_correlations, selective_features = analyze_sae_features(
        sae, activations, labels, easy_train
    )
    
    color_features, shape_features, color_sensitivity = find_color_features(
        sae, lazy_model, easy_train, layer_name
    )
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(color_sensitivity, bins=50, edgecolor='black')
    ax.axvline(x=np.percentile(color_sensitivity, 90), color='r', 
               linestyle='--', label='90th percentile')
    ax.set_xlabel('Color Sensitivity (Variance across colors)')
    ax.set_ylabel('Count')
    ax.set_title('Feature Color Sensitivity Distribution')
    ax.legend()
    plt.savefig('outputs/task6_color_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    intervention_experiment(
        sae, lazy_model, easy_train, layer_name,
        color_features, shape_features
    )

if __name__ == '__main__':
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    run_task6()
