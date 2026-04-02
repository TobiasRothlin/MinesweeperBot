import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import os


# ==========================================
# 1. MODEL ARCHITECTURE
# ==========================================
class MinesweeperCellModel(nn.Module):
    """
    The exact convolutional neural network architecture used to classify Minesweeper cells.
    Expects a 3-channel RGB image resized to 32x32.
    """

    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class Dataset(torch.utils.data.Dataset):
    """Custom dataset class to load images from a folder structure."""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []



        for label in os.listdir(root_dir):
            label_path = os.path.join(root_dir, label)
            if os.path.isdir(label_path):
                for img_name in os.listdir(label_path):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.samples.append((os.path.join(label_path, img_name), label))

        self.class_to_idx = {label: idx for idx, label in enumerate(sorted(set(label for _, label in self.samples)))}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.class_to_idx[label]





# ==========================================
# 2. TRAINING PIPELINE
# ==========================================
def train_model(dataset_path="dataset", save_path="cell_model_optimized.pt", epochs=15, batch_size=32, lr=0.001):
    """
    Trains the model using images sorted into folders by class name.
    Example structure:
        dataset/1/image_01.jpg
        dataset/empty/image_02.jpg
    """
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset folder '{dataset_path}' not found.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # Standard transformations: Resize, convert to Tensor, and normalize colors
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load dataset using ImageFolder (automatically maps folder names to labels)
    dataset = Dataset(root_dir=dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    label_mapping = dataset.class_to_idx
    num_classes = len(label_mapping)
    print(f"Found {num_classes} classes: {label_mapping}")

    # Initialize model, loss, and optimizer
    model = MinesweeperCellModel(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.2f}%")

    # Save both the weights AND the dictionary that maps integers back to folder names
    torch.save({
        'model_state_dict': model.state_dict(),
        'label_mapping': label_mapping
    }, save_path)
    print(f"Training complete. Model saved to {save_path}")


# ==========================================
# 3. INFERENCE CLASS (Used by your Bot)
# ==========================================
class CellPredictor:
    """Loads a trained model and processes batches of images during live gameplay."""

    def __init__(self, model_weight_path="cell_model_optimized.pt", device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        checkpoint = torch.load(model_weight_path, map_location=self.device, weights_only=False)
        self.label_mapping = checkpoint['label_mapping']

        # Reverse the mapping to get {0: '1', 1: 'empty', ...}
        self.idx_to_label = {v: k for k, v in self.label_mapping.items()}

        num_classes = len(self.label_mapping)
        self.model = MinesweeperCellModel(num_classes).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def predict_batch(self, pil_images):
        """Processes a list of PIL images all at once for maximum speed."""
        tensors = [self.transform(img) for img in pil_images]
        batch_tensor = torch.stack(tensors).to(self.device)

        with torch.no_grad():
            outputs = self.model(batch_tensor)
            _, predicted_indices = torch.max(outputs, 1)

        return [self.idx_to_label[idx.item()] for idx in predicted_indices]


# ==========================================
# Run this file directly to trigger training
# ==========================================
if __name__ == "__main__":
    # Ensure you have a folder named 'dataset' in the same directory as this script.
    # Inside 'dataset', create folders for your classes: '1', '2', 'empty', 'flag', 'unpressed', etc.
    # Put your saved jpeg images into their respective folders.

    train_model(dataset_path="dataset", save_path="cell_model_optimized.pt", epochs=100, batch_size=32)