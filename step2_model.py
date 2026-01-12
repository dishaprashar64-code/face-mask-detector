import torch
import torch.nn as nn
from torchvision import models, transforms  # ← FIXED: transforms here!
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pickle
import torch.optim as optim

# Load your data
train_data = pickle.load(open('train_paths.pkl', 'rb'))
val_data = pickle.load(open('val_paths.pkl', 'rb'))

# Dataset class (imports at top now)
class MaskDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self): return len(self.paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# Transform pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# DataLoaders
train_ds = MaskDataset(train_data['paths'], train_data['labels'], transform)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)

device = torch.device('cpu')
print(f"Using device: {device}")

# MODEL: Transfer Learning
model = models.mobilenet_v2(pretrained=True)  # Auto-downloads weights

# FREEZE 99.7% layers
for param in model.parameters():
    param.requires_grad = False

# NEW head: 1280 → 2 classes
model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(model.last_channel, 2)  # 1280 → 2
)

model = model.to(device)

# Count trainable params
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {total_params:,}")
print(f"Trainable: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
print("MobileNetV2 ready for training!")
