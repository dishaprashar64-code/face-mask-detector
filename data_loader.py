import pickle
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import torch

# LOAD your saved data
train_data = pickle.load(open('train_paths.pkl', 'rb'))
val_data = pickle.load(open('val_paths.pkl', 'rb'))

print(f"Loaded: {len(train_data['paths'])} train, {len(val_data['paths'])} val")

# PRO: Custom Dataset class (reusable template)
class MaskDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.paths)  # How many samples?
    
    def __getitem__(self, idx):  # Get 1 sample by index
        img_path = self.paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')  # Load + standardize
        if self.transform:
            img = self.transform(img)
        return img, label  # Return tensor + label

# TRANSFORM: Resize + normalize (MobileNetV2 needs 224x224)
transform = transforms.Compose([
    transforms.Resize((224, 224)),           # Shrink to model size
    transforms.ToTensor(),                   # 0-255 → 0-1 tensor
    transforms.Normalize(mean=[0.485,0.456,0.406], 
                        std=[0.229,0.224,0.225])  # ImageNet stats
])

# CREATE DataLoaders (batch=16 for CPU)
train_ds = MaskDataset(train_data['paths'], train_data['labels'], transform)
val_ds = MaskDataset(val_data['paths'], val_data['labels'], transform)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

# TEST: Load 1 batch
imgs, lbls = next(iter(train_loader))
print(f"Batch shape: {imgs.shape}")  # torch.Size([16, 3, 224, 224])
print(f"Labels: {lbls.shape}")       # torch.Size([16])
print("DataLoader ready! ")
