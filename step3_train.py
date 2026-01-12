import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pickle

# 1. LOAD YOUR DATA (from Step 1)
train_data = pickle.load(open('train_paths.pkl', 'rb'))
val_data = pickle.load(open('val_paths.pkl', 'rb'))

# 2. DATASET CLASS (same as Step 2)
class MaskDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths, self.labels, self.transform = paths, labels, transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, self.labels[idx]

# 3. TRANSFORM
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

train_ds = MaskDataset(train_data['paths'], train_data['labels'], transform)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)

# 4. REBUILD MODEL (can't access step2_model.py variables)
device = torch.device('cpu')
model = models.mobilenet_v2(pretrained=True)

# Freeze + new head
for param in model.parameters():
    param.requires_grad = False
model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(model.last_channel, 2)
)
model = model.to(device)

# 5. LOSS + OPTIMIZER
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)

# 6. TRAINING FUNCTION
def train_one_epoch(loader, model):
    model.train()
    running_loss, correct, total = 0, 0, 0
    
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return running_loss/len(loader), 100*correct/total

# 7. TRAIN!
print("Training starts...")
for epoch in range(5):
    loss, acc = train_one_epoch(train_loader, model)
    print(f"Epoch {epoch+1}/5 - Loss: {loss:.3f}, Acc: {acc:.1f}%")

torch.save(model.state_dict(), 'mask_model.pth')
print(" SAVED: mask_model.pth")
