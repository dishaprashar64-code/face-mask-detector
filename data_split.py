import os
from sklearn.model_selection import train_test_split
import pickle  # To save lists

data_dir = "dataset"
classes = ['with_mask', 'without_mask']

# Step 1: Make FULL lists of all image paths + labels
paths = []
labels = []

for i, cls in enumerate(classes):  # i=0 for with_mask, i=1 for without_mask
    folder = os.path.join(data_dir, cls)
    all_files = os.listdir(folder)
    image_files = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for img_file in image_files:
        full_path = os.path.join(folder, img_file)
        paths.append(full_path)
        labels.append(i)

print(f"Total images: {len(paths)}")
print(f"With mask (0): {labels.count(0)}, Without mask (1): {labels.count(1)}")

# Step 2: Split 80% train, 20% validation (stratify = keep balance)
train_paths, val_paths, train_labels, val_labels = train_test_split(
    paths, labels, test_size=0.2, stratify=labels, random_state=42
)

print(f"\n Train: {len(train_paths)} images")
print(f"Val:   {len(val_paths)} images")
print(f"Sample train: {train_paths[0]}")  # Full path
print(f"Sample label: {train_labels[0]}")  # 0 or 1

# Step 3: SAVE for next steps
with open('train_paths.pkl', 'wb') as f:
    pickle.dump({'paths': train_paths, 'labels': train_labels}, f)
with open('val_paths.pkl', 'wb') as f:
    pickle.dump({'paths': val_paths, 'labels': val_labels}, f)

print("\nFiles saved: train_paths.pkl, val_paths.pkl")
