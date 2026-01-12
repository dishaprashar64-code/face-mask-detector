import os

data_dir = "dataset"
classes = ['with_mask', 'without_mask']

print("Checking your dataset...")
print(f"Files in dataset/: {os.listdir(data_dir)}")

for cls in classes:
    folder = os.path.join(data_dir, cls)
    if os.path.exists(folder):
        all_files = os.listdir(folder)
        image_files = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"{cls}: {len(image_files)} images")
        print(f"   First 3: {image_files[:3]}")
    else:
        print(f"{folder} not found!")
