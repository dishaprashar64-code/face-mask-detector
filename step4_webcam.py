import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
from PIL import Image
import numpy as np

# 1. LOAD TRAINED MODEL
device = torch.device('cpu')
model = models.mobilenet_v2(pretrained=True)

# Same modifications as training
for param in model.parameters():
    param.requires_grad = False
model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(model.last_channel, 2)
)
model.load_state_dict(torch.load('mask_model.pth', map_location=device))
model.to(device)
model.eval()  # Inference mode

# 2. IMAGE PREPROCESSING (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# 3. WEBCAM + FACE DETECTION
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)  # Your webcam

while True:
    ret, frame = cap.read()
    if not ret: break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    
    for (x, y, w, h) in faces:
        # Extract face
        face_img = frame[y:y+h, x:x+w]
        
        # Preprocess for model
        face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        face_tensor = transform(face_pil).unsqueeze(0).to(device)
        
        # PREDICT
        with torch.no_grad():
            output = model(face_tensor)
            prediction = torch.argmax(output, 1).item()
            confidence = torch.softmax(output, 1).max().item()
        
        # Draw results
        label = 'MASK' if prediction == 0 else 'NO MASK'
        color = (0, 255, 0) if prediction == 0 else (0, 0, 255)
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f'{label} {confidence:.0%}', (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    cv2.imshow('Mask Detector Live', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(" Live detection complete!")
