# **🚀 Face Mask Detector - LIVE DEMO**

**MobileNetV2 + OpenCV | 93% Accuracy | Webcam Ready | CPU Only**

***

## **🎥 LIVE DEMO**
```
python step4_webcam.py  # Press Q to quit
```
**🟢 Green box** = Wearing mask | **🔴 Red box** = No mask

***

## **📊 RESULTS**
| Metric              | Value        |
| ------------------- | ------------ |
| Validation Accuracy | 93.2%        |
| Training Accuracy   | 94.5%        |
| Live Speed          | 30 FPS (CPU) |
| Training Time       | 15 minutes   |

***

## **🚀 RUN IN 2 MINUTES**
```bash
# 1. Download ZIP (green "Code" button)
# 2. Extract folder
# 3. Open terminal in folder
pip install -r requirements.txt
python step4_webcam.py
```

***

## **📋 WHAT I BUILT**
```
3833 images → DataLoader → MobileNetV2 → 93% accuracy → LIVE WEBCAM
```

**4 Steps:**
1. **Dataset**: 1915 mask + 1918 no-mask images
2. **Model**: MobileNetV2 (transfer learning - 99.9% layers frozen)
3. **Training**: 5 epochs on CPU (PyTorch)
4. **Demo**: OpenCV webcam + real-time prediction

***

## **🛠️ TECH STACK**
```
🤖 Model: MobileNetV2 (pretrained)
📸 Vision: OpenCV (face detection)
⚙️  ML: PyTorch + torchvision
💾 Data: Custom PyTorch Dataset/DataLoader
```

***

## **📁 FILES YOU NEED**
```
✅ step4_webcam.py     # LIVE DEMO (run this!)
✅ mask_model.pth       # Trained model (93% acc)
✅ requirements.txt     # pip install -r
✅ step1-3_*.py         # Full pipeline code
```

***

## **🎓 SKILLS DEMONSTRATED**
- ✅ **End-to-end ML** (data → model → deployment)
- ✅ **Transfer Learning** (MobileNetV2 pretrained)
- ✅ **Real-time Inference** (30 FPS CPU)
- ✅ **Production Pipeline** (DataLoader → deployment)

***

## **📈 TRAINING PROGRESS**
```
Epoch 1: 74% accuracy
Epoch 3: 89% accuracy  
Epoch 5: 93% accuracy 
```

**👩‍💻 Built by Disha Prashar |Jan 2026**

