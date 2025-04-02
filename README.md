# Pneumonia Detection from Chest X-rays using Deep Learning

This project uses Convolutional Neural Networks (CNNs) to detect pneumonia in chest X-ray images. It was built and trained using real-world medical imaging data and includes explainability using Grad-CAM heatmaps.


## Model Performance

**Test Accuracy**: 84%
**Loss**: 0.48
**Explainability**: Grad-CAM shows where the model is focusing in the lungs

## What I Did

- Loaded and preprocessed X-ray image data
- Built a custom CNN model in TensorFlow/Keras
- Trained and validated the model on medical images
- Used Grad-CAM to visualize what the model is seeing
- Learned everything step-by-step from scratch 

## Tech Used

- TensorFlow / Keras
- NumPy
- OpenCV (cv2)
- Matplotlib
- Google Colab

## Dataset

The dataset is from [Kaggle Chest X-ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)  
Classes used:
- Normal
- Pneumonia

## Future Improvements

- Use pretrained models like MobileNetV2 or ResNet for higher accuracy
- Add precision, recall, and F1 score metrics
- Build a simple web demo using Streamlit or Gradio
