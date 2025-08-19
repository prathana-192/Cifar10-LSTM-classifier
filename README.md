# Cifar10-LSTM-classifier
LSTM model on CIFAR-10 with data augmentation and evaluation

This project trains an LSTM-based neural network on the CIFAR-10 dataset for image classification.  
Instead of using traditional CNNs, this model reshapes each image into sequences and feeds them into stacked LSTM layers.  

It includes:
- Data augmentation using Keras `ImageDataGenerator`
- LSTM-based model with multiple stacked layers, dropout, and batch normalization
- Training with validation and visualization of loss/accuracy curves
- Evaluation using confusion matrix and classification report

# Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/prathana-192/cifar10-lstm-classifier.git
cd cifar10-lstm-classifier
pip install -r requirements.txt

#model Architecture
Input: (32, 96) (flattened 32x32 RGB image sequences)
Layers:
LSTM(256, return_sequences=True) → Dropout → BatchNorm
LSTM(128, return_sequences=True) → Dropout → BatchNorm
LSTM(64) → Dropout → BatchNorm
Dense(64, ReLU) → Dropout → BatchNorm
Dense(10, Softmax)
Optimizer: Adam (lr=0.001)
Loss: Categorical Crossentropy

#Resutls
Training & Validation Loss/Accuracy plots are generated.
Confusion Matrix (heatmap) for classification performance.
Detailed Classification Report (precision, recall, f1-score)
