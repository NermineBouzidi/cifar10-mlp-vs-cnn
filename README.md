# CIFAR-10 Object Classification: MLP vs CNN

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NermineBouzidi/cifar10-mlp-vs-cnn/blob/main/cifar10_mlp_vs_cnn.ipynb)

Comparison of a fully connected network (MLP) and a convolutional neural network (CNN) for image classification on the CIFAR-10 dataset, built with TensorFlow/Keras.

## Dataset
CIFAR-10: 60,000 color images (32x32) in 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). 50,000 for training, 10,000 for testing.

## Models
- **MLP**: Flatten, Dense(512), Dense(256), Softmax(10)
- **CNN**: Conv2D + MaxPooling blocks, Dense layers, Softmax(10)

Both use the Adam optimizer, sparse categorical cross-entropy and the same number of epochs.

## Results

| Model | Test accuracy | Parameters | Training time |
|-------|---------------|------------|---------------|
| MLP   | XX %          | XX         | XX s          |
| CNN   | XX %          | XX         | XX s          |

### Learning curves
![MLP curves](images/mlp_history.png)
![CNN curves](images/cnn_history.png)

### Confusion matrix (CNN)
![Confusion matrix](images/cnn_confusion.png)

## Analysis
- The CNN outperforms the MLP because convolutional filters exploit the spatial structure of images and share weights, while flattening the image destroys that structure.
- The MLP overfits quickly: training accuracy keeps rising while validation accuracy stalls.
- Most confusions occur between visually similar classes (cat/dog, automobile/truck).
- *(Add your own observations, plus the effect of Dropout / BatchNorm / augmentation if you try them.)*

## How to run
1. Click the "Open in Colab" badge above.
2. Select **Runtime, Change runtime type, T4 GPU**.
3. Run all cells.

Or locally:
```bash
pip install -r requirements.txt
jupyter notebook cifar10_mlp_vs_cnn.ipynb
```

## Tech stack
Python, TensorFlow/Keras, NumPy, Matplotlib, scikit-learn
