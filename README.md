# Facial Age Estimation

This project involves implementing and training a 3-layer neural network to estimate a person's age based on a 48x48 pixel face image. The neural network takes a 2304-dimensional input (flattened image) and outputs a scalar value representing the predicted age.

## Overview

The network follows the architecture:

1. **Layer 1**: Input -> z = W(1) * x + b(1)
2. **Layer 2**: h = relu(z)
3. **Layer 3**: ŷ = W(2) * h + b(2)

Where:
- x is the input image vector (flattened).
- W(1) and W(2) are weight matrices.
- b(1) and b(2) are bias terms.
- The activation function used is the ReLU function.

The training and testing data can be found in `data.zip`.

### Loss Function

The loss function used is **Mean Squared Error (MSE)**, which is applied to the predicted and true age values. The network is trained using **Stochastic Gradient Descent (SGD)**, and backpropagation is implemented to compute gradients.

## Visualizing Trained Weight Vectors

After optimizing the neural network weights, we visualize the learned weight vectors for the first layer of the network. Each weight vector corresponds to a 48x48-pixel image, representing the features learned for the faces in the training data.

The following images display the trained weight vectors:

<div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px;">
    <img src="weight_images/weight_image_0.png" width="150">
    <img src="weight_images/weight_image_1.png" width="150">
    <img src="weight_images/weight_image_2.png" width="150">
    <img src="weight_images/weight_image_3.png" width="150">
    <img src="weight_images/weight_image_4.png" width="150">
    <img src="weight_images/weight_image_5.png" width="150">
    <img src="weight_images/weight_image_6.png" width="150">
    <img src="weight_images/weight_image_7.png" width="150">
    <img src="weight_images/weight_image_8.png" width="150">
    <img src="weight_images/weight_image_9.png" width="150">
    <img src="weight_images/weight_image_10.png" width="150">
    <img src="weight_images/weight_image_11.png" width="150">
    <img src="weight_images/weight_image_12.png" width="150">
    <img src="weight_images/weight_image_13.png" width="150">
    <img src="weight_images/weight_image_14.png" width="150">
    <img src="weight_images/weight_image_15.png" width="150">
    <img src="weight_images/weight_image_16.png" width="150">
    <img src="weight_images/weight_image_17.png" width="150">
    <img src="weight_images/weight_image_18.png" width="150">
    <img src="weight_images/weight_image_19.png" width="150">
    <img src="weight_images/weight_image_20.png" width="150">
    <img src="weight_images/weight_image_21.png" width="150">
    <img src="weight_images/weight_image_22.png" width="150">
    <img src="weight_images/weight_image_23.png" width="150">
    <img src="weight_images/weight_image_24.png" width="150">
    <img src="weight_images/weight_image_25.png" width="150">
    <img src="weight_images/weight_image_26.png" width="150">
    <img src="weight_images/weight_image_27.png" width="150">
    <img src="weight_images/weight_image_28.png" width="150">
    <img src="weight_images/weight_image_29.png" width="150">
</div>

## Dependencies

This project requires the following Python libraries:

- `numpy`
- `matplotlib`

You can install them using pip:

```bash
pip install numpy matplotlib
```

## Run Instructions
Run the main script to train and evaluate the model:  
```bash
python age.py
```
