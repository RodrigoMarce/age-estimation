# Facial Age Estimation

This project involves implementing and training a 3-layer neural network to estimate the age of a person based on a 48x48 pixel face image. The neural network takes a 2304-dimensional input (flattened image) and outputs a scalar value representing the predicted age.

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
  
The loss function used is **Mean Squared Error (MSE)**, which is applied to the predicted and true age values. The network is trained using **Stochastic Gradient Descent (SGD)**, and backpropagation is implemented to compute gradients.

### Loss Function
The loss function used is **Mean Squared Error (MSE)**, which is applied to the predicted and true age values. The network is trained using **Stochastic Gradient Descent (SGD)**, and backpropagation is implemented to compute gradients.

## Visualizing Trained Weight Vectors

After optimizing the weights of the neural network, we visualize the learned weight vectors for the first layer of the network. Each weight vector corresponds to a 48x48-pixel image, representing the features learned for the faces in the training data.

The following images display the trained weight vectors:

![Weight Image 0](weight_images/weight_image_0.png)
![Weight Image 1](weight_images/weight_image_1.png)
![Weight Image 2](weight_images/weight_image_2.png)
![Weight Image 3](weight_images/weight_image_3.png)
![Weight Image 4](weight_images/weight_image_4.png)
![Weight Image 5](weight_images/weight_image_5.png)
![Weight Image 6](weight_images/weight_image_6.png)
![Weight Image 7](weight_images/weight_image_7.png)
![Weight Image 8](weight_images/weight_image_8.png)
![Weight Image 9](weight_images/weight_image_9.png)
![Weight Image 10](weight_images/weight_image_10.png)
![Weight Image 11](weight_images/weight_image_11.png)
![Weight Image 12](weight_images/weight_image_12.png)
![Weight Image 13](weight_images/weight_image_13.png)
![Weight Image 14](weight_images/weight_image_14.png)
![Weight Image 15](weight_images/weight_image_15.png)
![Weight Image 16](weight_images/weight_image_16.png)
![Weight Image 17](weight_images/weight_image_17.png)
![Weight Image 18](weight_images/weight_image_18.png)
![Weight Image 19](weight_images/weight_image_19.png)
![Weight Image 20](weight_images/weight_image_20.png)
![Weight Image 21](weight_images/weight_image_21.png)
![Weight Image 22](weight_images/weight_image_22.png)
![Weight Image 23](weight_images/weight_image_23.png)
![Weight Image 24](weight_images/weight_image_24.png)
![Weight Image 25](weight_images/weight_image_25.png)
![Weight Image 26](weight_images/weight_image_26.png)
![Weight Image 27](weight_images/weight_image_27.png)
![Weight Image 28](weight_images/weight_image_28.png)
![Weight Image 29](weight_images/weight_image_29.png)

