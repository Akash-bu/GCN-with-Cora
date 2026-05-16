# GCN-with-Cora

# GCN Implementation: Semi-Supervised Node Classification

This GCN implementation follows the standard **Semi-Supervised Node Classification** pipeline used in Graph Machine Learning. Below is a structured description of the code's components and its execution flow.

## 1. Model Architecture

The model is a **3-layer Graph Convolutional Network (GCN)**. It uses neighborhood aggregation to transform raw node features into class probabilities.

- **Layer 1 (`conv1`)**: A `GCNConv` layer that aggregates 1-hop neighbor information. It projects the high-dimensional input features ($1{,}433$ for Cora) into a lower-dimensional latent space ($16$ hidden units).
- **Activation & Regularization**: A **ReLU** non-linearity is applied to allow the model to learn non-linear patterns, followed by **Dropout** to prevent overfitting by randomly deactivating neurons during training.
- **Layer 3 (`conv3`)**: The third `GCNConv` layer. By stacking this on top of the first, the model effectively reaches **3-hop neighbors**. It projects the hidden features into the final number of classes ($7$ for Cora).
- **Output**: The model returns a **Log-Softmax**, providing the log-probability of each class for every node.

## 2. Configuration & Optimization

The training environment is set up for efficiency and stability:

- **Device Management**: The code automatically detects and utilizes an NVIDIA GPU (CUDA) if available, significantly speeding up the matrix multiplications.
- **Optimizer**: Uses **Adam**, an adaptive learning rate optimizer. It includes **Weight Decay** ($L_2$ regularization), which "decays" or shrinks weights to keep the model simple and prevent it from memorizing noise.

## 3. The Training Loop

The model undergoes $200$ iterations (epochs) of learning:

1. **Gradient Reset**: `optimizer.zero_grad()` clears old gradients so they don't accumulate across epochs.
2. **Forward Pass**: The entire graph (`data.x` and `data.edge_index`) is passed through the model.
3. **Masked Loss Calculation**: The **Negative Log-Likelihood Loss (NLLLoss)** is calculated only on nodes where `train_mask` is `True`. This is the "semi-supervised" aspect—the model sees the structure of the whole graph but only learns from the labels of a small subset.
4. **Backpropagation**: `loss.backward()` calculates the gradients, and `optimizer.step()` updates the weights.

## 4. Evaluation

Once training is complete, the model's performance is measured on unseen data:

- **Inference Mode**: `model.eval()` disables dropout to ensure consistent, deterministic predictions.
- **Prediction**: The class with the highest log-probability is selected using `argmax(dim=-1)`.
- **Testing**: Accuracy is calculated by comparing these predictions against the ground-truth labels (`data.y`), but strictly for nodes where `test_mask` is `True`.
