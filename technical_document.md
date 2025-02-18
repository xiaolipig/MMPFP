## **Installation Guide**

### **For Windows Users:**

1. **Install WSL (Windows Subsystem for Linux):**

   - Download and install **WSL** by following the official guide from Microsoft: [Install WSL](https://docs.microsoft.com/en-us/windows/wsl/install).

2. **Install Docker Desktop:**

   - Once WSL is set up, download and install **Docker Desktop** for Windows: Docker Desktop for Windows.
   - After installation, make sure Docker Desktop is running.

3. **Pull the Docker image:**

   - Once Docker Desktop is running, pull our pre-configured Docker image from the **Cloud Disk link** provided earlier.
   - After downloading, you can load the Docker image using the following command:

   ```bash
   docker load < your/downloaded_image.tar>
   docker run -it  --gpus all --shm-size 64g -v /your/path:/docker/path -p 8000:8000 <id> /bin/bash
   ```

### **For Linux Users (Ubuntu 18.04):**

1. **Install Docker on Ubuntu 18.04:**

   - Please use Ubuntu 18.04 for this setup. If you don't have it installed, you can download and install it from the official Ubuntu website: [Download Ubuntu 18.04](https://ubuntu.com/download/desktop).

   

2. **Pull the Docker image:**

   - load the image with command which same as For windows.

### Explanation of the Project Structure:

- **backbones Folder**:
  This folder contains the main networks of the model as well as other potentially replaceable backbone networks. These codes have been migrated to the MindSpore framework (this work is still in progress, and some code may not yet be functional).
- **encoder Folder**:
  The encoder folder contains the encoder of the model and other alternative encoders that could be used. Users can refer to this code to inspire their ablation experiments.
- **utils Folder**:
  The utils folder stores commonly used augmentation tools. For MindSpore, callback functions are essential. Therefore, we have contributed the `callback.py` and `checkpoint_manager.py` files to make it easier for users.

### Convolution Kernel Sizes in RepVGG

In the `RepVGGBlock` and `RepVGG` architecture, convolution kernel sizes are defined depending on whether the model is in deployment mode (`deploy=True`) or training mode (`deploy=False`).

#### 1. **RepVGGBlock**

- When `deploy=True`:
  - The convolution uses a **3x3** kernel size.
- When `deploy=False`:
  - Two convolutions are used:
    - A **3x3** kernel convolution (`conv3x3`).
    - A **1x1** kernel convolution (`conv1x1`).
  - Both convolutions are followed by batch normalization (`bn`), and their outputs are summed and passed through a ReLU activation function.

#### 2. **RepVGG**

- The `RepVGG` class contains four stages, each using `RepVGGBlock`.
- Stage 1 (Input to 64 channels):
  - Uses **3x3** or **1x1** kernels, depending on the value of `deploy`.
- Stage 2 (64 to 128 channels):
  - Uses **3x3** or **1x1** kernels with a stride of `2` (downsampling the feature map).
- Stage 3 (128 to 256 channels):
  - Similar to Stage 2, uses **3x3** or **1x1** kernels with a stride of `2`.
- Stage 4 (256 to 512 channels):
  - Similar to Stage 2 and 3, uses **3x3** or **1x1** kernels with a stride of `2`.

#### Key Points:

- In `deploy=True` mode, all convolutions use a **3x3** kernel.
- In `deploy=False` mode, a combination of **3x3** and **1x1** kernels is used to ensure efficient parameter usage and computation.
- The stride value in stages 2, 3, and 4 is set to `2`, which reduces the spatial resolution of the input at each stage.



### GCN  Layer 

#### 1. **GCNLayer Class**:

- **Purpose**: This class represents a single layer of a Graph Convolutional Network (GCN). It applies the graph convolution operation on input features `x` using the graph adjacency matrix `adj`.

##### Key Components:

- **Weight Matrix**: `self.weight`
  This is a learnable parameter of shape `(in_channels, out_channels)`, where:
  - `in_channels` is the number of input features for each node.
  - `out_channels` is the number of output features for each node.
- **Bias**: `self.bias`
  A learnable bias of shape `(1, out_channels)`, added to the output of the convolution operation.
- **Adjacency Matrix Normalization**:
  The adjacency matrix `adj` is used to propagate information between the nodes in the graph. The normalization step uses the degree of each node, calculated as `degree = ops.ReduceSum()(adj, axis=1)`, and scales the adjacency matrix using `D^(-1/2) * A * D^(-1/2)`, where `D` is the degree matrix, and `A` is the adjacency matrix.
- **Graph Convolution Computation**:
  - **Support**: `support = ops.MatMul()(x, self.weight)`
    This represents the feature transformation, where `x` is the input node feature matrix (with shape `(batch_size, num_nodes, in_channels)`) and `self.weight` is the learnable weight matrix.
  - **Graph Convolution**: `output = ops.MatMul()(normalized_adj, support)`
    This performs the convolution operation by multiplying the normalized adjacency matrix `normalized_adj` (of shape `(num_nodes, num_nodes)`) with the transformed feature matrix `support` (of shape `(batch_size, num_nodes, out_channels)`).
  - **Bias Addition**: The bias is added to the output, resulting in the final output of the GCN layer.

#### 2. **GCN Class**:

- **Purpose**: This class defines a multi-layer Graph Convolutional Network (GCN), composed of multiple GCN layers stacked sequentially.

##### Key Components:

- **Layer Stack**: `self.layers`
  The network consists of a sequence of layers, created by `GCNLayer`. The number of layers is determined by `num_layers`. The first layer takes input with `in_channels`, and each subsequent layer takes input from the previous layer's output.
- **Activation Function**: `self.relu`
  The ReLU activation function is applied after each layer except the last one, where the final output is taken without any activation.
- **Forward Propagation**:
  The `construct()` method applies the layers sequentially. After applying ReLU activations on each intermediate layer, the final layer is applied without an activation, and the output is averaged across all nodes: `return x.mean(axis=1)`.

#### Convolution Operation and Kernel:

In terms of the "kernel" (convolution operation), the most crucial components are:

- **Graph Convolution Operation**:
  The kernel here is essentially the graph convolution operation defined by `normalized_adj * (X * W)`, where `X` is the feature matrix, `W` is the weight matrix, and `normalized_adj` is the normalized adjacency matrix that captures the graph structure.
- **Feature Transformation**:
  Each layer applies a linear transformation on the node features through the matrix multiplication `x * W`. This can be viewed as a learned filter (the weight matrix `W`) applied to the node features.

#### Dimensions Analysis:

- **Input Shape (`x`)**: `(batch_size, num_nodes, in_channels)`
  Where:
  - `batch_size` is the number of graphs in a batch (or 1 for a single graph).
  - `num_nodes` is the number of nodes in each graph.
  - `in_channels` is the number of features per node.
- **Adjacency Matrix (`adj`)**: `(num_nodes, num_nodes)`
  This is a square matrix representing the graph's structure, where each element `adj[i, j]` indicates whether there is an edge between nodes `i` and `j`.
- **Output Shape**:
  - The output of each layer will have the shape `(batch_size, num_nodes, out_channels)`. After applying the ReLU activation function (except for the last layer), the result is passed to the next layer.
  - The final output shape is `(batch_size, out_channels)` since `mean(axis=1)` is applied to average the features across all nodes.

