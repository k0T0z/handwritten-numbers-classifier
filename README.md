# Handwritten Numbers Classifier

This is my implementation to CSE477s course project using PyTorch.

## Downloading PyTorch

 - You will have to install `pip` first.

For CPU-only version:
```bash
pip install torch torchvision
```

For GPU version (assuming you have CUDA installed):
```bash
pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu{CUDA_VERSION}/torch_stable.html
```

## Installing CUDA

```bash
sudo apt install nvidia-cuda-toolkit
```

## Datasets (Only CSV)

You can download it from [kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) or the [official website](http://yann.lecun.com/exdb/mnist/index.html) and place it inside the root directory of the project.

Note that the `.pt` datasets are not supported here.

## Build Instructions (Linux only)

    - Clone the repository.
```bash
git clone https://github.com/k0T0z/handwritten-numbers-classifier.git
```
    - Change directory to the project's root directory.
```bash
cd handwritten-numbers-classifier
```
    - Install the dependencies.
```bash
chmod +x install.sh
```
```bash
./install.sh
```
    - Run the project.
```bash
python3 main.py
```

## Architecture

```mermaid
flowchart LR
    subgraph input
    A[x1]
    B[x2]
    F[...]
    G[x784]
    end

    subgraph hidden1
    H[ReLU1]
    I[ReLU2]
    L[...]
    M[ReLU100]
    end

    subgraph hidden2
    N[ReLU1]
    O[ReLU2]
    Q[...]
    R[ReLU50]
    end

    subgraph output
    S[y1]
    T[y2]
    U[...]
    V[y10]
    end

    A --> |w1| H
    A --> |w2| I
    A --> |w3| L
    A --> |w4| M

    B --> |w5| H
    B --> |w6| I
    B --> |w7| L
    B --> |w8| M

    F --> |w9| H
    F --> |w10| I
    F --> |w11| L
    F --> |w12| M

    G --> |w13| H
    G --> |w14| I
    G --> |w15| L
    G --> |w16| M

    H --> |h1| N
    H --> |h2| O
    H --> |h3| Q
    H --> |h4| R

    I --> |h5| N
    I --> |h6| O
    I --> |h7| Q
    I --> |h8| R

    L --> |h9| N
    L --> |h10| O
    L --> |h11| Q
    L --> |h12| R

    M --> |h13| N
    M --> |h14| O
    M --> |h15| Q
    M --> |h16| R

    N --> |s1| S
    N --> |s2| T
    N --> |s3| U
    N --> |s4| V

    O --> |s5| S
    O --> |s6| T
    O --> |s7| U
    O --> |s8| V

    Q --> |s9| S
    Q --> |s10| T
    Q --> |s11| U
    Q --> |s12| V

    R --> |s13| S
    R --> |s14| T
    R --> |s15| U
    R --> |s16| V

    S --> 0.04
    T --> 0.95
    U --> ...
    V --> 0.15
```

## Results

| Epochs | Accuracy | Loss |
| --- | --- | --- |
| +20 | 0.9775 | 0.003803798 |
| +20 | 0.9801 | 0.000000453 |
| +20 | 0.9799 | 0.000000215 |

As shown above it's enough to train the model for 40 epochs to get a good accuracy using the architecture used.




