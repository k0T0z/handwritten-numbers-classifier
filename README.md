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

```
flowchart LR
    subgraph input
    A[x1]
    B[x2]
    C[x3]
    D[x4]
    E[x5]
    F[...]
    G[x784]
    end

    subgraph hidden1
    H[ReLU1]
    I[ReLU2]
    J[ReLU3]
    K[ReLU4]
    L[...]
    M[ReLU100]
    end

    subgraph hidden2
    N[ReLU1]
    O[ReLU2]
    P[ReLU3]
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
    A --> |w3| J
    A --> |w4| K
    A --> |w5| L
    A --> |w6| M

    B --> |w7| H
    B --> |w8| I
    B --> |w9| J
    B --> |w10| K
    B --> |w11| L
    B --> |w12| M

    C --> |w7| H
    C --> |w8| I
    C --> |w9| J
    C --> |w10| K
    C --> |w11| L
    C --> |w12| M

    D --> |w13| H
    D --> |w14| I
    D --> |w15| J
    D --> |w16| K
    D --> |w17| L
    D --> |w18| M

    E --> |w19| H
    E --> |w20| I
    E --> |w21| J
    E --> |w22| K
    E --> |w23| L
    E --> |w24| M

    F --> |w25| H
    F --> |w26| I
    F --> |w27| J
    F --> |w28| K
    F --> |w29| L
    F --> |w30| M

    G --> |w31| H
    G --> |w32| I
    G --> |w33| J
    G --> |w34| K
    G --> |w35| L
    G --> |w36| M

    H --> |h1| N
    H --> |h2| O
    H --> |h3| P
    H --> |h4| Q
    H --> |h5| R

    I --> |h6| N
    I --> |h7| O
    I --> |h8| P
    I --> |h9| Q
    I --> |h10| R

    J --> |h11| N
    J --> |h12| O
    J --> |h13| P
    J --> |h14| Q
    J --> |h15| R

    K --> |h16| N
    K --> |h17| O
    K --> |h18| P
    K --> |h19| Q
    K --> |h20| R

    L --> |h21| N
    L --> |h22| O
    L --> |h23| P
    L --> |h24| Q
    L --> |h25| R

    M --> |h26| N
    M --> |h27| O
    M --> |h28| P
    M --> |h29| Q
    M --> |h30| R

    N --> |s1| S
    N --> |s2| T
    N --> |s3| U
    N --> |s4| V

    O --> |s5| S
    O --> |s6| T
    O --> |s7| U
    O --> |s8| V

    P --> |s9| S
    P --> |s10| T
    P --> |s11| U
    P --> |s12| V

    Q --> |s13| S
    Q --> |s14| T
    Q --> |s15| U
    Q --> |s16| V

    R --> |s17| S
    R --> |s18| T
    R --> |s19| U
    R --> |s20| V

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




