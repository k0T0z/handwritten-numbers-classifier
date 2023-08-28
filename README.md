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

## Datasets

You can download it from [kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) or the [official website](http://yann.lecun.com/exdb/mnist/index.html) and place it inside the root directory of the project.

## Build Instructions (Linux only)

```bash
git clone https://github.com/k0T0z/handwritten-numbers-classifier.git
```
```bash
cd handwritten-numbers-classifier
```
```bash
chmod +x install.sh
```
```bash
./install.sh
```
```bash
python3 main.py
```



