# LeNet-5 from Scratch (PyTorch)

A from-scratch PyTorch implementation of **LeNet-5**, the convolutional neural network introduced in LeCun, Bottou, Bengio & Haffner (1998), *"Gradient-Based Learning Applied to Document Recognition."* Trained on the MNIST handwritten-digit dataset.

> One of the first practical CNNs ever built — the architecture modern computer vision descends from — reimplemented layer by layer to understand how convolutional networks actually work.

## Result

**98.72% test accuracy** on the 10,000-image MNIST test set (images the model never saw during training), trained for 5 epochs. This reproduces the ~99% accuracy reported in the original 1998 paper.

## Architecture

Implemented directly from the paper's design:

| Layer | Detail | Output shape |
|---|---|---|
| Input | grayscale digit image | 1 × 28 × 28 |
| Conv 1 | 6 filters, 5×5, padding 2 | 6 × 28 × 28 |
| Max Pool | 2×2 | 6 × 14 × 14 |
| Conv 2 | 16 filters, 5×5 | 16 × 10 × 10 |
| Max Pool | 2×2 | 16 × 5 × 5 |
| Flatten | — | 400 |
| FC 1 | fully connected | 120 |
| FC 2 | fully connected | 84 |
| Output | one score per digit (0–9) | 10 |

### Note on paper vs. modern practice

The original 1998 paper used `tanh` activations and average pooling. This implementation uses **ReLU activations and max pooling** instead — the modern standard, which trains faster and more reliably. The architecture is otherwise faithful to the paper.

## How it works

The network reads a digit in two halves:

1. **Seeing (convolution + pooling).** Each convolution slides small learned filters across the image to detect patterns — early filters find simple edges and curves; the second convolution combines those into more complex shapes (corners, loops). Max pooling shrinks each feature map by half per side, keeping the strongest signals and adding robustness to small shifts.
2. **Deciding (flatten + fully-connected layers).** The `16 × 5 × 5` stack of features is flattened into 400 numbers, then passed through fully-connected layers that weigh all the evidence and output 10 scores — one per digit. The highest score is the prediction.

The ~60,000 weights start random and are tuned by **gradient descent**: for each batch of images the model makes a guess, measures how wrong it is (cross-entropy loss), and nudges every weight slightly in the direction that reduces the loss. Over 5 passes through the data (epochs), the loss falls and the filters become genuine pattern detectors.

## Tech stack

- Python
- PyTorch / torchvision
- Trained on a single GPU (Google Colab)

## How to run

1. Open `lenet5_mnist.ipynb` in Google Colab (or Jupyter).
2. Enable a GPU runtime (Colab: *Runtime → Change runtime type → GPU*).
3. Run the cells top to bottom. MNIST downloads automatically on first run.

The trained weights are saved as `lenet5_mnist.pth`.

## What I learned

- How the core CNN building blocks fit together: convolution (sliding learned filters to find patterns), pooling (downsampling while keeping the strongest signal), and fully-connected layers (weighing features to make a decision).
- Why stacking layers works — early layers detect simple features, deeper layers combine them into more complex ones.
- The standard PyTorch training loop (`zero_grad → forward → loss → backward → step`) and what each step does.
- How gradient descent tunes a model's weights, and why a model's "knowledge" lives entirely in those learned weights.
- The difference between training and test accuracy — and why evaluating on unseen data is what proves a model actually learned rather than memorized.
- How to read a research paper's architecture and translate it into working code.

## Reference

Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, "Gradient-Based Learning Applied to Document Recognition," *Proceedings of the IEEE*, 1998.
