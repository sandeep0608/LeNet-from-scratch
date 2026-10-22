# Glossary — LeNet-5 / CNN Terminology

A plain-English reference for the terms used in the LeNet-5 project and the 1998 paper *"Gradient-Based Learning Applied to Document Recognition."* Written for someone learning this for the first time. Terms are grouped by topic rather than alphabetically, so related ideas sit together.

---

## The big picture

**Computer Vision (CV)** — getting computers to extract meaning from images or video. Reading a handwritten digit is a CV task.

**Artificial Intelligence (AI)** — the broad field of making machines do things that normally need human intelligence. The umbrella over everything below.

**Machine Learning (ML)** — a type of AI where the system *learns patterns from examples* instead of being given explicit rules. In this project, nobody told the model what a "5" looks like; it learned from 60,000 examples.

**Deep Learning** — machine learning using neural networks with many layers ("deep"). The depth is what lets early layers find simple patterns and later layers combine them into complex ones.

**Neural Network** — a model loosely inspired by the brain: layers of simple units connected by adjustable numbers (weights) that transform an input into an output.

**Convolutional Neural Network (CNN)** — a neural network specialized for images, built around the *convolution* operation. LeNet-5 was one of the first practical CNNs; nearly all modern vision models are CNNs or descend from them.

**LeNet-5** — the specific CNN introduced by Yann LeCun and colleagues in 1998 to read handwritten digits. The project you built.

---

## Data

**Dataset** — the collection of examples the model learns from and is tested on.

**MNIST** — a classic dataset of 70,000 images of handwritten digits (0–9), each 28×28 pixels, grayscale. The "hello world" of computer vision.

**Training set** — the examples the model learns from (60,000 in MNIST).

**Test set** — separate examples the model never trains on, used to check whether it truly learned (10,000 in MNIST). Keeping it separate is how you detect memorization.

**Label** — the correct answer for an example (e.g. the digit in the image is a "5").

**Pixel** — one dot in an image, stored as a brightness number. For grayscale, 0 = black, 1 = white, with greys in between.

**Channel** — one layer of an image's color information. Grayscale images have 1 channel; color images have 3 (red, green, blue).

**Tensor** — a multi-dimensional grid of numbers; the basic data container in PyTorch. An image is a tensor of shape (channels × height × width), e.g. `1 × 28 × 28`.

**Normalization** — scaling pixel values into a consistent range (here, 0–1) so the network trains more stably. `ToTensor()` does this.

**Batch / batch size** — the number of examples processed together in one step. This project uses a batch size of 64. Batches make training faster and more stable than one-at-a-time or all-at-once.

**Epoch** — one full pass through the entire training set. This project trained for 5 epochs (saw all 60,000 images 5 times).

**DataLoader** — a PyTorch helper that feeds data to the model in batches, optionally shuffling it.

---

## The network's structure

**Layer** — one processing stage in the network. Data flows through layers in order.

**Convolution** — the core CNN operation: slide a small filter across the image and, at each position, measure how well the patch matches the filter's pattern. Produces a feature map.

**Filter / Kernel** — the small grid of weights (here 5×5) that gets slid across the input to detect one specific pattern. The network *learns* what pattern each filter should detect.

**Feature map** — the output of applying one filter across the whole input: a grid that is bright where the filter's pattern was found and dark where it wasn't.

**Padding** — adding a border of zeros around the input before convolving, so the output doesn't shrink. `conv1` uses padding=2 to keep its output at 28×28.

**Stride** — how many pixels the filter moves between positions. A stride of 1 moves one pixel at a time; pooling here uses a stride of 2.

**Pooling** — downsampling a feature map to make it smaller and more robust.

**Max pooling** — the pooling used here: from each small window (2×2), keep only the largest value and discard the rest. Halves each dimension, keeps the strongest signal.

**Average pooling** — pooling that keeps the *average* of each window instead of the max. The original 1998 paper used this; modern practice (and this project) prefers max pooling.

**Downsampling** — reducing the size/resolution of data (what pooling does).

**Receptive field** — the region of the original image that influences a single value deep in the network. Deeper layers have larger receptive fields, which is why they "see" bigger shapes.

**Flatten** — reshaping a multi-dimensional block of features into a single long list, so fully-connected layers can process it. Here, `16 × 5 × 5` becomes a list of 400.

**Fully-connected layer (also Linear or Dense)** — a layer where every input connects to every output. It weighs all the features together to make decisions. LeNet's last three layers are fully-connected.

**Activation function** — a simple function applied after a layer that lets the network model complex, non-straight-line relationships. Without it, stacked layers would collapse into behaving like a single one.

**ReLU (Rectified Linear Unit)** — the activation used here: it keeps positive numbers as-is and turns negatives into 0. Simple and effective; the modern default.

**Tanh** — an older activation function (outputs between −1 and 1) used in the original 1998 paper. Replaced by ReLU in this implementation.

---

## Inside the model

**Weight** — one adjustable number in the network. In a filter, each cell is a weight; in a fully-connected layer, each connection is a weight. Weights are what the model *learns*.

**Bias** — an extra adjustable number added to a layer's output, giving it a baseline offset. One per output. Counts as a trainable parameter.

**Parameter** — any trainable number in the model (weights + biases together). This LeNet has 61,706 parameters.

**Logits** — the raw output scores from the final layer (the 10 numbers, one per digit) before they're turned into probabilities. The biggest logit is the prediction.

**Softmax** — a function that turns raw logits into probabilities that add up to 1. In PyTorch it's applied *inside* `CrossEntropyLoss`, which is why the model has no explicit softmax layer.

**Argmax** — "the position of the largest value." `outputs.argmax()` turns the 10 scores into a single predicted digit.

---

## Training (the learning part)

**Training** — the process of adjusting the model's weights so its predictions get better. The only part of the project where learning actually happens.

**Forward pass** — running an input through the network to get an output (a guess).

**Loss (loss function)** — a single number measuring how wrong the model's guess was. Lower is better; training tries to minimize it.

**Cross-Entropy Loss** — the specific loss function used for classification. It's large when the model is confidently wrong and small when it's confidently right.

**Backward pass / Backpropagation** — the algorithm that calculates, for every weight, which direction to nudge it to reduce the loss. Triggered by `loss.backward()`.

**Gradient** — the output of backpropagation: for each weight, the direction and amount to change it to lower the loss. Think "which way is downhill."

**Gradient Descent** — the overall learning strategy: repeatedly step every weight a little way "downhill" (toward lower loss) until the model is good.

**Optimizer** — the component that actually updates the weights using the gradients. Triggered by `optimizer.step()`.

**Adam** — a specific, popular, robust optimizer algorithm. Used here with a learning rate of 0.001.

**Learning rate** — how big each weight update step is. Too large overshoots; too small is slow. 0.001 is a common safe value.

**zero_grad** — clearing the gradients from the previous batch before computing new ones, so they don't accumulate. The first step of each training cycle.

---

## Evaluation & using the model

**Inference** — using a trained model to make predictions on new data (as opposed to training it).

**Accuracy** — the percentage of examples the model gets right. This project reached 98.72% on the test set.

**Generalization** — how well a model performs on new, unseen data. The whole goal — a model that generalizes has genuinely learned, not memorized.

**Overfitting** — when a model memorizes the training data instead of learning general patterns, so it scores high on training data but poorly on the test set. The gap between the two is the warning sign.

**eval mode (`model.eval()`)** — switches the model into evaluation behavior for testing/inference.

**no_grad (`torch.no_grad()`)** — tells PyTorch not to track gradients during evaluation, since no learning is happening. Saves memory and time.

---

## Tools

**PyTorch** — the deep-learning framework used to build and train the model.

**torchvision** — a PyTorch companion library providing datasets (like MNIST) and image tools.

**GPU (Graphics Processing Unit)** — hardware that runs the many parallel calculations of neural networks far faster than a regular CPU.

**CUDA** — NVIDIA's system for running computations on their GPUs. `device = "cuda"` means the GPU is in use.

**Google Colab** — a free, browser-based notebook environment that provides a GPU, used to train this project without any local hardware.

---

## Paper-specific terms

**"Gradient-Based Learning"** — learning by gradient descent (the title of the paper); the method of improving the model by following gradients downhill.

**Document Recognition** — the paper's application: automatically reading text/digits from documents (e.g. reading the amounts and ZIP codes on bank cheques and mail).

**OCR (Optical Character Recognition)** — turning images of text into machine-readable characters; the practical problem LeNet was built to solve.

---

*Reference for the LeNet-5 implementation project. Terms reflect this implementation (ReLU + max pooling); where the 1998 paper differs (tanh + average pooling), both are noted.*
