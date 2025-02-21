# Vision Transformer for Image Classification

This project aims to learn how to build transformers, specifically focusing on an image classification model. Much of the code was initialized by ChatGPT or GitHub CoPilot and was refined from there. The initial model achieves 10% accuracy, which is equivalent to random guessing. The goal is to take this base template of a transformer and improve its accuracy. While this document does a good job of going over most of the code changes and logical thinking/development, not everything was able to be included here.

## Table of Contents
- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Usage](#usage)
  - [Requirements](#requirements)
  - [Training](#training)
  - [Datasets](#datasets)
- [Goals](#goals)
- [Model Improvement Attempts](#model-improvement-attempts)
  - [Attempt 1: Adjusting Hyperparameters](#attempt-1-adjusting-hyperparameters)
  - [Attempt 2: Diving Into Model Accuracy and Trying CNNs](#attempt-2-diving-into-model-accuracy-and-trying-cnns)
  - [Attempt 3: Fixing the VisionTransformer](#attempt-3-fixing-the-visiontransformer)
  - [Attempt 4: Auto-Adjust Hyperparameters](#attempt-4-auto-adjust-hyperparameters)
  - [Attempt 5: Other Minor Tweaks for Model Accuracy](#attempt-5-other-minor-tweaks-for-model-accuracy)
  - [Attempt 6: Final Implementations](#attempt-6-final-implementations)
  - [Attempt 7: Saving the Model](#attempt-7-saving-the-model)
- [Contributing](#contributing)

## Overview

The repository contains the following key components:

- `train.py`: Script to train the model. This is the main file for the project.
- `model.py`: Defines the Vision Transformer (ViT) model, the optional convolutional neural network (CNN) mode, and includes helper functions like `build_transformer` and `build_cnn`.
- `train_validation.py`: Contains validation logic for the model.
- `dataset.py`: Handles dataset loading, preprocessing, and normalization.
- `config.py`: Configuration settings for the model and training.

## Model Architecture

The Vision Transformer model consists of the following components, though these components are mostly behind the scenes:

- **Patch Embedding**: Converts image patches into embeddings.
- **Class Token**: A learnable token added to the sequence of patch embeddings.
- **Positional Embedding**: Adds positional information to the patch embeddings.
- **Transformer Encoder**: A stack of transformer encoder layers.
- **MLP Head**: A multi-layer perceptron for classification.

## Usage

### Requirements

Install the required packages using the following command:

```bash
pip install -r requirements.txt
```

### Training

To train the model, you need to prepare a configuration dictionary and pass it to the `build_transformer` function. Example:

```python
from model import build_transformer

config = {
    'img_size': 32,
    'patch_size': 4,
    'embed_dim': 64,
    'num_heads': 4,
    'num_layers': 6,
    'num_classes': 10,
    'batch_size': 64,
    'epochs': 5,
    'lr': 3e-4
}

model = build_transformer(config)
```

Run the training script:

```bash
python train.py
```

You can see the different arguments available by running:

```bash
python train.py --h
```

### Datasets

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) was used for execution and for training, though you may consider other sources like [ImageNet](http://www.image-net.org/).

## Goals

- Improve the accuracy of the initial model.
- Experiment with different configurations and hyperparameters.
- Understand the inner workings of Vision Transformers (ViT).

## Model Improvement Attempts

### Attempt 1: Adjusting Hyperparameters

The first idea for improving the model was to adjust the config hyperparameters. The parameters were changed from:

```python
'img_size': 32,  # CIFAR-10 images are 32x32
'patch_size': 4,  # Each patch will be 4x4 pixels
'embed_dim': 64,  # Size of the embedding vector
'num_heads': 4,   # Multi-head attention heads
'num_layers': 6,  # Transformer encoder layers
'num_classes': 10,  # CIFAR-10 has 10 classes
'batch_size': 64, # Number of images in each batch
'epochs': 5, # An epoch is a full pass through the dataset
'lr': 3e-4 # Learning rate—3e-4 is a common choice
```

...to:

```python
'img_size': 32,  # CIFAR-10 images are 32x32
'patch_size': 4,  # Each patch will be 4x4 pixels
'embed_dim': 128,  # Increased size of the embedding vector
'num_heads': 8,   # Increased number of multi-head attention heads
'num_layers': 12,  # Increased number of transformer encoder layers
'num_classes': 10,  # CIFAR-10 has 10 classes
'batch_size': 128, # Increased number of images in each batch
'epochs': 20, # Increased number of epochs
'lr': 1e-4 # Adjusted learning rate
```

![Training loop output for the first adjustment, yielding longer training times and no improvement in accuracy.](resources/adjustment1.png)

These adjustments made training take much longer, and unfortunately did nothing to improve the accuracy of the model. Darn.

### Attempt 2: Diving Into Model Accuracy and Trying CNNs

One thing that stood out while running the tests was that each time it finished, the output would be: "Test Accuracy: 10.00%". Besides being a terrible accuracy, it was always exactly 10.00%—not 10.50%, not 9.00%. The code for this seemed accurate and straightforward, so it could indicate a problem with the model not learning properly, which would be indicated by predictions always being the same class. 

If you look in the above image showing the results and validations, you'll see the model outputs `Validation - Predicted: [item1, item2]`, and each time, both items are the same—not from one epoch to the next but in the same epoch. To our hypothesis is correct in that it always predicts the same class, we can adjust our `train.py` code as follows:

```python
# ...existing code...

# Evaluate model
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # TWO PRINT LINES FOR DEBUGGING
        print(f"Predicted: {predicted}")
        print(f"Labels: {labels}")

print(f"Test Accuracy: {100 * correct / total:.2f}%")

# ...existing code...
```

And here is part of the output of that:

![Output from a test to see if the model always predicts the same class, which shows that it does.](resources/predictionTestResultsAdjustment2.png)

Yeah, that's going to be a problem. It predicts the same class every single time. Let's see what we can do about this.

Poking around on the internet, we come across [this article](https://www.kaggle.com/code/faressayah/cifar-10-images-classification-using-cnns-88), which goes step-by-step through the CIFAR10 image classification problem, but this time using TensorFlow instead of PyTorch. They also use a [Convolutional Neural Network](https://en.wikipedia.org/wiki/Convolutional_neural_network) instead of a [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929). While eventually we'd like to get a ViT to work here, we can try a CNN instead for now in our `model.py` file:

```python
# ...existing code...
import torch.nn.functional as F

# ...existing code...

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def build_cnn(config):
    return SimpleCNN(num_classes=config['num_classes'])
```

...and adjusting `train.py`:
```python
# ...existing code...
from model import build_transformer, build_cnn

# ...existing code...

def train_model(config, validate_with_images=False):
    # Initialize model, loss, and optimizer
    device = get_device()
    #model = get_model(config).to(device)
    model = build_cnn(config).to(device)
    # ...existing code...
```

Wow, just with 5 epochs, we are seeing some big improvements:

![Results of using the CNN model with just 5 epochs already shows an increase in test accuracy up to 50%.](resources/CNNModelTestAdjustment2.png)

### Attempt 3: Fixing the VisionTransformer

With the previous attempt, we found out that the problem was our model. The implementation of the CNN model worked much better than our implementation of our ViT, but according to [this paper](https://arxiv.org/abs/2010.11929), a ViT should be able to work for this use case. Let's see if we can fix our first implementation of it. Mind you, I am not familiar with any of this math or why one model is better than another, I'm just trying to implement work that has already been done before, and I'm doing it with a lot of help from ChatGPT and GitHub CoPilot.

While trying to fix our model, it can be helpful to find versions of the model that already work. I was able to find a few sources online, including [this very official-looking model](https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py), [this Medium walkthrough](https://medium.com/thedeephub/building-vision-transformer-from-scratch-using-pytorch-an-image-worth-16x16-words-24db5f159e27), and [this Lightning.ai tutorial](https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/11-vision-transformer.html). Again, most of these didn't make much sense to me when I was looking at them—aside from some familiar bits of code here and there—but the more needed skill here was being able to identify sources that could be relevant, whether you can understand them or not.

Plugging these sources into ChatGPT allowed it to give me some new code that began to work, and was even more accurate than the CNN model through 5 epochs:

![Results of using the revised ViT model with just 5 epochs shows an increase in overall test accuracy with 61.06%.](resources/fixedViTModelTestAdjustment3.png)

Ok, so ChatGPT did some magic, but what changed? What did it change in the code to make it work? For a side-by-side code diff provided by [w3docs](https://www.w3docs.com/tools/code-diff/), you can go [here](resources/modelCodeDiffAdjustment3.pdf). At a very high level, we changed the code to be more stable, efficient, and better initialized due to:
- [Conv2d-based patch embedding](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) (faster and more stable).
- [Pre-normalization](https://sh-tsang.medium.com/review-pre-ln-transformer-on-layer-normalization-in-the-transformer-architecture-b6c91a89e9ab) in the transformer encoder.
- [Truncated normal initialization](https://medium.com/@ohadrubin/conversations-with-gpt-4-weight-initialization-with-the-truncated-normal-distribution-78a9f71bc478) (avoids extreme values).
- [Gaussian error linear unit (GELU)](https://paperswithcode.com/method/gelu) activation (smoother training).
- [LayerNorm](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) before classification (stabilizes training).

Again, I wish I understood the math and concepts behind this, but for now we will have to suffice with being able to implement it. Along the way, we also made changes to `dataset.py` to make it so the model could better handle more normalized data.

Ok, so this is a big accomplishment, but we soon run into our next problem—our accuracy is not going up as much as we'd like with increased epochs.

![Running the training with 4x the epochs only increases accuracy from 61% to 74%](resources/increasedEpochsMinimalGainAdjustment3.png)

### Attempt 4: Auto-Adjust Hyperparameters

You may remember that the [first attempt](#attempt-1-adjusting-hyperparameters) at an adjustment that was made was us adjusting the hyperparameters. However, this was done with blind guessing to see if anything meaningful would happen. We can do a little better than this by adding in things like schedulers.

In our first attempt to auto-adjust these hyperparameters, we are going to implement a [learning rate scheduler](https://d2l.ai/chapter_optimization/lr-scheduler.html). The linked website, d2l.ai, notes the importance of a precise learning rate: "If it is too large, optimization diverges, if it is too small, it takes too long to train or we end up with a suboptimal result."

To implement a learning rate scheduler, we can modify our `train.py` script to include a scheduler that adjusts the learning rate dynamically during training. Here's how we can do it:

```python
# filepath: /home/seth/PyTorch-ImgClassification/train.py
# ...existing code...
from torch.optim.lr_scheduler import StepLR

def train_model(config, validate_with_images=False):
    # Initialize model, loss, optimizer, and scheduler
    device = get_device()
    if use_cnn:
        model = build_cnn(config).to(device)
    else:
        model = get_model(config).to(device)

    train_loader, test_loader = get_dataloaders(config)
    criterion = nn.CrossEntropyLoss()
    
    # Use AdamW optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=config['lr_start_val'], weight_decay=config['weight_decay'])
    
    # Cosine Annealing Learning Rate Scheduler
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=config['min_lr'])

    # ...existing code...

    for epoch in range(config['epochs']):
        # ...existing code...
        
        # Step the scheduler
        scheduler.step()

        # ...existing code...
```

With this change, the learning rate will decrease by a factor of `scheduler_gamma` every `scheduler_step_size` epochs. This should help the model converge more effectively.

We can also add in dynamic epochs, meaning that we can set a maximum number of epochs we want to run in our config file, then measure the loss at each point to see if we want to exit early due to a lack of progress. The code for this is pretty simple, we'll just add the following to our `train.py`:

```python
# ...existing code...

# Training loop
loss_history = [] # Keep track of loss values to see when we should stop
for epoch in range(config['epochs']):
    # ...existing code...
    
    # Step the scheduler
    scheduler.step()
    
    # Calculate average loss for the epoch
    average_loss = total_loss / len(train_loader)
    loss_history.append(average_loss)
    
    # Print epoch loss and learning rate
    print(f"Epoch {epoch+1}/{config['epochs']}, Loss: {total_loss/len(train_loader):.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
    
    # Run validation
    validate(model, test_loader, device, epoch+1, validate_with_images)
    
    # Check for early stopping
    if dynamic_epochs and epoch >= config['patience']:
        recent_losses = loss_history[-config['patience']:]
        if np.std(recent_losses) < config['std_threshold']:
            print(f"Stopping early at epoch {epoch+1} due to low loss standard deviation.")
            break
```

...and we'll add a few more parameters to our config while adjusting the present values:

```python
'epochs': 50, # An epoch is a full pass through the dataset
'patience': 5, # Minimum epochs to run before stopping early
'std_threshold': 1e-2, # Threshold for early stopping (difference in loss)
'lr_start_val': 1e-3, # Initial learning rate
'min_lr': 1e-6, # Minimum learning rate
'weight_decay': 1e-4, # Weight decay
'scheduler_step_size': 10, # Step size for the scheduler
'scheduler_gamma': 0.1, # Multiplicative factor of learning rate decay
'dropout_rate': 0.1, # Dropout rate (not sure how we forgot about this earlier)
```

Now we can set our epochs quite a bit higher without worrying of wasting resources.

### Attempt 5: Other Minor Tweaks for Model Accuracy

As we keep making bigger changes, we can also keep making some tweaks to our model or to our config. For example, it looks like my dataset normalization values are [a bit off](https://github.com/kuangliu/pytorch-cifar/issues/19), so adjusting those will work a bit better:

```python
# Data augmentation and normalization for training
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Only normalization for testing
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
```

We can also implement a better validation system by splitting the training data 90/10 so the validation process has different examples to validate. It seems like there are an endless amount of things to try and implement! Hyperparameters were also tweaked, as well as code logic here and there.

After all of these changes, we get an output that looks like this:

```
...
Epoch 37/50, Loss: 0.7187, LR: 0.000159
Validation - Predicted: ['bird', 'cat'], Actual: ['horse', 'cat'], Loss: 1.4525
Epoch 38/50, Loss: 0.7126, LR: 0.000136
Validation - Predicted: ['truck', 'deer'], Actual: ['truck', 'deer'], Loss: 0.6286
Epoch 39/50, Loss: 0.7007, LR: 0.000116
Validation - Predicted: ['airplane', 'ship'], Actual: ['airplane', 'ship'], Loss: 0.2973
Epoch 40/50, Loss: 0.6973, LR: 0.000096
Validation - Predicted: ['bird', 'truck'], Actual: ['bird', 'truck'], Loss: 0.2409
Epoch 41/50, Loss: 0.6944, LR: 0.000079
Validation - Predicted: ['deer', 'bird'], Actual: ['deer', 'bird'], Loss: 0.1391
Stopping early at epoch 41 due to low loss standard deviation.
Test Accuracy: 77.63%
```

77.63%—that's some good progress!

### Attempt 6: Final Implementations

We are looking pretty good with just above 75% accuracy on the model, but if we really want to make this useful, something above 80% would be nice. So what can we do? Well, I don't know. I know nothing about machine learning, but I've talked with ChatGPT quite a bit up until this point, so let's see if it has any guidance here. At this stage, we've already poured a lot of hours into this model that is just for learning purposes, so let's not spend too much time on these final features:

```md
(From ChatGPT)
If you want the biggest accuracy boost with the least effort, here are the top three strategies that will likely give you the most gains on CIFAR-10:

1. Data Augmentation (AutoAugment or CutMix) → ~3-5% Boost
	•	Why? Augmenting your dataset helps generalization without changing the model.
	•	How? Use torchvision.transforms.autoaugment.AutoAugment() or apply CutMix (replaces image patches) and MixUp (blends images).
	•	Expected Boost: 3-5% increase in accuracy.

2. Pretrained Weights from ImageNet → ~5-10% Boost
	•	Why? CIFAR-10 is small; a model pretrained on ImageNet learns general features.
	•	How? Fine-tune a ViT model with timm or torchvision.models.
	•	Expected Boost: 5-10% increase in accuracy.

3. Better Learning Rate & Regularization Tweaks → ~2-5% Boost
	•	Why? Adjusting learning rate and weight decay can stabilize training.
	•	How?
	•	Use AdamW (betas=(0.9, 0.95), weight_decay=0.05)
	•	Apply a cosine annealing learning rate schedule (torch.optim.lr_scheduler.CosineAnnealingLR)
	•	Expected Boost: 2-5% increase in accuracy.

If you just pretrain + augment without modifying architecture, you can push above 85% accuracy easily.
```

Ok those look like some good options that could be fairly quick. We'll start with the last suggestion since this is probably the quickest. We are going to change out weight_decay to 0.05 from 1e-4 in our config file to see if that makes a difference. We'll also add the betas to the AdamW model, which control how momentum and adaptive learning rate updates behave. In vision ViTs, it’s common to reduce Beta2 to 0.95 to make learning rate adaptation slightly more responsive, and to improve convergence on image datasets, as too much smoothing can slow adaptation. From further prompting, we also find out that Cosine Annealing usually benefits from a warmup phase, so ChatGPT recommends we use the LinearLR scheduler for warmup, then switch to CosineAnnealingLR (note: I know I am referencing ChatGPT a bit too much in this project, and that's ok—this is just a learning experience).

It also seems like our training model may have exited just a little too soon, since we were still making some improvements. We can update our config to have a smaller threshold or a larger patience.

Ok, the next implementation we can try is the data augmentation. Up to this point we've already finetuned our `dataset.py` pretty well, but we can try a few other methods. Here are the suggestions ChatGPT gave:

1. Stronger Random Cropping – RandomResizedCrop(32, scale=(0.8, 1.0)) can randomly zoom in slightly, making the model more robust.
2.	Cutout Regularization – RandomErasing() helps the model learn to handle occlusions.
3.	More Diverse Rotations & Jitter – Keeping the transformations but ensuring they are within optimal ranges.
4.	AutoAugment – Adds a learned augmentation policy (can be powerful).

These seem like edge-cases if we were really serious about making a robust model, but we can try them anyways. The changes here were very minor: the commented-out lines are the ones we replaced with the lines with comments at the end of them:

```python
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),  # Randomly crop & resize
    transforms.RandomHorizontalFlip(),
    #transforms.RandomCrop(32, padding=4),
    transforms.RandomRotation(20),  # Slightly increased rotation range
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.ToTensor(), # This had to be moved up before RandomErasing
    #transforms.RandomRotation(15),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),  # Cutout regularization
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
```

We also adjusted the ColorJitter values from 0.2 to 0.3 and lowered the hue from 0.2 to 0.1, since slightly stronger ColorJitter ensures lighting/color conditions are varied, meaning our model gets better training.

Before we get to our pretrained weights, it seems like we have our early stopping logic a bit off. Instead of basing it off of the model's loss, which could keep getting lower with overfitting, we will be better off doing it against our validation's loss. We will need to adjust some things in our validation code since right now we are only testing two entries, which will not give us reliable losses. Our validate method will stay basically the same (except it will return the loss to our training loop), but our run_validation method will instead go through the whole val_loader while only showing two of the values for direct feedback to the user. Other than that, everything there works the same, and we change our early stopping logic as follows:

```python
# Check for overfitting:
if len(loss_history_train) > 1 and len(loss_history_val) > 1:
    # Check if validation loss is increasing while training loss is still decreasing
    if loss_history_val[-1] > loss_history_val[-2] * 1.05 and loss_history_train[-1] < loss_history_train[-2]:
        print(f"Potential overfitting detected at epoch {epoch+1}. Validation loss increased significantly while training loss is still decreasing.")

# Check for early stopping
if dynamic_epochs and epoch >= config['patience']:
    recent_losses = loss_history_train[-config['patience']:]
    if np.std(recent_losses) < config['loss_threshold']:
        print(f"Stopping early at epoch {epoch+1} due to low loss standard deviation.")
        break
```

You'll notice I didn't implement anything of what I just said above. That's because I kept seeing it stop way to early in the training, same thing goes for potential overfitting. I just want to see how accurate I can get this model, so I'm scrapping it and just adding a warning for overfitting.

Ok, now what are these pretrained weights about? That's a big jump—5-10% it said—but it seems like it relies on other people's work. I say that like this is all my original work, but I digress. I'd like this model to be made just using established parts, not just find a different model entirely. Pretrained weights come from a model that has already been trained on a large dataset (often ImageNet, which contains millions of images). Instead of randomly initializing our neural network, we start with a model that already “knows” a lot about image features—like edges, textures, and object parts. This would definitely help and would be simple to implement, but it somewhat defeats the purpose of this project. We'll just have to hope that what we've done so far will be sufficient to get over 80%.

Another note before we get to the [end results](#end-results), it turns out there was one suggestion that ChatGPT had for the dataset.py file:
```python
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
```
Wow, this is a game changer. With `num_workers=4`, PyTorch will use 4 separate processes to load data in parallel, which can speed up data loading. With `pin_memory=True`, PyTorch will try to move the data to CUDA (GPU memory) faster by pre-pinning memory. This sped things up so much it's not even funny.

### Attempt 7: Saving the Model

Last step, let's save the model so we can use it elsewhere. Kind of sad to spend so much time training a model to have it disappear by the end of it. Initially, I had adjusted the code so you could pass in a parameter that would toggle whether it saves or not, and it would save the model at the end. Instead of this, it seemed better to save models throughout the training. Let's say epoch 41's model was actually just a bit better than epoch 42's, and the training stopped after 42 epochs. Now we're stuck with a worse model. Instead, we can save throughout and use conditional statements to make sure we always have the best model. Here's how that code looks in the loop:

```python
# Save the best model
if val_loss < best_val_loss:
    best_val_loss = val_loss
    torch.save(model.state_dict(), 'best_model.pth')
```

Then at the end of everything, we can still evaluating the model as follows:
```python
# Load the best model saved during training
model.load_state_dict(torch.load('best_model.pth'))

# Evaluate the best model
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy of the best model: {100 * correct / total:.2f}%")
```

## End Results

So what was the end result of all of this trial and error trying to implement a ViT model? We started from a good outline that ChatGPT provided for us, that we modified to be separated out into multiple files, and which gave us an astounding 10% accuracy. From there, we found out some things that were wrong with our model by first implementing a CNN model, and we were slowly able to finetune things from there. In the final run of this model, with all of the code being the same as the commit which includes everything up to attempt 7, the results can be found [here](resources/improvedModelResults.txt).

Unfortunately, our hyperparameters were probably off, and our accuracy only improved to just over 80%, so our extra time and effort to try and improve the model may have been a bit wasted. All in the learning process! And there's more than we can try and improve if we wanted to keep working. You'll notice that there were several times that we got the `"Potential overfitting detected at epoch 21. Validation loss increased significantly while training loss is still decreasing."` message. The reason we didn't have the model stop is due to those being anomalies more than consistent reads. If we saw two or three of those messages in a row—one epoch after another—that's when we might need to consider stopping the training. Still, that's just my guess and I could be wrong; my model could definitely be overfit right now.

In the end, with 100 epochs, we got our model up to 80% efficiency, which I am happy with! The only sad thing? You know that CNN model we implemented? I kept it and it's getting 85%...

## Contributing

Feel free to clone or fork the repository if you'd like to implement your own solutions and modifications.
