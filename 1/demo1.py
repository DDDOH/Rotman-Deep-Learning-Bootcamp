# import packages
import torch
from torchvision import datasets, transforms

import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

import torch.nn.functional as F

# Define the transformation to normalize the data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load the MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# print the number of images in train and test dataset
print(f'Number of images in train dataset: {len(train_dataset)}')
print(f'Number of images in test dataset: {len(test_dataset)}')


# Configs
MODE = 'softmax_mlp'   # 'linear' or 'softmax' or 'softmax_mlp'
input_size = 28 * 28  # MNIST images are 28x28 pixels

########### Visualize the data
# Get a batch of training data
dataiter = iter(train_loader)
images, labels = next(dataiter)

# Function to show an image
def imshow(img, ax):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    ax.imshow(np.squeeze(npimg), cmap="gray")  # Fix for grayscale images

# Plot 9 images in a 3x3 grid
fig, axes = plt.subplots(3, 3, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    imshow(images[i], ax)  # Pass ax to imshow
    ax.set_title(f'Label: {labels[i].item()}')
    ax.axis('off')

plt.tight_layout()
plt.savefig('mnist.png')
plt.close()

########### Logger class
class Logger:
    def __init__(self, mode):
        self.mode = mode
        self.train_loss = {'value': [], 'epoch': []}
        self.test_loss = {'value': [], 'epoch': []}
        self.train_accuracy = {'value': [], 'epoch': []}
        self.test_accuracy = {'value': [], 'epoch': []}
        
    def log_train_loss(self, epoch, loss):
        self.train_loss['value'].append(loss)
        self.train_loss['epoch'].append(epoch)
        
    def log_test_loss(self, epoch, loss):
        self.test_loss['value'].append(loss)
        self.test_loss['epoch'].append(epoch) 
        
    def log_train_accuracy(self, epoch, accuracy):
        self.train_accuracy['value'].append(accuracy)
        self.train_accuracy['epoch'].append(epoch)
        
    def log_test_accuracy(self, epoch, accuracy):
        self.test_accuracy['value'].append(accuracy)
        self.test_accuracy['epoch'].append(epoch)
        
    def plot(self):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.train_loss['epoch'], self.train_loss['value'], label='Train')
        plt.plot(self.test_loss['epoch'], self.test_loss['value'], label='Test')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracy['epoch'], self.train_accuracy['value'], label='Train')
        plt.plot(self.test_accuracy['epoch'], self.test_accuracy['value'], label='Test')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'log_{self.mode}.png')
        



def test(epoch, mode):
    """
    Test the model on the test dataset.

    This method will plot 9 images from the test dataset and display the model's prediction for each image, evaluate and return the model's accuracy on the test dataset.
    
    Suitable for both linear regression and softmax regression models.

    """
    # Select one image from the dataset
    # Get a batch of test data
    dataiter = iter(test_loader)
    images, labels = next(dataiter)

    # Plot 9 images in a 3x3 grid
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        imshow(images[i], ax)  # Pass ax to imshow
        ax.set_title(f'Real: {labels[i].item()}')
        ax.axis('off')

        # Prepare the image for the model
        image = images[i].view(-1, 28*28)  # Flatten the image
        label = labels[i].view(-1, 1)

        # Get the model output for this image
        model.eval()
        with torch.no_grad():
            # output = model(image)
            predicted, loss = model.predict(image, label)
            ax.set_title(f'Real: {labels[i].item()}, Pred: {predicted.item()}')

    plt.tight_layout()
    plt.savefig('output_{}_{}.png'.format(mode, epoch))
    plt.close()
    
    
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        loss = 0
        for images, labels in test_loader:
            images = images.view(-1, 28*28)
            predicted, loss_batch = model.predict(images, labels)
            total += labels.size(0)
            correct += (predicted == labels.view(-1,1)).sum().item()
            loss += loss_batch * labels.size(0)
        print(f'Accuracy of the model on the 10000 test images: {100 * correct / total} %')
        
    return correct / total, loss / total



######### Linear regression model
if MODE == 'linear':
    
    class LinearRegression(nn.Module):
        def __init__(self, input_size, output_size):
            super(LinearRegression, self).__init__()
            # self.linear = nn.Linear(input_size, 1)
            self.a = nn.Parameter(torch.randn(input_size, output_size))
            self.b = nn.Parameter(torch.randn(output_size))
                
        def forward(self, x):
            out = torch.matmul(x, self.a) + self.b
            return out
        
        def predict(self, x, labels):
            # for linear regression model, the output is y directly. The predicted label is simply the rounded value of y.
            out = self.forward(x)
            loss = square_error(out, labels)
            return torch.round(out), loss

        
        
    model = LinearRegression(input_size, 1)

    # Loss function
    def square_error(outputs, labels):
        # outputs: (batch_size, 1)
        # labels: (batch_size, 1)
        loss = ((outputs - labels) ** 2).mean()
        return loss
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    
    logger = Logger(MODE)
    
    
    # Train the model
    num_epochs = 100
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images = images.view(-1, 28*28)  # Flatten the images
            
            # Forward pass
            outputs = model(images)
            loss = square_error(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
        logger.log_train_loss(epoch, loss.item())
        
        
        if epoch % 10 == 0:
            accuracy, loss = test(epoch, MODE)
            logger.log_test_accuracy(epoch, accuracy)
            logger.log_test_loss(epoch, loss)
            logger.plot()


####################
if MODE in ['softmax', 'softmax_mlp']:
    # now use softmax regression
    num_classes = 10  # Digits 0-9
    
    class SoftmaxLinear(nn.Module):
        def __init__(self, input_size, output_size):
            super(SoftmaxLinear, self).__init__()
            # self.linear = nn.Linear(input_size, 1)
            self.a = nn.Parameter(torch.randn(input_size, output_size))
            self.b = nn.Parameter(torch.randn(output_size))
                
        def forward(self, x):
            out = torch.matmul(x, self.a) + self.b
            return out
        
        def predict(self, x, labels):
            # for softmax regression model, the output is logits. The predicted label is the argmax of the logits.
            out = self.forward(x)
            loss = - softmax_likelihood(out, labels)
            return torch.argmax(out, dim=1, keepdim=True), loss
        
        
    class SoftmaxMLP(nn.Module):
        def __init__(self, input_size, output_size):
            super(SoftmaxMLP, self).__init__()
            self.fc1 = nn.Linear(input_size, 128)
            self.fc2 = nn.Linear(128, 128)
            self.fc3 = nn.Linear(128, output_size)
            
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
        
        def predict(self, x, labels):
            out = self.forward(x)
            loss = - softmax_likelihood(out, labels)
            return torch.argmax(out, dim=1, keepdim=True), loss
            
        
    if MODE == 'softmax':
        model = SoftmaxLinear(input_size, num_classes)
    elif MODE == 'softmax_mlp':
        model = SoftmaxMLP(input_size, num_classes)
        
    # Compute the loss manually
    def softmax_likelihood(logits, labels):
        # # outputs: (batch_size, num_class)
        # # labels: (batch_size)
        # Apply log-softmax to logits directly (numerically stable)
        log_prob = F.log_softmax(logits, dim=1)  # Directly compute log probabilities

        # Convert labels to one-hot encoding
        num_classes = logits.size(1)
        labels_one_hot = torch.zeros(labels.size(0), num_classes, device=logits.device)
        labels_one_hot.scatter_(1, labels.view(-1, 1), 1)

        # Compute negative log-likelihood
        log_likelihood = torch.sum(labels_one_hot * log_prob, dim=1)
        log_likelihood = log_likelihood.mean()
        
        return log_likelihood
        
    
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # Train the model
    logger = Logger(MODE)
    num_epochs = 100
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images = images.view(-1, 28*28)  # Flatten the images
            
            # Forward pass
            outputs = model(images)
            loss = - softmax_likelihood(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        logger.log_train_loss(epoch, loss.item())
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        if epoch % 10 == 0:
            accuracy, loss = test(epoch, MODE)
            logger.log_test_accuracy(epoch, accuracy)
            logger.log_test_loss(epoch, loss)
            logger.plot()