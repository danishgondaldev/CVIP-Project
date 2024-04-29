import os
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np

# Function to load, resize, convert to grayscale, and normalize images from a directory
def load_preprocess_images_from_dir(directory, target_size=(100, 100)):
    # this list of images will  be returned by this function
    images = []
    for filename in os.listdir(directory):
        image_path = os.path.join(directory, filename)
        if os.path.isfile(image_path):
            try:
                # Load image   
                img = cv2.imread(image_path)
                # Resize image to target size
                img_resized = cv2.resize(img, target_size)
                # Convert image to grayscale
                img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
                # Normalize image between 0 and 1
                img_normalized = img_gray / 255.0
                 # Convert image to np.float32
                img_float32 = np.float32(img_normalized)
                images.append(img_float32)
            except Exception as e:
                print(f"Error loading image '{filename}': {e}")
    return images



# Directories with training images
train_freshbanana_dir = r'D:\CVIP\Project\archive\dataset\Train\freshbanana'
train_rottenbanana_dir = r'D:\CVIP\Project\archive\dataset\Train\rottenbanana'

# Load and resize training images
train_freshbanana_images = load_preprocess_images_from_dir(train_freshbanana_dir)
train_rottenbanana_images = load_preprocess_images_from_dir(train_rottenbanana_dir)


# Directories with test images
test_freshbanana_dir = r'D:\CVIP\Project\archive\dataset\Test\freshbanana'
test_rottenbanana_dir = r'D:\CVIP\Project\archive\dataset\Test\rottenbanana'

# Load and resize test images
test_freshbanana_images = load_preprocess_images_from_dir(test_freshbanana_dir)
test_rottenbanana_images = load_preprocess_images_from_dir(test_rottenbanana_dir)

# Define transforms to apply to images
# h x w ->  1 x h x w
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
])

# Convert images to PyTorch tensors and apply transformations
train_freshbanana_tensors = [transform(image) for image in train_freshbanana_images]
train_rottenbanana_tensors = [transform(image) for image in train_rottenbanana_images]
test_freshbanana_tensors = [transform(image) for image in test_freshbanana_images]
test_rottenbanana_tensors = [transform(image) for image in test_rottenbanana_images]

print("Number of training fresh banana images:", len(train_freshbanana_tensors))
print("Number of training rotten banana images:", len(train_rottenbanana_tensors))
print("Number of test fresh banana images:", len(test_freshbanana_tensors))
print("Number of test rotten banana images:", len(test_rottenbanana_tensors))

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) 
        
        # Additional convolutional layers
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


# Create an instance of the CNN model
model = CNN()

# Print model summary
print(model)

# Convert labels to tensors
train_labels = torch.tensor([1] * len(train_freshbanana_tensors) + [0] * len(train_rottenbanana_tensors))
test_labels = torch.tensor([1] * len(test_freshbanana_tensors) + [0] * len(test_rottenbanana_tensors))

# Combine fresh and rotten banana images and labels
train_images = train_freshbanana_tensors + train_rottenbanana_tensors
test_images = test_freshbanana_tensors + test_rottenbanana_tensors

# Create datasets
train_dataset = torch.utils.data.TensorDataset(torch.stack(train_images), train_labels)
test_dataset = torch.utils.data.TensorDataset(torch.stack(test_images), test_labels)

# Define batch size
batch_size = 32

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

# Training loop
num_epochs = 15
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        # Convert inputs to float
        inputs = inputs.float()
        outputs = model(inputs)
        loss = criterion(outputs, labels.float().view(-1, 1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")


# Test the model
model.eval()
correct = 0
total = 0
#no gradients
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        predicted = torch.round(outputs)
        total += labels.size(0)
        # item converts tensor to scalar
        correct += (predicted == labels.float().view(-1, 1)).sum().item()

accuracy = correct / total
print(f"Accuracy on test set: {accuracy * 100:.2f}%")
# Save the model
model_path = 'D:/CVIP/Project/fruit_insight.pth'
torch.save(model.state_dict(), model_path)
print(f"Model saved at: {model_path}")

# Check if the file exists
if os.path.exists(model_path):
    print("Model file exists.")
else:
    print("Model file does not exist. There might be an issue with saving the model.")