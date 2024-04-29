import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tkinter import *
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import time

# Define the CNN model architecture
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

# Load the saved model
model = CNN()
model_path = 'D:/CVIP/Project/fruit_insight.pth'
model.load_state_dict(torch.load(model_path))
model.eval()  # Set the model to evaluation mode

# Function to retrieve screen resolution
def get_screen_resolution():
    root = Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()
    return screen_width, screen_height

# Function to convert OpenCV image to PhotoImage
def convert_image_to_photo(image):
    if image is not None:
        # Convert OpenCV image to RGB format
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize the image to fit the frame
        resized_image = cv2.resize(rgb_image, (550, 450))
        
        # Convert PIL image to PhotoImage
        photo = ImageTk.PhotoImage(Image.fromarray(resized_image))
        return photo
    else:
        return None

# Function to set the initial frame of the image
def initialize_image_frame(image):
    if image is not None:
        # Convert the image to PhotoImage
        photo = convert_image_to_photo(image)
        
        # Display the image in the label
        if photo is not None:
            image_frame_label.config(image=photo)
            image_frame_label.image = photo  # Keep a reference to prevent garbage collection
            return True
    return False

def resize_image(image_path, new_width, new_height):
    image = Image.open(image_path)
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return ImageTk.PhotoImage(resized_image)

# Function to capture live camera feed
def capture_camera_feed():
    cap = cv2.imageCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open camera")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        # Display the captured frame in a window
        cv2.imshow("Camera Feed", frame)

        # Classify the captured image and update UI
        classify_image_and_update_ui(frame)

        # Check for 'q' key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to handle the browse button click event
def on_browse_clicked():
    selected_image_path = filedialog.askopenfilename(
        title="Select Image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.jfif")]
    )
    image = cv2.imread(selected_image_path)

    if initialize_image_frame(image):
        image_name_label.config(text=selected_image_path)
         # Classify the captured image and update UI
        classify_image_and_update_ui(image)
        print("Image file has been opened")
    else:
        print("Image file not opened!")


# Initializing the root window
root_window = Tk()
root_window.configure(bg="sea green")
screen_width, screen_height = get_screen_resolution()
root_window.geometry(f"{screen_width}x{screen_height}")
root_window.columnconfigure(0, weight=1)

# Container for image name and browse Button
frame_header = Frame(root_window, padx=5, pady=5, bd=2)
frame_header.grid(row=0, column=0, pady=20)

# Label that displays image name
image_name_frame = Frame(frame_header, padx=5, pady=20, bd=10, relief=SOLID, bg="yellow")
image_name_frame.grid(row=0, column=0, columnspan=2)
image_name_label = Label(image_name_frame, text="Fruit Insight", width=60, font=("Times New Roman", 18))
image_name_label.pack()

# Label for displaying image file name
image_name_label = Label(frame_header, text="Image File Name", width=50, bg="light green", font=("Times New Roman", 12))
image_name_label.grid(row=1, column=0, padx=5, pady=5)

# Browse Button
browse_button = Button(frame_header, text="Browse Files",pady=2, bd=2, command=on_browse_clicked, bg="light green", font=("Times New Roman", 12), width=15)
browse_button.grid(row=1, column=1, padx=120)

# Label for displaying the result
result_label = Label(frame_header, text="Banana State: ", font=("Times New Roman", 18), bg="orange")
result_label.grid(row=2, column=0, columnspan=2, pady=10)

# Button to start camera feed
camera_button = Button(frame_header, text="Start Camera Feed", command=capture_camera_feed, font=("Times New Roman", 12), width=15, bg="light green")
camera_button.grid(row=3, column=0, columnspan=2, pady=10)

# image displayer
image_frame = Frame(root_window, height=450, width=550, padx=5, pady=5, bd=2, relief=SOLID)
image_frame.grid(row=1, column=0)
image_frame.pack_propagate(False)
image_frame_label = Label(image_frame)
image_frame_label.pack()

# image Options Panel
image_option_frame = Frame(root_window, padx=5, pady=5, bd=2)
image_option_frame.grid(row=2, column=0)

# Function to preprocess a single image
def preprocess_image(image, target_size=(100, 100)):
    img_resized = cv2.resize(image, target_size)
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    img_normalized = img_gray / 255.0
    img_float32 = np.float32(img_normalized)
    img_tensor = torch.tensor(img_float32).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    return img_tensor

# Function to classify the image and update UI
def classify_image_and_update_ui(image):
    # Preprocess the image
    img_tensor = preprocess_image(image)

    # Make prediction
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.sigmoid(output)
        predicted_class = torch.round(probabilities).item()

    # Get the predicted label
    predicted_label = "Fresh" if predicted_class == 1 else "Rotten"

    # Update UI with prediction
    result_label.config(text=f"Banana State: {predicted_label}")

    # Add a delay to see the result on the UI
    root_window.update()  # Update the UI to ensure the label is displayed
    time.sleep(0.1)  # Adjust the delay time as needed

# Running Program
root_window.mainloop()
