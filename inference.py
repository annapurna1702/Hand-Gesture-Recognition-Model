import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np


classes = ['palm', 'l', 'fist', 'fist_moved', 'thumb', 'index', 'ok', 'palm_moved', 'c', 'down']
class_to_idx = {name: i for i, name in enumerate(classes)}
idx_to_class = {i: name for name, i in class_to_idx.items()}


class SimpleCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(64 * 37 * 37, 128)
        self.fc2 = torch.nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool2(torch.nn.functional.relu(self.conv2(x)))
        x = self.flatten(x)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


num_classes = len(classes)
model = SimpleCNN(num_classes=num_classes)
model.load_state_dict(torch.load('image_classifier.pth'))
model.eval()


def preprocess_image(image_path, target_size=(150, 150)):
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)
    return image


def classify_image(image_path):
    img = preprocess_image(image_path)
    with torch.no_grad():
        predictions = model(img)
        predicted_class_idx = torch.argmax(predictions, axis=1).item()
        predicted_class = idx_to_class[predicted_class_idx]
    return predicted_class


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_predictions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = preprocess_image(img)
        with torch.no_grad():
            predictions = model(img)
            predicted_class_idx = torch.argmax(predictions, axis=1).item()
            frame_predictions.append(predicted_class_idx)

    cap.release()
    
    final_prediction_idx = max(set(frame_predictions), key=frame_predictions.count)
    final_prediction = idx_to_class[final_prediction_idx]
    return final_prediction


def main(input_path):
    if input_path.endswith(('.mp4', '.avi', '.mov')):
        result = process_video(input_path)
        print(f"Predicted Class for Video: {result}")
    else:
        result = classify_image(input_path)
        print(f"Predicted Class for Image: {result}")


if __name__ == '__main__':
    input_path = 'ok.png' 
    main(input_path)
