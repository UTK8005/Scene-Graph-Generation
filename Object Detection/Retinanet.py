import os
import torch
import torchvision.transforms as transforms
from torchvision.models.detection import retinanet_resnet50_fpn
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Load a pre-trained RetinaNet model
model = retinanet_resnet50_fpn(pretrained=True)
model.eval()

# Define class labels (modify according to your dataset)
class_labels = ["class1", "class2", ...]

# Define a preprocessing function for your input image
def preprocess_image(img_path, img_size):
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor

# Specify the path to the folder containing your dataset images
dataset_folder = "path_to_dataset_folder"

# Set confidence threshold
confidence_threshold = 0.5

# Loop through all images in the folder, perform inference, and save annotated images
for filename in os.listdir(dataset_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Add more extensions as needed
        input_image_path = os.path.join(dataset_folder, filename)
        img_size = 800  # Adjust this to match the input size of your RetinaNet model

        # Preprocess the input image
        img_tensor = preprocess_image(input_image_path, img_size)

        # Perform inference
        with torch.no_grad():
            output = model(img_tensor)

        # Post-process the model's output to get detections
        def post_process(output, confidence_threshold=0.5):
            boxes = output[0]['boxes']
            labels = output[0]['labels']
            scores = output[0]['scores']

            # Filter out detections with confidence scores below the threshold
            mask = scores >= confidence_threshold
            boxes = boxes[mask]
            labels = labels[mask]
            scores = scores[mask]

            return boxes, labels, scores

        det_boxes, det_labels, det_scores = post_process(output, confidence_threshold)

        # Load the original image
        img = Image.open(input_image_path)

        # Create a copy of the image for annotation
        annotated_img = img.copy()
        draw = ImageDraw.Draw(annotated_img)

        # Visualize the detections on the image
        for box, label, score in zip(det_boxes, det_labels, det_scores):
            x, y, x_max, y_max = box
            width = x_max - x
            height = y_max - y

            # Create a Rectangle patch
            rect = plt.Rectangle((x, y), width, height, linewidth=2, edgecolor='red', facecolor='none')
            draw.rectangle([x, y, x_max, y_max], outline="red")

            # Display class label and confidence score
            label_name = class_labels[label - 1]
            draw.text((x, y - 10), f"{label_name}: {score:.2f}", fill="white")

        # Save or display the annotated image
        annotated_img.save(f"annotated_{filename}")  # You can save the annotated image with a new name

