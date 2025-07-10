import os
import cv2
import numpy as np
from ultralytics import YOLO
import yaml
import torch




# Define dataset path
train_path = r"D:\fyp\fypcode\dataset\train"    #  Update to your dataset train path
val_path = r"D:\fyp\fypcode\dataset\val"        #  Update to your dataset validation path

# Define Class Names
classes = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# Function to load images and labels
def load_data(data_path):
    images = []
    labels = []

    for class_label in classes:
        class_path = os.path.join(data_path, class_label, 'images')     # Directory containing images for this class.
        label_path = os.path.join(data_path, class_label, 'labels')     # Directory containing labels for this class.

        for img_file in os.listdir(class_path):
            img = cv2.imread(os.path.join(class_path, img_file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                  # read images and convert in from BGR to RGB

            label_file = img_file.replace('.jpg', '.txt')   # Replaces the .jpg file extension with .txt
            label_file_path = os.path.join(label_path, label_file)      # constructs the absolute path to the label file.

            # Check if the Label File Exists and read the label file
            if os.path.exists(label_file_path):
                with open(label_file_path, 'r') as file2:
                    label_data = file2.readline().strip().split()

                    # Append the images and label to the list if label_data is not empty
                    if len(label_data) > 0:
                        images.append(img)
                        labels.append(label_data)
                    else:
                        print(f"Label file {label_file_path} is empty, skipping this image.")
            else:
                print(f"Label file {label_file_path} not found, skipping this image.")
    return images, labels

# Define function to preprocess Images
def preprocess_images(images):
    processed_images = []
    for img in images:
        img_resized = cv2.resize(img, (640, 640))
        processed_images.append(img_resized)
    return np.array(processed_images)

# Creates a dataset.yaml file
dataset_yaml = {
    'path': r'D:\fyp\fypcode\dataset',  # update your dataset path
    'train': 'train',
    'val': 'val',
    'names':  classes
}
with open(r'D:\fyp\fypcode\dataset.yaml', 'w') as file:  # # update your dataset.yaml path
    yaml.dump(dataset_yaml, file)


'''
 Main Function Execution
'''
if __name__ == '__main__':
    #print(torch.cuda.is_available())
    train_images, train_labels = load_data(train_path)
    val_images, val_labels = load_data(val_path)
    train_images = preprocess_images(train_images)
    val_images = preprocess_images(val_images)

    # Load YOLOv10n model
    model = YOLO("yolov10n_CBAM.yaml")  # update your model path
    #model = YOLO("yolov9t.yaml")

    # Train the model
    #print(torch.cuda.is_available())
    #print(torch.cuda.get_device_name(0))
    result = model.train(data=r'D:\fyp\fypcode\dataset.yaml',  # update your dataset.yaml path
                         epochs=10,  # Increase training to 1000
                         imgsz=640,     # Image size
                         lr0=0.001,  # Initial learning rate
                         lrf=0.2,  # Final learning rate multiplier
                         mosaic=True,  # Enable mosaic augmentation
                         mixup=True,  # Enable mixup augmentation
                         hsv_h=0.015,  # Adjust hue
                         hsv_s=0.7,  # Adjust saturation
                         hsv_v=0.4,  # Adjust value (brightness)
                         batch=16,  # Batch size (adjust based on GPU capacity)
                         workers=4,  # Number of data loader workers
                         deterministic = False,
                         )

    model.save(r'D:\fyp\fypcode\FYPv10CM_model.pt')























