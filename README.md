# Accident Detector
## Overview
This project aims to train image detection model that is capable of detecting road accidents in real time and send emergency messages containig current location of device
to all helplines.
For demonstration purpose this model accepts a video from user and the model analyses the video and detects accidents (if its present in the video) and 
with the help of geopy the python module finds the complete address(along with longitude and latitude) and sends message with vonage api.

## Installation
To run this project locally, follow these steps:
1. Clone the repository: bash git clone https://github.com/Shobhit043/AccidentDetector.git
2. Install the required dependencies: bash pip install -r requirements.txt

## Dataset
link to the dataset : https://www.kaggle.com/datasets/ckay16/accident-detection-from-cctv-footage .
This dataset contains around 900 images in total distributed in 3 directories train,test and val.

# Data Augmentation
This project implements a data augmentation pipeline using TensorFlow to enhance dataset variability. The augmentations include random flipping, rotation, and zooming, which help improve model robustness by introducing diverse transformations to the input images.


## Neural Network Architecture

### MobileNetV2 as Base Model

We utilize MobileNetV2, a lightweight convolutional neural network designed for mobile and edge devices, as the foundation of our model. MobileNetV2 is known for its efficiency and performance, making it suitable for real-time applications on devices with limited computational resources.

### Additional Convolutional Layers

Following MobileNetV2, we have added additional convolutional layers to further refine the features learned by the base model. The architecture includes the following convolutional layers:

1. Conv2D Layer (32 filters, kernel size 3x3, activation function: Sigmoid)
2. Conv2D Layer (64 filters, kernel size 3x3, activation function: Sigmoid)
3. Conv2D Layer (128 filters, kernel size 3x3, activation function: Sigmoid)

These layers are designed to capture more complex patterns and hierarchies in the data, enabling the model to learn representations that are specific to our particular task.

![Screenshot (457)](https://github.com/Shobhit043/Accident_Detection/blob/main/Readme_images/model_architecture.png)

# Early Stopping
This project utilizes an EarlyStopping callback in TensorFlow to optimize training efficiency. The callback monitors validation accuracy and stops training if no improvement is detected beyond a specified threshold (min_delta=0.01) for 15 consecutive epochs. It helps prevent overfitting and reduces unnecessary computation by restoring the best model weights.


## Model Performance
### Training accuracy = 83.55% 
### Validation accuracy = 78.95% <br><br>

![Screenshot (456)](https://github.com/Shobhit043/Accident_Detection/blob/main/Readme_images/Training_plots.png)


## Geopy
Geopy is a Python library that provides a simple interface to various geocoding services, allowing users to convert addresses into geographic coordinates and vice versa. It supports multiple providers, such as OpenStreetMap Nominatim, Google Maps, and more. Geopy facilitates location-based applications and services by offering geocoding, reverse geocoding, and distance calculation functionalities.
I used it find the current address along with longitude and latitude that is being sent ove the sms.

## Vonage
Vonage offers a free SMS API that enables developers to integrate SMS capabilities into their applications. With Vonage SMS API, you can send text messages globally, track message delivery status, and receive SMS replies. It's a reliable solution for adding communication features to your applications without the need for a significant upfront cost.
I used it to send message to helplines(though in this project i have only set my own phone number)
#### Note: all credential are not present in the code and you must create your own vonage account the send sms succesfully, temp folder is necessary to temporarily store the input video and it also contain Accident-1.mp4 in case you want to test the project

## Sample Accident Video

https://github.com/Shobhit043/Accident_Detection/blob/main/sample_accident_video_for_input/accident_video.mp4

## SMS MESSAGE

![Screenshot (459)](https://github.com/Shobhit043/Accident_Detection/blob/main/Readme_images/sample_sms_message.png)



## License
This project is licensed under the MIT License - see the LICENSE file for details
