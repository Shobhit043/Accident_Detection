from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import numpy as np
import tensorflow as tf
import cv2
from vonage import Auth, Vonage
from vonage_sms import SmsMessage, SmsResponse
from geopy.geocoders import Nominatim
from credentials import key, secret, phone_num

app = Flask(__name__)

# Set upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to preprocess and predict an image frame
def predict(frame):
    # Resize frame to 224x224
    resized_frame = cv2.resize(frame, (224, 224))
    
    # Normalize the frame (assuming the model expects values between 0 and 1)
    img_array = resized_frame.astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]["index"], img_batch)
    interpreter.invoke()

    # Get predictions
    tflite_model_predictions = interpreter.get_tensor(output_details[0]["index"])
    
    # Return the predicted class
    return np.argmax(tflite_model_predictions)

# Initialize geolocator
geolocator = Nominatim(user_agent="accident_detection_app")
location = geolocator.geocode("Jawaharlal Nehru University")
loc = f"\n Address: {location.address} \n Latitude: {location.latitude} \n Longitude: {location.longitude}"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/", methods=["POST"])
def predict_func():
    if "videoFile" not in request.files:
        return render_template("result.html", message="No video file uploaded.")

    video_file = request.files["videoFile"]

    # Validate file extension
    allowed_extensions = {"mp4", "avi", "mov", "mkv"}
    if video_file.filename.split(".")[-1].lower() not in allowed_extensions:
        return render_template("result.html", message="Invalid video file format.")

    # Save video file
    video_filename = secure_filename(video_file.filename)
    video_path = os.path.join(app.config["UPLOAD_FOLDER"], video_filename)
    video_file.save(video_path)

    # Process video for predictions
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    accident_detected = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Stop if no more frames

        # Process every 10th frame
        pred = predict(frame)

        if pred == 0:  # If accident detected
            accident_detected = True
            break

        frame_count += 1

    cap.release()

    if accident_detected:
        status = "🚨 Accident detected! 🚨"
        
        # Send SMS alert
        client = Vonage(Auth(api_key=key, api_secret=secret))
                        
        message = SmsMessage(
            to=phone_num,
            from_="Accident Alert",
            text= status + loc
        )

        # Uncomment to send message
        # response: SmsResponse = client.sms.send(message)
        # print(response)


    else:
        status = "✅ No accident detected."

    return render_template("result.html", status=status, loc=loc)

if __name__ == "__main__":
    app.run(debug=True)
