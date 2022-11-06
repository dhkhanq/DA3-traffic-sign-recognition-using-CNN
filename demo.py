import numpy as np
import cv2 as cv
import pickle
import tensorflow as tf

# Camera resolution
frame_width = 640  
frame_height = 480
brightness = 180
threshold = 0.80 # Probablity threshold
font = cv.FONT_HERSHEY_SIMPLEX

# Setup the video camera
cap = cv.VideoCapture(0)
cap.set(3, frame_width)
cap.set(4, frame_height)
cap.set(10, brightness)

# Import the trained model
# pickle_in = open('models/ngocnet.h5', 'rb') # rb = READ_BYTE
# model = pickle.load(pickle_in)
model = tf.keras.models.load_model("da3.h5")

def grayscale(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv.equalizeHist(img)
    return img

def preprocessing(img):
    # img = grayscale(img)
    # img = equalize(img)
    img = img/255.0
    return img

classNames = {0: 'Speed limit (20km/h)',
 1: 'Speed limit (30km/h)',
 2: 'Speed limit (50km/h)',
 3: 'Speed limit (60km/h)',
 4: 'Speed limit (70km/h)',
 5: 'Speed limit (80km/h)',
 6: 'End of speed limit (80km/h)',
 7: 'Speed limit (100km/h)',
 8: 'Speed limit (120km/h)',
 9: 'No passing',
 10: 'No passing for vehicles over 3.5 metric tons',
 11: 'Right-of-way at the next intersection',
 12: 'Priority road',
 13: 'Yield',
 14: 'Stop',
 15: 'No vehicles',
 16: 'Vehicles over 3.5 metric tons prohibited',
 17: 'No entry',
 18: 'General caution',
 19: 'Dangerous curve to the left',
 20: 'Dangerous curve to the right',
 21: 'Double curve',
 22: 'Bumpy road',
 23: 'Slippery road',
 24: 'Road narrows on the right',
 25: 'Road work',
 26: 'Traffic signals',
 27: 'Pedestrians',
 28: 'Children crossing',
 29: 'Bicycles crossing',
 30: 'Beware of ice/snow',
 31: 'Wild animals crossing',
 32: 'End of all speed and passing limits',
 33: 'Turn right ahead',
 34: 'Turn left ahead',
 35: 'Ahead only',
 36: 'Go straight or right',
 37: 'Go straight or left',
 38: 'Keep right',
 39: 'Keep left',
 40: 'Roundabout mandatory',
 41: 'End of no passing',
 42: 'End of no passing by vehicles over 3.5 metric tons'}

def get_class_name(class_no):
    if class_no == 0: return "Speed Limit 20 km/h"
    elif class_no == 1: return "Speed Limit 30 km/h"
    elif class_no == 2: return "Speed Limit 50 km/h"
    elif class_no == 3: return "Speed Limit 60 km/h"
    elif class_no == 4: return "Speed Limit 70 km/h"
    elif class_no == 5: return "Speed Limit 80 km/h"
    elif class_no == 6: return "End of Speed Limit 80 km/h"
    elif class_no == 7: return "Speed Limit 100 km/h"
    elif class_no == 8: return "Speed Limit 120 km/h"
    elif class_no == 9: return "No Passing"
    elif class_no == 10: return "No passing for vechiles over 3.5 metric tons"
    elif class_no == 11: return "Right-of-way at the next intersection"
    elif class_no == 12: return "Priority road"
    elif class_no == 13: return "Yield"
    elif class_no == 14: return "Stop"
    elif class_no == 15: return "No vehicles"
    elif class_no == 16: return "Vehicles over 3.5 metric tons prohibited"
    elif class_no == 17: return "No entry"
    elif class_no == 18: return "General caution"
    elif class_no == 19: return "Dangerous curve to the left"
    elif class_no == 20: return "Dangerous curve to the right"
    elif class_no == 21: return "Double curve"
    elif class_no == 22: return "Bumpy road"
    elif class_no == 23: return "Slippery road"
    elif class_no == 24: return "Road narrows on the right"
    elif class_no == 25: return "Road work"
    elif class_no == 26: return "Traffic signals"
    elif class_no == 27: return "Pedestrians"
    elif class_no == 28: return "Children crossing"
    elif class_no == 29: return "Bicycles crossing"
    elif class_no == 30: return "Beware of ice/snow"
    elif class_no == 31: return "Wild animals crossing"
    elif class_no == 32: return "End of all speed and passing limits"
    elif class_no == 33: return "Turn right ahead"
    elif class_no == 34: return "Turn left ahead"
    elif class_no == 35: return "Ahead only"
    elif class_no == 36: return "Go straight or right"
    elif class_no == 37: return "Go straight or left"
    elif class_no == 38: return "Keep right"
    elif class_no == 39: return "Keep left"
    elif class_no == 40: return "Roundabout mandatory"
    elif class_no == 41: return "End of no passing"
    elif class_no == 42: return "End of no passing by vechiles over 3.5 metric tons"

while True:

    # Read image
    success, original_img = cap.read()

    # Process image
    img = np.asarray(original_img)
    img = cv.resize(img, (32, 32))
    img = preprocessing(img)
    cv.imshow("Processed Image", img)
    img = img.reshape(1, 32, 32, 3)
    cv.putText(original_img, "CLASS: ", (20, 35), font, 0.75, (0, 0, 255), 2, cv.LINE_AA)
    cv.putText(original_img, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv.LINE_AA)

    # Predict image
    # result = model.predict()
    predictions = model.predict(img)
    # class_index = model.predict_classes(img)
    predicted_label = np.argmax(predictions)
    probability_value = np.amax(predictions)
    if probability_value > threshold:
        cv.putText(original_img, str(classNames[predicted_label]), (120, 35), font, 0.75, (0, 0, 255), 2, cv.LINE_AA)
        cv.putText(original_img, str(round(probability_value*100, 2))+"%", (180, 75), font, 0.75, (0, 0, 255), 2, cv.LINE_AA)
    cv.imshow('Result', original_img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break