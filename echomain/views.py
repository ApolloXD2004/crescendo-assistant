from django.shortcuts import render
from django.http import HttpResponse, request
import base64
import cv2
import time
import keyboard
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import sys
import os
from .forms import ImageForm
# Create your views here.
form = ImageForm()

def home(request):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
        
        
        path = "media/blabla.jpg"
        model = load_model("mlmodels/emotion_detection.h5")
        size = (48, 48)

        def _resize_image(image, size):
            return cv2.resize(image,
                              dsize=(size[0], size[1]),
                              interpolation=cv2.INTER_LINEAR)

        emotion_classes = [
            "angry", "fear", "disgust", "happy", "sad", "surprised", "neutral"
        ]

        def detect_emotion(image):
            emotion = model.predict(image)
            emotion_nums = max(emotion)
            emotion_index = np.argmax(emotion)
            real_emotion = emotion_classes[emotion_index]
            real_emotion = real_emotion.capitalize()
            return real_emotion

        def create_facial_boundary(path, classifier):
            image = cv2.imread(path)
            grayScaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = cascadeClassifier.detectMultiScale(grayScaled, 1.1, 10)
            coords = []
            color = {
                "black": (0, 0, 0),
                "orange": (0, 145, 255),
                "white": (255, 255, 255)
                }
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), color["orange"], 2)
                the_face = image[y:y + h, x:x + w]
                gray_scaled = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)
                the_face = np.resize(gray_scaled, model.input_shape[1:3])
                the_face = np.expand_dims(the_face, 0)
                the_face = np.expand_dims(the_face, -1)
                real_emotion = detect_emotion(the_face)
                cv2.putText(image, real_emotion, (x, y - 4),
                        cv2.FONT_HERSHEY_PLAIN, 2, color["white"], 2)

            coords = [x, y, w, h]

            return coords, image, real_emotion

        def detect_face(image, cascadeClassifier):
            coords, image, real_emotion = create_facial_boundary(
                image, cascadeClassifier)

            return image, coords, real_emotion

        cascadeClassifier = cv2.CascadeClassifier(
        "mlmodels/haarcascade_frontalface_default.xml")

        image, coords, emotionstatus = detect_face(path, cascadeClassifier)

        cv2.imwrite(("(editied)" + path), image)

    
        if (emotionstatus):
            print(emotionstatus)
            return render(request, 'emotionfound.html',
                      {'emotionstatus': emotionstatus})
            # Get the current instance object to display in the template
            
    else:
        form = ImageForm()
    return render(request, 'index.html', {'form': form})


def camera(request):
    return render(request, 'webcamphoto.html')


def modelate(request):
    if request.method == 'POST':
        incomingimage = request.POST['mydata']

        imgdata = base64.b64decode(incomingimage)
        filename = 'found/testImage.jpg'
        with open(filename, 'wb') as f:
            f.write(imgdata)

        path = "found/testImage.jpg"
        model = load_model("mlmodels/emotion_detection.h5")
        size = (48, 48)

        def _resize_image(image, size):
            return cv2.resize(image,
                              dsize=(size[0], size[1]),
                              interpolation=cv2.INTER_LINEAR)

        emotion_classes = [
            "angry", "fear", "disgust", "happy", "sad", "surprised", "neutral"
        ]

    def detect_emotion(image):
        emotion = model.predict(image)
        emotion_nums = max(emotion)
        emotion_index = np.argmax(emotion)
        real_emotion = emotion_classes[emotion_index]
        real_emotion = real_emotion.capitalize()
        return real_emotion

    def create_facial_boundary(path, classifier):
        image = cv2.imread(path)
        grayScaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = cascadeClassifier.detectMultiScale(grayScaled, 1.1, 10)
        coords = []
        color = {
            "black": (0, 0, 0),
            "orange": (0, 145, 255),
            "white": (255, 255, 255)
        }
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), color["orange"], 2)
            the_face = image[y:y + h, x:x + w]
            gray_scaled = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)
            the_face = np.resize(gray_scaled, model.input_shape[1:3])
            the_face = np.expand_dims(the_face, 0)
            the_face = np.expand_dims(the_face, -1)
            real_emotion = detect_emotion(the_face)
            cv2.putText(image, real_emotion, (x, y - 4),
                        cv2.FONT_HERSHEY_PLAIN, 2, color["white"], 2)

        coords = [x, y, w, h]

        return coords, image, real_emotion

    def detect_face(image, cascadeClassifier):
        coords, image, real_emotion = create_facial_boundary(
            image, cascadeClassifier)

        return image, coords, real_emotion

    cascadeClassifier = cv2.CascadeClassifier(
        "mlmodels/haarcascade_frontalface_default.xml")

    image, coords, emotionstatus = detect_face(path, cascadeClassifier)

    cv2.imwrite(("(editied)" + path), image)

    print(f)
    if (emotionstatus):
        print(emotionstatus)
        return render(request, 'emotionfound.html',
                      {'emotionstatus': emotionstatus})
    else:
        return render(request, 'error.html')




    