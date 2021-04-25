#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2020

@author: Walid Khelifa
"""
# importer les labraries
import face_recognition 
import cv2
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json

#loading the image to detect
image_detection = cv2.imread('images/walid2.JPG')

#load the model and load the weights
face_exp_model = model_from_json(open("dataset/facial_expression_model_structure.json","r").read())
face_exp_model.load_weights('dataset/facial_expression_model_weights.h5')

#declare the emotions label
emotions_label = ('En colere', 'Degouter', 'Peur', 'Heureux', 'Triste', 'Surpris', 'Neutre')

#detect all faces in the image (number of faces and his location)
all_face_locations = face_recognition.face_locations(image_detection,model='hog') 
print('\n')
print('Le nombre de visages detectes est : {}'.format(len(all_face_locations)))
print('\n')
print(all_face_locations) # all_face_locations contient les positions de chaque visage


for index,current_face_location in enumerate(all_face_locations):
    #Get the position values
    top_pos,right_pos,bottom_pos,left_pos = current_face_location
    print('Le visage {} est en position : {}h {}d {}b {}g '.format(index+1,top_pos,right_pos,bottom_pos,left_pos))
    # after getting the location for each face , now we will splitting just the face for each person
    current_face_image = image_detection[top_pos:bottom_pos,left_pos:right_pos]
    #draw rectangle around the face detected
    cv2.rectangle(image_detection,(left_pos,top_pos),(right_pos,bottom_pos),(0,0,255),2)
    
    #preprocess input, convert it to an image like as the data in dataset
    #convert to grayscale
    current_face_image = cv2.cvtColor(current_face_image, cv2.COLOR_BGR2GRAY) 
    #resize to 48x48 px size 
    current_face_image = cv2.resize(current_face_image, (48, 48))
    #convert the PIL image into a 3d numpy array
    img_pixels = image.img_to_array(current_face_image)
    #expand the shape of an array into single row multiple columns
    img_pixels = np.expand_dims(img_pixels, axis = 0)
    #pixels are in range of [0, 255]. normalize all pixels in scale of [0, 1]
    img_pixels /= 255 
    
    #do prodiction using model, get the prediction values for all 7 expressions
    exp_predictions = face_exp_model.predict(img_pixels) 
    #find max indexed prediction value (0 till 7)
    max_index = np.argmax(exp_predictions[0])
    #get corresponding lable from emotions_label
    emotion_label = emotions_label[max_index]
    
    #display the name as text in the image
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image_detection, emotion_label, (left_pos+20,bottom_pos+25), font, 1, (255,255,255),2)
    
cv2.imshow("Face emotion detection ",image_detection)
    

