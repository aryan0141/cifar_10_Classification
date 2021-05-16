from django.http import HttpResponse
from django.shortcuts import render, redirect 
from .models import Images
import cv2
from imutils import rotate_bound
from os import listdir,remove
from os.path import isfile, join
import tensorflow as tf
from tensorflow.keras.models import load_model
#from tensorflow import keras

import joblib
import numpy as np
import pandas as pd

# print("[INFO] loading the trained networks...")
#cnn_model = keras.models.load_model('trained_models/new_model.h5')
cnn_model = load_model('trained_models/cnn.h5')
# rfc_model = joblib.load('trained_models/random_forest_model')
dic =  {0: 'Airplane',										
        1: 'Automobile',										
        2: 'Bird',										
        3: 'Cat',										
        4: 'Deer',										
        5: 'Dog',										
        6: 'Frog',									
        7: 'Horse',										
        8: 'Ship',										
        9: 'Truck'}
# Create your views here.
def index(request): 
    if request.method == 'POST': 
        # form = Image(request.POST, request.FILES)
        pic = request.FILES['image']
        print(pic)
        image = Images(UploadImage = pic)
        image.save()
        # if form.is_valid(): 
        data_path = 'images'
        onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]
        img=cv2.imread(f"images/{onlyfiles[0]}")
        img=cv2.resize(img,(32,32),interpolation=cv2.INTER_AREA)
        # print(img.shape)
        img = np.array(img).astype('float32')/255
        img = np.expand_dims(img, axis=0)
        
        label = cnn_model.predict(img)
        index = list(label[0]).index(max(label[0]))
        label = dic[index]
        proba = max(label[0])*100

        # proba = "{:.2f}".format(proba)

        params = {'label':label,  
                'proba':proba
                }
        # print(params)
        remove(f"images/{onlyfiles[0]}")	 	

        return render(request,'result.html', params)  
    else: 
        return render(request,'index.html') 
