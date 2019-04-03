# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import time
from django.shortcuts import render
from django.conf import settings

from django.http import HttpResponse

import skimage
import numpy as np
from skimage import data, color, transform
import keras
from keras.utils import np_utils
from keras.models import model_from_json
import json
import tensorflow as tf

model_path = settings.BASE_DIR
with open(model_path + '/ohfruit1.3.3_model.json', 'r') as f:
    model_json = f.read()

fruit_model = model_from_json(model_json)
fruit_model.load_weights(model_path + '/ohfruit1.3.3_weights.h5')
#get the main process's graph（获取主线程的计算图）
mygraph = tf.get_default_graph()

def index(request):
    if request.method == "POST":
        img = request.FILES.get('img')
        tmp_time = int(round(time.time()*1000))
        save_path = settings.MEDIA_ROOT + "/" + str(tmp_time) + img.name
        with open(save_path, 'wb') as f:
            for content in img.chunks():
                f.write(content)

        #predict picture
        images = skimage.data.imread(save_path)
        images = np.array(images)
        images = images.reshape(1, 100, 100, 3)

        global fruit_model, mygraph
        #set the graph as the main process's graph(设置当前的计算图为主线程的计算图)
        with mygraph.as_default():
            result = fruit_model.predict(images)

        label_index = 0

        result = result[0]

        for myiter in range(len(result)):
            if result[myiter] == 1:
                label_index = myiter
                break

        return HttpResponse(label_index)
    return render(request, 'fruit_predict/index.html')
