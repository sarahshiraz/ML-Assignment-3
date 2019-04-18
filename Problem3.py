from keras.applications.vgg19 import VGG19
import tensorflow as tf
import os
import cv2
from keras import models

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from keras.preprocessing.image import ImageDataGenerator


# note that we keep the densely connected classifier;
# in the two previous vizualization methods, we discarded it
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("path", help="path of the dataset base directory",
                    type=str)
args = parser.parse_args()
print(args.path+'2')


#base_dir = '/home/shiraz/Desktop/ML/Assignment3/cats_and_dogs_filtered'
base_dir = args.path
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# Directory with our training cat pictures
train_cats_dir = os.path.join(train_dir, 'cats')

# Directory with our training dog pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')

# Directory with our validation cat pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')

# Directory with our validation dog pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

model = VGG19()
model.load_weights('cats_and_dogs_small_4_problem2_vgg19.h5', by_name=True)

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions

img_path1 = base_dir + '/train/cats/cat.35.jpg'
#img_path2 = 'cats_and_dogs_filtered/train/cats/cat.39.jpg'
#img_path3 = 'cats_and_dogs_filtered/train/cats/cat.40.jpg'

last_conv_layer = model.get_layer('block5_conv4')
import cv2
# from google.colab.patches import cv2_imshow
from matplotlib import pyplot as plt
import numpy as np
from keras import backend as K

def show_superimposed_image(img_path, heatmap):
    img = cv2.imread(img_path)
    cv2.imwrite('Image.png', img)
    cv2.imshow('dsaada', img)
    cv2.waitKey(0)

    # resize the heatmap to be the same size as the original image
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * 0.4 + img

    print(np.min(superimposed_img))
    print(np.max(superimposed_img))
    superimposed_img = np.array(superimposed_img, dtype=np.uint8)
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)


    print(np.min(superimposed_img))
    print(np.max(superimposed_img))
    print(superimposed_img.shape)
    cv2.imwrite('Class Activation.png', superimposed_img)
    cv2.imshow('dsd', superimposed_img)
    cv2.waitKey(0)

def process_image(img_path, idx=0):
    img = image.load_img(img_path, target_size=(224, 224))
    print('imageee')
    print(image)
    plt.imshow(img)
    plt.grid(None)
    plt.show()
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds, top=3)[0])
    class_index = np.argsort(preds[0])[-(1+idx)]
    class_output = model.output[:, class_index]
    grads = K.gradients(class_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x])
    for i in range(512):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    plt.matshow(heatmap)
    plt.grid(None)
    plt.show()
    show_superimposed_image(img_path, heatmap)

process_image(img_path1)
