import os
from keras.applications.vgg19 import VGG19
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.patheffects as PathEffects
import seaborn as sns
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow as tf

base_dir = '/home/shiraz/Desktop/ML/Assignment3/cats_and_dogs_filtered'
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

from keras.applications import VGG19

conv_base = VGG19(
    weights='imagenet',
    include_top=False,
    input_shape=(150, 150, 3))

conv_base.trainable = False

#Concatenate the convolutional base and densely connected layers
with tf.device("/gpu:1"):
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    #model.add(layers.Dropout(0.1))
    model.add(layers.Dense(1, activation='sigmoid'))


#Train the model end to end with frozen convolutional base
# data augmentation

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

# compile model

model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.RMSprop(lr=2e-5),
    metrics=['acc'])

# train

# history = model.fit_generator(
#     train_generator,
#     steps_per_epoch=100,
#     epochs=25,
#     validation_data=validation_generator,
#     validation_steps=50
# )


import matplotlib.pyplot as plt

# acc = history.history['acc']
# val_acc = history.history['val_acc']
#
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# epochs = range(1, len(acc) + 1)
#
# # training and validation accuracy
#
# plt.plot(epochs, acc, 'bo', label='training acc')
# plt.plot(epochs, val_acc, 'b', label='validation acc')
# plt.title('training and validation accuracy')
# plt.legend()
#
# plt.figure()
#
# # training and validation loss
#
# plt.plot(epochs, loss, 'bo', label='training loss')
# plt.plot(epochs, val_loss, 'b', label='validation loss')
# plt.title('training and validation loss')
# plt.legend()
#
# plt.show()

# Fine tuning
conv_base.trainable = True

set_trainable = False
for layer in conv_base.layers:
  if layer.name == 'block5_conv4':
    set_trainable = True
  if set_trainable:
    layer.trainable = True
  else:
    layer.trainable = False


# compile model

model.compile(
    loss='binary_crossentropy',
    #
    # choose a smaller learning rate
    #
    optimizer=optimizers.RMSprop(lr=1e-5),
    metrics=['acc'])
#
# # train
#
# history = model.fit_generator(
#     train_generator,
#     steps_per_epoch=100,
#     epochs=25,
#     validation_data=validation_generator,
#     validation_steps=50)

# acc = history.history['acc']
# val_acc = history.history['val_acc']
#
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs = range(1, len(acc) + 1)

# training and validation accuracy

# plt.plot(epochs, acc, 'bo', label='training acc')
# plt.plot(epochs, val_acc, 'b', label='validation acc')
# plt.title('training and validation accuracy')
# plt.legend()
#
# plt.figure()
#
# # training and validation loss
#
# plt.plot(epochs, loss, 'bo', label='training loss')
# plt.plot(epochs, val_loss, 'b', label='validation loss')
# plt.title('training and validation loss')
# plt.legend()
#
# plt.show()

# Print out validation loss and accuracy

val_loss, val_acc = model.evaluate_generator(validation_generator, steps=50)
print("Validation loss:", val_loss)
print("Validation accuracy:", val_acc)
#
# model_fname = 'cats_and_dogs_small_4_problem2_vgg19.h5'
# model.save(model_fname)



'''-----------------------------------------------------------------------------'''

model.load_weights('cats_and_dogs_small_4_problem2_vgg19.h5', by_name=True)
model.summary()
layer = model.get_layer('dense_2')
layer_output = layer.output
activation_model = models.Model(input=model.input, outputs=[layer_output])
dd = []
ll = []
i =0
for d,l in validation_generator:
  dd.append(d)
  ll.append(l)
  i = i+1
  if i >50:
    break

data = np.array(dd)
labels = np.array(ll)

data = np.reshape(data, (data.shape[0]*data.shape[1],) + data.shape[2:])
labels = np.reshape(labels, (labels.shape[0]*labels.shape[1],) + labels.shape[2:])
activations = activation_model.predict(data)



import seaborn as sns
class_names = ['Cats', 'Dogs']

for idx in range(len(class_names)):
  print(idx, ":", class_names[idx])
def data_scatter(vecs, labels):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(labels))
    palette = np.array(sns.color_palette("husl", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(12, 12))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(vecs[:, 0], vecs[:, 1], c=palette[labels])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')


    for idx in range(num_classes):

        # Place label at median position of vectors with corresponding label

        x_coord, y_coord = np.median(vecs[labels == idx, :], axis=0)
        txt = ax.text(x_coord, y_coord, class_names[idx], fontsize=16)
        # plot class index black with white contour
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=6, foreground="w"),
            PathEffects.Normal()])
    plt.show()
from sklearn.manifold import TSNE
import time
time_start = time.time()

tsne = TSNE(random_state=42).fit_transform(activations)

print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))

data_scatter(tsne, labels.astype('uint8'))
