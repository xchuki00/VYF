from tensorflow.keras.applications import VGG16

from encoding import Encoding
from decoding import Decoding
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

img_height = 512
img_width = 512
img_channels = 3
w1 = 1
w2 = 0
import os
vgg16 = VGG16(weights='imagenet', include_top=False)


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


def lossFunction(y_true, y_pred):
    # L = LV(E,D) + w1*Lc(E) + w2*Lq(E)
    Lv = mean_squared_error(y_true, y_pred)
    # l1 = ew-abs(ew-max(|G-L(I)|-Mo,M0))
    l1 = tf.keras.regularizers.l1(tf.keras.backend.maximum(tf.keras.backend.abs(y_true - y_pred) - 70, 0))
    lc = tf.keras.regularizers.l1(vgg16.predict(y_true)-vgg16.predict(y_pred))
    ls = tf.keras.regularizers.l1(tf.image.total_variation(y_true)-tf.image.total_variation(y_true))
    Lc = l1 + 1**(-7)*lc + 0.5*ls
    # Lq = Element-wise-minum(G-Md)
    Lq = tf.keras.regularizers.l1(tf.keras.backend.minimum(tf.keras.backend.abs(y_true - y_pred)))
    return Lv + w1 * Lc + w2 * Lq


def train(enc,dec):
    for file in os.listdir("./data/color"):
        if file.endswith(".jpg"):
            print(os.path.join(file))
            img = image.load_img("./data/color/" + file, target_size=(img_height, img_height))
            # imgG = image.load_img("./data/gray/"+file, target_size=(img_height, img_height))

            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            # y = image.img_to_array(imgG)
            # y = np.expand_dims(y, axis=0)
            enc.fit(x, x)
            dec.fit(enc.predict(x),x)

image_tensor = layers.Input(shape=(512, 512, 3))
encodig_output = Encoding(image_tensor)

enc = models.Model(inputs=[image_tensor], outputs=[encodig_output])

decoding_output = Decoding(encodig_output)

dec = models.Model(inputs=[encodig_output], outputs=[decoding_output])

enc.compile(optimizer='sgd',
              loss=lossFunction,
              metrics=['mse'])
dec.compile(optimizer='sgd',
              loss=lossFunction,
              metrics=['mse'])
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.MeanSquaredError(),
#               metrics=['mse'])
train(enc,dec)
img_path = './data/color/christ_church_000317.jpg'
img = image.load_img(img_path, target_size=(img_height, img_height))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

# x = preprocess_input(x)
preds = enc.predict(x)
print(preds.dtype)
print(preds.shape)
# convert back to image
pred_mask = tf.argmax(preds, axis=-1)
pred_mask = pred_mask[..., tf.newaxis]

display([pred_mask[0]])
preds = dec.predict(preds)
print(preds.dtype)
print(preds.shape)
# convert back to image
pred_mask = tf.argmax(preds, axis=-1)
pred_mask = pred_mask[..., tf.newaxis]

display([pred_mask[0]])

# create a data generator
# datagen = ImageDataGenerator(
#         rescale=1./255,
#     featurewise_center=True,
#     featurewise_std_normalization=True,
#       )
# # load and iterate training dataset
# train_it = datagen.flow_from_directory('./data/color/', target_size=(img_height, img_width), color_mode='rgb',class_mode='binary', batch_size=32)
# # load and iterate validation dataset
# val_it = datagen.flow_from_directory('./data/gray/', target_size=(img_width, img_width), color_mode='rgb',class_mode='binary', batch_size=32)
# # model.fit(train_it, steps_per_epoch=10, validation_data=val_it, validation_steps=8)
# # configure batch size and retrieve one batch of images
# # fits the model on batches with real-time data augmentation:
#
# for e in range(1):
#     print('Epoch', e)
#     batches = 0
#     for x_batch in datagen.flow(train_it, batch_size=32):
#         print('Epoch start - ', e)
#         model.fit(x_batch)
#         print('Epoch end - ', e)
#         batches += 1
#         if batches >= 0: #len(train_it)/32:
#             # we need to break the loop by hand because
#             # the generator loops indefinitely
#             break
