from encoding import Encoding
from decoding import Decoding
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
img_height = 512
img_width = 512
img_channels = 3



def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


image_tensor = layers.Input(shape=(512, 512, 3))
network_output = Encoding(image_tensor)
network_output = Decoding(network_output)

model = models.Model(inputs=[image_tensor], outputs=[network_output])

model.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['accuracy'])

# create a data generator
datagen = ImageDataGenerator(
        rescale=1./255,
    featurewise_center=True,
    featurewise_std_normalization=True,
      )
# load and iterate training dataset
train_it = datagen.flow_from_directory('./data/color/', target_size=(img_height, img_width), color_mode='rgb',class_mode='binary', batch_size=32)
# load and iterate validation dataset
val_it = datagen.flow_from_directory('./data/gray/', target_size=(img_width, img_width), color_mode='rgb',class_mode='binary', batch_size=32)
# model.fit(train_it, steps_per_epoch=10, validation_data=val_it, validation_steps=8)
# configure batch size and retrieve one batch of images
# fits the model on batches with real-time data augmentation:
# model.fit_generator(datagen.flow(train_it, val_it, batch_size=32),
#                     steps_per_epoch=len(train_it) / 32, epochs=10)
for e in range(1):
    print('Epoch', e)
    batches = 0
    for x_batch, y_batch in datagen.flow(train_it, val_it, batch_size=1):
        print('Epoch start - ', e)
        model.fit(x_batch)
        print('Epoch end - ', e)
        batches += 1
        if batches >= len(train_it):
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break

img_path = './data/color/74c5a964fdf745c8207a63919c02f9b1.jpg'
img = image.load_img(img_path, target_size=(img_height, img_height))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)

preds = model.predict(x)
print(preds.dtype)
print(preds.shape)
# convert back to image
pred_mask = tf.argmax(preds, axis=-1)
pred_mask = pred_mask[..., tf.newaxis]

display([pred_mask[0]])