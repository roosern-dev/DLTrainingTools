import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Conv2D
import os
from tensorflow.python.framework import graph_io

image_size = 64 # All images will be resized to 32x32
batch_size = 32
train_dir = r'E:\Estek-AIProject\INTEL-SOOP\RaspberryPiFolder\Images\Training\TrainingSet31_12_19\trainValidSplit\train'
validation_dir = r'E:\Estek-AIProject\INTEL-SOOP\RaspberryPiFolder\Images\Training\TrainingSet31_12_19\trainValidSplit\valid'
saveDirectory = 'E:\\Estek-AIProject\\INTEL-SOOP\\Trainings\\'
saveFile = r'classifier_pass-soop_31-12-19-MobileNetV2I1_cornersonly64'
backSlash = '\\'

saveSubDirectory = saveDirectory + saveFile
if not os.path.exists(saveSubDirectory):
  os.makedirs(saveSubDirectory)

savePath = saveSubDirectory + backSlash + saveFile + '.h5'
weightsSavePath = saveSubDirectory + backSlash + 'easy_checkpoint'



# Rescale all images by 1./255 and apply image augmentation
train_datagen = keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255)

validation_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
                train_dir,  # Source directory for the training images
                target_size=(image_size, image_size),
                batch_size=batch_size, shuffle= True,
                # Since we use binary_crossentropy loss, we need binary labels
                class_mode='binary')

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = validation_datagen.flow_from_directory(
                validation_dir, # Source directory for the validation images
                target_size=(image_size, image_size),
                batch_size=batch_size, shuffle=True,
                class_mode='binary')

IMG_SHAPE = (image_size, image_size, 3)


# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')


base_model.trainable = True

model = tf.keras.Sequential([
  base_model,keras.layers.GlobalAveragePooling2D(),
  keras.layers.Dense(10, activation='sigmoid', kernel_regularizer=keras.regularizers.L1L2(0.01)),
  #keras.layers.Dense(5, activation='sigmoid', kernel_regularizer=keras.regularizers.L1L2(0.01)),
  keras.layers.Dense(2, activation='sigmoid', kernel_regularizer=keras.regularizers.L1L2(0.01), name='output_node')
])
'''
base_model = tf.keras.applications.ResNet50(include_top= False, weights = 'imagenet', input_shape= IMG_SHAPE, classes= 2)
base_model.trainable = False

model = tf.keras.Sequential([base_model, keras.layers.GlobalAveragePooling2D(),
keras.layers.Dense(10, activation='sigmoid'),
keras.layers.Dense(1, activation='sigmoid')
])
'''
model.summary()
optim = tf.keras.optimizers.RMSprop(lr=0.001, decay=0.0001)
schedule = tf.keras.optimizers.schedules.LearningRateSchedule()

model.compile(optimizer= optim,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']) 

epochs = 40
steps_per_epoch = train_generator.n // batch_size
validation_steps = validation_generator.n // batch_size

logdir = saveSubDirectory + r'\tensorboard\logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

saveBestDirectory = saveSubDirectory + '\\' + 'mymodelCorners64px_{epoch}.h5'
saveBest = keras.callbacks.ModelCheckpoint(filepath = saveBestDirectory, save_best_only=True, monitor='val_acc', verbose = 1)
callbacks = [saveBest]

history = model.fit_generator(train_generator,
                              steps_per_epoch = steps_per_epoch,
                              epochs=epochs,
                              workers=4,
                              validation_data=validation_generator,
                              validation_steps=validation_steps,
                              callbacks=callbacks)


model.save(savePath)
#model.save(savedModelPath, save_format='tf')




print('save complete!')

acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,max(plt.ylim())])
plt.title('Training and Validation Loss')
figureSavePath = saveSubDirectory + '\\plot.png'
plt.savefig(figureSavePath)
plt.show()

sess = tf.compat.v1.keras.backend.get_session()
frozen = tf.compat.v1.graph_util.convert_variables_to_constants(sess, sess.graph_def, [model.output.op.name])
graph_io.write_graph(frozen, './', logdir=saveSubDirectory, name='inference_graph.pb', as_text = False)


model.save_weights(weightsSavePath, save_format = 'tf')
savedModelPath = saveSubDirectory + backSlash + 'savedModel'