import kagglehub
from keras import backend as K
from keras.api.layers import Activation, Dropout, Flatten, Dense
from keras.api.layers import Conv2D, MaxPooling2D
from keras.api.models import Sequential
from keras_preprocessing.image import ImageDataGenerator

print("Keras backend:", K.backend())

img_width, img_height = 250, 250

# Download latest version
path = kagglehub.dataset_download("csafrit2/plant-leaves-for-image-classification")

print("Path to dataset files:", path)

train_data_dir = path + r"\Plants_2\train"
validation_data_dir = path + r"\Plants_2\valid"
nb_train_samples = 4274
nb_validation_samples = 110
epochs = 10
batch_size = 16

# Correct the data format
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# Model Parameters
model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(
    loss='binary_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy']
)

# Data generation
train_datagen = ImageDataGenerator(
    rotation_range=40,
    horizontal_flip=True,
)

test_datagen = ImageDataGenerator(
    rotation_range=40,
    horizontal_flip=True,
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    classes=['healthy', 'unhealthy']
)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    classes=['healthy', 'unhealthy']
)

if __name__ == "__main__":
    model.fit(
        train_generator,
        # steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        # validation_steps=nb_validation_samples // (batch_size * epochs)
    )

    model.save_weights('model_saved.weights.h5')
    model.save('model_saved.keras')
