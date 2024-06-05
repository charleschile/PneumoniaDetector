# Import libraries
import os

from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.metrics import categorical_accuracy
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == '__main__':
    # Determine the paths of the files
    main_folder = r"chest_xray/"
    train_folder = main_folder + r'train/'
    val_folder = main_folder + r'val/'
    test_folder = main_folder + r'test/'
    # predict_folder = main_folder + r'pred/'

    # Generate pseudo-random numbers
    # seed(1)

    batch_size = 22

    # Scale the data with normalization
    # Also augment the training data
    train_data_generator = ImageDataGenerator(rescale=1. / 255,
                                              shear_range=0.2,
                                              zoom_range=0.2,
                                              horizontal_flip=True)

    test_data_generator = ImageDataGenerator(rescale=1. / 255)

    training_set = train_data_generator.flow_from_directory(train_folder,
                                                            target_size=(64, 64),
                                                            batch_size=batch_size,
                                                            class_mode='categorical',
                                                            seed=1)

    validation_set = test_data_generator.flow_from_directory(val_folder,
                                                             target_size=(64, 64),
                                                             batch_size=batch_size,
                                                             class_mode='categorical',
                                                             seed=1)

    test_set = test_data_generator.flow_from_directory(test_folder,
                                                       target_size=(64, 64),
                                                       batch_size=batch_size,
                                                       class_mode='categorical',
                                                       seed=1)

    # Build a CNN model with linear stack of layers
    model = Sequential()
    # 1st convolution layer
    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)))
    # 1st max pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 2nd convolution layer
    model.add(Conv2D(32, (3, 3), activation="relu"))
    # 2nd max pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Flatten the layer
    model.add(Flatten())
    # Dense Layers
    model.add(Dense(activation='relu', units=128))
    model.add(Dense(activation='softmax', units=3))
    # Generate the optimizer
    opt = Adam(learning_rate=6e-4, decay=1e-6, amsgrad=True)
    # Compile the model
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=[categorical_accuracy])

    num_of_test_samples = 600

    # Print the summary of the model: structure, model size, etc.
    model.summary()
    # Fit the model with images
    cnn_model = model.fit(training_set,
                          steps_per_epoch=int(5216/batch_size),
                          epochs=20,
                          validation_data=validation_set,
                          validation_steps=int(22/batch_size))
    # Get the testing accuracy
    test_accuracy = model.evaluate(test_set, steps=624)
    print(f'Testing accuracy : {test_accuracy[1] * 100}%')  # 84.61538553237915%

    # Save model structure and weights
    model.save('simple_CNN.h5')
    # del model

    # Plot the evolution
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(cnn_model.history['loss'], label='Loss')
    plt.plot(cnn_model.history['val_loss'], label='Val_Loss')
    plt.legend()
    plt.title('Loss Evolution')

    plt.subplot(2, 2, 2)
    plt.plot(cnn_model.history['categorical_accuracy'], label='Accuracy')
    plt.plot(cnn_model.history['val_categorical_accuracy'], label='Val_Accuracy')
    plt.legend()
    plt.title('Accuracy Evolution')
    plt.show()

    """
    results = model.evaluate(test_set, steps=len(test_set))
    print('Test loss:', results[0])
    print('Test categorical_accuracy:', results[1])
    print('Test precision:', results[2])
    print('Test recall:', results[3])
    print('Test auc:', results[4])
    """

    # Print the classification report
    y_pred = model.predict(test_set)
    y_pred_classes = np.argmax(y_pred, axis=1)
    print(classification_report(test_set.classes, y_pred_classes, target_names=['Normal',
                                                                                'Bacteria',
                                                                                'Virus']))
