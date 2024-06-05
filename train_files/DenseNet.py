# Import libraries
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.metrics import categorical_accuracy
from keras.applications import DenseNet121
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == '__main__':
    # Determine the paths of the files
    main_folder = r"chest_xray/"
    train_folder = main_folder + r'train/'
    val_folder = main_folder + r'val/'
    test_folder = main_folder + r'test/'

    # Scale the data with normalization
    # Also augment the training data
    train_data_generator = ImageDataGenerator(rescale=1. / 255,
                                              shear_range=0.2,
                                              zoom_range=0.2,
                                              horizontal_flip=True)

    test_data_generator = ImageDataGenerator(rescale=1. / 255)

    batch_size = 22

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

    base_model = DenseNet121(input_shape=(64, 64, 3), include_top=False, weights='imagenet')
    # base_model.summary()
    model = Sequential([
            base_model,
            GlobalAveragePooling2D(),  # Global average pooling operation for spatial data
            BatchNormalization(),  # transform the data with a mean of 0 and a standard deviation of 1
            Dense(3, activation="softmax")
        ])
    opt = Adam(amsgrad=True)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=[categorical_accuracy])

    num_of_test_samples = 600
    model.summary()
    cnn_model = model.fit(training_set,
                          steps_per_epoch=int(5216 / batch_size),
                          epochs=20,
                          validation_data=validation_set,
                          validation_steps=int(22 / batch_size))
    # Get the testing accuracy
    test_accuracy = model.evaluate(test_set, steps=624)
    print(f'Testing accuracy : {test_accuracy[1] * 100}%')  #

    # Save model structure and weights
    model.save('DenseNet121.h5')

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

    # Print the classification report
    y_pred = model.predict(test_set)
    y_pred_classes = np.argmax(y_pred, axis=1)
    print(classification_report(test_set.classes, y_pred_classes, target_names=['Normal',
                                                                                'Bacteria',
                                                                                'Virus']))

