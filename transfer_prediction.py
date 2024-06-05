"""
transfer_prediction.py is created to call the trained ResNet 50 and DenseNet 121 model to do pneumonia classification.
"""

# Import libraries
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import shutil
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Functions

def pneumonia_resnet_prediction(import_img):
    """Call the trained 'ResNet50.h5' model to do CNN classification.

    :param import_img: The input image file path.
    :return: Return the classification (label) of the pneumonia in the input image.
    """
    main_folder = r"chest_xray/"
    predict_folder = main_folder + r'pred/'
    child_folder = predict_folder + r'Unknown'

    # Load the trained model
    model = load_model('trained_models/ResNet50.h5')

    # Specify the path of the input file here
    file_path = import_img
    # Copy the image to the prediction set
    shutil.copy(file_path, child_folder)

    # Build ImageDataGenerator
    test_data_generator = ImageDataGenerator(rescale=1. / 255)
    predict_set = test_data_generator.flow_from_directory(predict_folder,
                                                          target_size=(64, 64),
                                                          batch_size=1,
                                                          class_mode=None,
                                                          shuffle=False,
                                                          seed=1)

    # Predict the index based on the input image
    prediction = model.predict(predict_set)
    index = np.argmax(prediction[0])

    # Get the label name
    if index == 0:
        label = 'Normal'
    elif index == 1:
        label = 'Bacteria'
    else:
        label = 'Virus'

    # Print the classification result using human language :)
    print(f'The image may belong to {label} class.')

    # Delete the image in the prediction folder
    for root, dirs, files in os.walk(child_folder):
        for name in files:
            os.remove(root+'/'+name)
    return label


def pneumonia_densenet_prediction(import_img):
    """Call the trained 'DenseNet121.h5' model to do CNN classification.

    :param import_img: The input image file path.
    :return: Return the classification (label) of the pneumonia in the input image.
    """
    main_folder = r"chest_xray/"
    predict_folder = main_folder + r'pred/'
    child_folder = predict_folder + r'Unknown'

    # Load the trained model
    model = load_model('trained_models/DenseNet121.h5')

    # Specify the path of the input file here
    file_path = import_img
    # Copy the image to the prediction set
    shutil.copy(file_path, child_folder)

    # Build ImageDataGenerator
    test_data_generator = ImageDataGenerator(rescale=1. / 255)
    predict_set = test_data_generator.flow_from_directory(predict_folder,
                                                          target_size=(64, 64),
                                                          batch_size=1,
                                                          class_mode=None,
                                                          shuffle=False,
                                                          seed=1)

    # Predict the index based on the input image
    prediction = model.predict(predict_set)
    index = np.argmax(prediction[0])

    # Get the label name
    if index == 0:
        label = 'Normal'
    elif index == 1:
        label = 'Bacteria'
    else:
        label = 'Virus'

    # Print the classification result using human language :)
    print(f'The image may belong to {label} class.')

    # Delete the image in the prediction folder
    for root, dirs, files in os.walk(child_folder):
        for name in files:
            os.remove(root + '/' + name)
    return label