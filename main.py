"""
Pneumonia Detector: a deep-learning based software to help with diagnosing pneumonia by displaying segmentation and classification on chest X-ray images.
It supports two different methods for chest X-ray segmentation: U-Net (VGG16 based) CNN model and Watershed algorithm.
Also, it supports three different models for classification of pneumonia: CNN, ResNet 50, DenseNet 121.
"""

# Import libraries
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox
from PyQt5 import QtGui
import _thread
import time
import sys
# From window.py package import ui
from window import Ui_mainWindow
import segmentation
import cnn_prediction
import transfer_prediction


class Main(QMainWindow, Ui_mainWindow):

    # Functions
    # Initiation of the main window
    def __init__(self):
        super(Main, self).__init__()
        self.setupUi(self)

    def openfile(self):
        """Open the image file path

        :return: Save the file path as the variable self.input_path
                Call the show_input_img function.
        """

        # Open a file dialog for selecting input image file path
        openfile_name = QFileDialog.getOpenFileName()
        if openfile_name[0] != '':
            # Get the selected image path
            self.input_path = openfile_name[0]
            # Call the function to show the input image
            self.show_input_img(self.input_path)

    def show_input_img(self, file_path):
        """Show the input image

        :param file_path: The input image file path

        :return: None - Display the image in the import_img QLabel
        """

        input_img = QtGui.QPixmap(file_path)
        self.import_img.setPixmap(input_img)

    def start_segmentation(self):
        """Start of the segmentation function

        :return: None - start a new thread, or if the image has not been imported, call a warning window.
        """

        if hasattr(main, 'input_path'):
            _thread.start_new_thread(lambda: self.is_segmentation(main.input_path), ())

        else:
            QMessageBox.warning(self, 'Warning', 'Pneumonia Detector Warning \n\nSegmentation Function: Please Check If An Image Has Been Imported!', QMessageBox.Close)

    def is_segmentation(self, file_path):
        """The function of segmentation

        :param file_path: The input image file path.
        :return: The segmented image.
        """

        # Timestamp currently
        t = str(int(time.time()))
        # The export path of the coming segmented image
        export_path = 'export_img/export_img' + t + '.png'
        # Get the method user choose in the combo box
        method = main.comboBox_segmentation.currentText()
        # call image_segmentation function in segmentation.py
        after_seg = segmentation.image_segmentation(file_path, export_path, method)
        if after_seg == False:
            main.show_export_img(export_path)

    def show_export_img(self, file_path):
        """Open the pixmap of the segmented image and then show it

        :param file_path: The input image file path.
        :return: None - Display the segmented image in the export_img QLabel.
        """

        export_img = QtGui.QPixmap(file_path)
        self.export_img.setPixmap(export_img)

    def saveImage(self):
        """Save the segmented image

        :return: None - The segmented image will be stored to the specified dictionary, or no image has been segmented, call a warning window.
        """

        try:
            # Save the pixmap of the image in the export_img QLabel
            img = self.export_img.pixmap().toImage()
            fpath, ftype = QFileDialog.getSaveFileName(self.centralwidget, "Save the segmented picture", "Segmented Picture",
                                                   "*.jpg;;*.png;;All Files(*)")
            img.save(fpath)
        except:
            QMessageBox.warning(self, 'Warning', 'Pneumonia Detector Warning \n\nSave Function: Please Check If An Image Has Been Imported!', QMessageBox.Close)

    def is_cnn(self):
        """The function of running cnn model

        :return: None - Call the pneumonia_cnn_prediction in cnn_prediction.py and the show the classification
        """

        try:
            path = str(main.input_path)
            cnn_classification = cnn_prediction.pneumonia_cnn_prediction(main.input_path)
            self.textBrowser_CNN.setText("The deep learning model being used now is CNN\nThe file address is %s\n"
                                     "The classification of this chest X-ray is %s" % (path, cnn_classification))
            self.textBrowser_CNN.repaint()
        except:
            QMessageBox.warning(self, 'Warning',
                                'Pneumonia Detector Warning \n\nCNN Model: Please Check If An Image Has Been Imported!',
                                QMessageBox.Close)

    def is_transfer(self):
        """The function of running transfer learning model, and user can choose the model before running

        :return: None - Call the pneumonia_resnet_prediction or pneumonia_densenet_prediction and show the classification
        """

        # Get the method user choose in the combo box
        model = main.comboBox_transfer.currentText()
        if model == "ResNet 50":
            try:
                path = str(main.input_path)
                resnet_classification = transfer_prediction.pneumonia_resnet_prediction(main.input_path)
                self.textBrowser_transfer.setText("The transfer learning model being used now is ResNet 50\nThe file address is %s\n"
                                                  "The classification of this chest X-ray is %s" % (path, resnet_classification))
                self.textBrowser_transfer.repaint()
            except:
                QMessageBox.warning(self, 'Warning',
                                    'Pneumonia Detector Warning \n\nResNet 50 Model: Please Check If An Image Has Been Imported!',
                                    QMessageBox.Close)

        elif model == "DenseNet 121":
            try:
                path = str(main.input_path)
                densenet_classification = transfer_prediction.pneumonia_densenet_prediction(main.input_path)
                self.textBrowser_transfer.setText("The transfer learning model being used now is DenseNet 121\nThe file address is %s\n"
                                                  "The classification of this chest X-ray is %s" % (path, densenet_classification))
                self.textBrowser_transfer.repaint()
            except:
                QMessageBox.warning(self, 'Warning',
                                    'Pneumonia Detector Warning \n\nDenseNet 121 Mdoel: Please Check If An Image Has Been Imported!',
                                    QMessageBox.Close)





if __name__ == '__main__':
    # Create GUI object, create main window ui class object and show the main window
    app = QApplication(sys.argv)
    main = Main()
    main.show()

    # Connect the actions on buttons to the defined functions
    main.pushButton_browse.clicked.connect(main.openfile)
    main.pushButton_segmentation.clicked.connect(main.start_segmentation)
    main.pushButton_save.clicked.connect(main.saveImage)
    main.pushButton_CNN.clicked.connect(main.is_cnn)
    main.pushButton_transfer.clicked.connect(main.is_transfer)

    # The program will run unless the exit closes the window
    sys.exit(app.exec_())