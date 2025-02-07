"""
This window.py is created by PyQt5 UI code generator 5.15.7 automatically after the BIA.ui file is generated by the Qt Designer.
Some manual changes have been made to this file to better the connection between our ui and our functions.
"""

# Import libraries
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QDesktopWidget


class Ui_mainWindow(object):
    def setupUi(self, mainWindow):

        mainWindow.setObjectName("main window")
        mainWindow.resize(821, 872)
        # Keep the main window size constant
        mainWindow.setMinimumSize(QtCore.QSize(821, 872))
        mainWindow.setMaximumSize(QtCore.QSize(821, 872))
        screen = QDesktopWidget().screenGeometry()
        size = mainWindow.geometry()
        mainWindow.move((int)((screen.width() - size.width()) / 2), (int)((screen.height() - size.height()) / 2))
        mainWindow.setStyleSheet("")
        self.centralwidget = QtWidgets.QWidget(mainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox_file = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_file.setGeometry(QtCore.QRect(10, 10, 801, 461))
        self.groupBox_file.setObjectName("groupBox_file")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.groupBox_file)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(20, 100, 761, 351))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")

        # The import image QLabel
        self.import_img = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.import_img.setStyleSheet("")
        self.import_img.setText("")
        self.import_img.setObjectName("import_img")
        # display at the center
        self.import_img.setAlignment(QtCore.Qt.AlignCenter)
        # autoscaling
        self.import_img.setScaledContents(True)
        self.import_img.setGeometry(QtCore.QRect(1, 1, 376, 349))

        # The export image QLabel
        self.export_img = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.export_img.setStyleSheet("")
        self.export_img.setText("")
        self.export_img.setObjectName("export_img")
        # display at the center
        self.export_img.setAlignment(QtCore.Qt.AlignCenter)
        # autoscaling
        self.export_img.setScaledContents(True)
        self.export_img.setGeometry(QtCore.QRect(385, 1, 375, 349))

        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self.groupBox_file)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(20, 30, 761, 51))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)

        # The browse file push button
        self.pushButton_browse = QtWidgets.QPushButton(self.horizontalLayoutWidget_2)
        self.pushButton_browse.setObjectName("pushButton_browse")

        self.horizontalLayout_2.addWidget(self.pushButton_browse)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem1)

        # The segmentation choices combo box
        self.comboBox_segmentation = QtWidgets.QComboBox(self.horizontalLayoutWidget_2)
        self.comboBox_segmentation.setObjectName("comboBox_segmentation")
        self.comboBox_segmentation.addItem("")
        self.comboBox_segmentation.addItem("")

        self.horizontalLayout_2.addWidget(self.comboBox_segmentation)

        # The segmentation function push button
        self.pushButton_segmentation = QtWidgets.QPushButton(self.horizontalLayoutWidget_2)
        self.pushButton_segmentation.setObjectName("pushButton_segmentation")

        self.horizontalLayout_2.addWidget(self.pushButton_segmentation)

        # The save function push button
        self.pushButton_save = QtWidgets.QPushButton(self.horizontalLayoutWidget_2)
        self.pushButton_save.setObjectName("pushButton_save")

        self.horizontalLayout_2.addWidget(self.pushButton_save)

        self.groupBox_classification = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_classification.setGeometry(QtCore.QRect(10, 480, 801, 341))
        self.groupBox_classification.setObjectName("groupBox_classification")

        self.horizontalLayoutWidget_3 = QtWidgets.QWidget(self.groupBox_classification)
        self.horizontalLayoutWidget_3.setGeometry(QtCore.QRect(20, 30, 761, 51))
        self.horizontalLayoutWidget_3.setObjectName("horizontalLayoutWidget_3")

        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_3)
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")

        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem2)

        self.label_CNN = QtWidgets.QLabel(self.horizontalLayoutWidget_3)
        self.label_CNN.setObjectName("label_CNN")

        self.horizontalLayout_5.addWidget(self.label_CNN)

        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem3)

        # The CNN model run function push button
        self.pushButton_CNN = QtWidgets.QPushButton(self.horizontalLayoutWidget_3)
        self.pushButton_CNN.setObjectName("pushButton_CNN")

        self.horizontalLayout_5.addWidget(self.pushButton_CNN)
        self.horizontalLayoutWidget_4 = QtWidgets.QWidget(self.groupBox_classification)
        self.horizontalLayoutWidget_4.setGeometry(QtCore.QRect(20, 90, 761, 81))
        self.horizontalLayoutWidget_4.setObjectName("horizontalLayoutWidget_4")

        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_4)
        self.horizontalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")

        self.label_CNN_classification = QtWidgets.QLabel(self.horizontalLayoutWidget_4)
        self.label_CNN_classification.setObjectName("label_CNN_classification")

        self.horizontalLayout_7.addWidget(self.label_CNN_classification)

        # The CNN result text browser
        self.textBrowser_CNN = QtWidgets.QTextBrowser(self.horizontalLayoutWidget_4)
        self.textBrowser_CNN.setObjectName("textBrowser_CNN")

        self.horizontalLayout_7.addWidget(self.textBrowser_CNN)
        self.horizontalLayoutWidget_5 = QtWidgets.QWidget(self.groupBox_classification)
        self.horizontalLayoutWidget_5.setGeometry(QtCore.QRect(20, 180, 761, 51))
        self.horizontalLayoutWidget_5.setObjectName("horizontalLayoutWidget_5")

        self.horizontalLayout_8 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_5)
        self.horizontalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")

        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_8.addItem(spacerItem4)

        self.label_transfer = QtWidgets.QLabel(self.horizontalLayoutWidget_5)
        self.label_transfer.setObjectName("label_transfer")

        self.horizontalLayout_8.addWidget(self.label_transfer)

        # The transfer learning choices combo box
        self.comboBox_transfer = QtWidgets.QComboBox(self.horizontalLayoutWidget_5)
        self.comboBox_transfer.setObjectName("comboBox_transfer")
        self.comboBox_transfer.addItem("")
        self.comboBox_transfer.addItem("")

        self.horizontalLayout_8.addWidget(self.comboBox_transfer)
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_8.addItem(spacerItem5)

        # The transfer learning model run function push button
        self.pushButton_transfer = QtWidgets.QPushButton(self.horizontalLayoutWidget_5)
        self.pushButton_transfer.setObjectName("pushButton_transfer")

        self.horizontalLayout_8.addWidget(self.pushButton_transfer)
        self.horizontalLayoutWidget_6 = QtWidgets.QWidget(self.groupBox_classification)
        self.horizontalLayoutWidget_6.setGeometry(QtCore.QRect(20, 240, 761, 81))
        self.horizontalLayoutWidget_6.setObjectName("horizontalLayoutWidget_6")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_6)
        self.horizontalLayout_9.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")

        self.label_4 = QtWidgets.QLabel(self.horizontalLayoutWidget_6)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_9.addWidget(self.label_4)

        # The transfer learning classification result text browser
        self.textBrowser_transfer = QtWidgets.QTextBrowser(self.horizontalLayoutWidget_6)
        self.textBrowser_transfer.setObjectName("textBrowser_transfer")

        self.horizontalLayout_9.addWidget(self.textBrowser_transfer)
        mainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(mainWindow)
        self.statusbar.setObjectName("statusbar")
        mainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(mainWindow)
        QtCore.QMetaObject.connectSlotsByName(mainWindow)

    def retranslateUi(self, mainWindow):
        _translate = QtCore.QCoreApplication.translate
        mainWindow.setWindowTitle(_translate("mainWindow", "Pneumonia Detector"))
        self.groupBox_file.setTitle(_translate("mainWindow", "File Import and Segmentation"))
        self.pushButton_browse.setText(_translate("mainWindow", "Browse from file"))
        self.comboBox_segmentation.setItemText(0, _translate("mainWindow", "U-Net model"))
        self.comboBox_segmentation.setItemText(1, _translate("mainWindow", "Watershed algorithm"))
        self.pushButton_segmentation.setText(_translate("mainWindow", "Segmentation"))
        self.pushButton_save.setText(_translate("mainWindow", "Save"))
        self.groupBox_classification.setTitle(_translate("mainWindow", "Classification by models"))
        self.label_CNN.setText(_translate("mainWindow", "Simple CNN model"))
        self.pushButton_CNN.setText(_translate("mainWindow", "Run the model"))
        self.label_CNN_classification.setText(_translate("mainWindow", "Classification: "))
        self.label_transfer.setText(_translate("mainWindow", "Transfer Learning model"))
        self.comboBox_transfer.setItemText(0, _translate("mainWindow", "ResNet 50"))
        self.comboBox_transfer.setItemText(1, _translate("mainWindow", "DenseNet 121"))
        self.pushButton_transfer.setText(_translate("mainWindow", "Run the model"))
        self.label_4.setText(_translate("mainWindow", "Classification: "))
import img_rc
