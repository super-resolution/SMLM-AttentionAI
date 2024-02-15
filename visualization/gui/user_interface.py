# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'user_interface.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(-1, -1, 801, 561))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.l_prec_max = QtWidgets.QLabel(self.verticalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.l_prec_max.sizePolicy().hasHeightForWidth())
        self.l_prec_max.setSizePolicy(sizePolicy)
        self.l_prec_max.setObjectName("l_prec_max")
        self.gridLayout.addWidget(self.l_prec_max, 0, 1, 1, 1)
        self.l_frame_max = QtWidgets.QLabel(self.verticalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.l_frame_max.sizePolicy().hasHeightForWidth())
        self.l_frame_max.setSizePolicy(sizePolicy)
        self.l_frame_max.setObjectName("l_frame_max")
        self.gridLayout.addWidget(self.l_frame_max, 3, 1, 1, 1)
        self.l_prec_min = QtWidgets.QLabel(self.verticalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.l_prec_min.sizePolicy().hasHeightForWidth())
        self.l_prec_min.setSizePolicy(sizePolicy)
        self.l_prec_min.setObjectName("l_prec_min")
        self.gridLayout.addWidget(self.l_prec_min, 0, 0, 1, 1)
        self.horizontalSlider = QtWidgets.QSlider(self.verticalLayoutWidget)
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.gridLayout.addWidget(self.horizontalSlider, 2, 1, 1, 1)
        self.horizontalSlider_2 = QtWidgets.QSlider(self.verticalLayoutWidget)
        self.horizontalSlider_2.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_2.setObjectName("horizontalSlider_2")
        self.gridLayout.addWidget(self.horizontalSlider_2, 2, 0, 1, 1)
        self.horizontalSlider_3 = QtWidgets.QSlider(self.verticalLayoutWidget)
        self.horizontalSlider_3.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_3.setObjectName("horizontalSlider_3")
        self.gridLayout.addWidget(self.horizontalSlider_3, 4, 0, 1, 1)
        self.l_frame_min = QtWidgets.QLabel(self.verticalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.l_frame_min.sizePolicy().hasHeightForWidth())
        self.l_frame_min.setSizePolicy(sizePolicy)
        self.l_frame_min.setObjectName("l_frame_min")
        self.gridLayout.addWidget(self.l_frame_min, 3, 0, 1, 1)
        self.horizontalSlider_4 = QtWidgets.QSlider(self.verticalLayoutWidget)
        self.horizontalSlider_4.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_4.setObjectName("horizontalSlider_4")
        self.gridLayout.addWidget(self.horizontalSlider_4, 4, 1, 1, 1)
        self.horizontalSlider_5 = QtWidgets.QSlider(self.verticalLayoutWidget)
        self.horizontalSlider_5.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_5.setObjectName("horizontalSlider_5")
        self.gridLayout.addWidget(self.horizontalSlider_5, 6, 0, 1, 1)
        self.horizontalSlider_6 = QtWidgets.QSlider(self.verticalLayoutWidget)
        self.horizontalSlider_6.setProperty("value", 99)
        self.horizontalSlider_6.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_6.setObjectName("horizontalSlider_6")
        self.gridLayout.addWidget(self.horizontalSlider_6, 6, 1, 1, 1)
        self.l_prob_min = QtWidgets.QLabel(self.verticalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.l_prob_min.sizePolicy().hasHeightForWidth())
        self.l_prob_min.setSizePolicy(sizePolicy)
        self.l_prob_min.setObjectName("l_prob_min")
        self.gridLayout.addWidget(self.l_prob_min, 5, 0, 1, 1)
        self.l_prob_max = QtWidgets.QLabel(self.verticalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.l_prob_max.sizePolicy().hasHeightForWidth())
        self.l_prob_max.setSizePolicy(sizePolicy)
        self.l_prob_max.setObjectName("l_prob_max")
        self.gridLayout.addWidget(self.l_prob_max, 5, 1, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.l_prec_max.setText(_translate("MainWindow", "Precision filter max: "))
        self.l_frame_max.setText(_translate("MainWindow", "Frame filter max:"))
        self.l_prec_min.setText(_translate("MainWindow", "Precision filter min: 0"))
        self.l_frame_min.setText(_translate("MainWindow", "Frame filter min: 0"))
        self.l_prob_min.setText(_translate("MainWindow", "Probability filter min: 0"))
        self.l_prob_max.setText(_translate("MainWindow", "Probability filter max: "))
