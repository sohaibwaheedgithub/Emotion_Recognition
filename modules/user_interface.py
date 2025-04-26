# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'user_interface.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1120, 845)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.video_feed = QtWidgets.QLabel(self.centralwidget)
        self.video_feed.setGeometry(QtCore.QRect(20, 20, 1081, 591))
        self.video_feed.setText("")
        self.video_feed.setScaledContents(True)
        self.video_feed.setAlignment(QtCore.Qt.AlignCenter)
        self.video_feed.setObjectName("video_feed")
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(450, 680, 191, 51))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.comboBox.setFont(font)
        self.comboBox.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")

        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(360, 630, 361, 51))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setScaledContents(True)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1120, 26))
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
        self.comboBox.setItemText(0, _translate("MainWindow", "Angry"))
        self.comboBox.setItemText(1, _translate("MainWindow", "Happy"))
        self.comboBox.setItemText(2, _translate("MainWindow", "Sad"))
        self.comboBox.setItemText(3, _translate("MainWindow", "Surprise"))
        self.comboBox.setItemText(4, _translate("MainWindow", "Multiclass"))

        self.label.setText(_translate("MainWindow", "Choose Emotion"))


if __name__ == "__main__":
    import sys
    import cv2
    from emotion_recognizer import Emotion_Recognizer

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()

    
    emotion_recognizer = Emotion_Recognizer()

    cap = cv2.VideoCapture(r'test_videos\test_videos4.mp4')
    
    
    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    e = 'angry'
    writer = cv2.VideoWriter(
        f'saved_videos/{e}_demo4.mp4',
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (480, 848)
    )

    while cap.isOpened():
        success, image = cap.read()
        if success:
            image = emotion_recognizer.detection_preprocessing(image)
            image = emotion_recognizer.recognize_emotion(image, ui.comboBox.currentText().lower())
            #image = emotion_recognizer.recognize_emotion(image, e)
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #image = cv2.resize(image, (480, 848))

            h, w, c = image.shape
            bytesPerLine = 3 * w
            qImg = QtGui.QImage(image.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
            ui.video_feed.setPixmap(QtGui.QPixmap(qImg))

            #cv2.imshow('', image)
            #writer.write(image)
            
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        else:
            break
    
    cap.release()
    #writer.release()
    cv2.destroyAllWindows()


    sys.exit(app.exec_()) 

