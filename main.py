import sys
from PyQt5.Qt import QWidget, QPushButton, QTableWidget, QLabel, QTextEdit, QVBoxLayout, QFileDialog, QApplication, QTableWidgetItem
from PyQt5 import QtWidgets
import pandas as pd
import numpy as np
import qdarkgraystyle
import matplotlib.pyplot as plt
import os
class GUIapp(QWidget):
    def __init__(self, *args, **kwargs):
        super(GUIapp, self).__init__(*args, **kwargs)
        self.setWindowTitle("TEST 1.0")
        self.resize(800, 600)
        self.button1 = QPushButton("Input File")
        self.button1.clicked.connect(self.getFile)
        self.label1 = QLabel("Prediction")

        self.table = QTableWidget()

        self.label2 = QLabel("Insert CSV File")
        self.textEditor = QTextEdit()



        #self.setCentralWidget(self.graphWidget)


        #https: // stackoverflow.com / questions / 31775468 / show - string - values - on - x - axis - in -pyqtgraph
        # plot data: x, y values

        layout = QVBoxLayout()
        layout.addWidget(self.button1)
        layout.addWidget(self.label2)
        layout.addWidget(self.textEditor)
        layout.addWidget(self.label1)
        layout.addWidget(self.table)


        self.setLayout(layout)



    def getFile(self):

        dialg = QFileDialog()
        dialg.setFileMode(QFileDialog.AnyFile)
        dialg.setNameFilter("CSV Files (*.csv)")
        if dialg.exec_():
            fileName = dialg.selectedFiles()
            f = open(fileName[0], 'r')
            ml = machineLearning(f)



            with open(fileName[0], 'r') as f:
                data = f.read()
                self.textEditor.setPlainText(data)
                self.table.setRowCount(len(ml))
                self.table.setColumnCount(1)
                x = -1

                while x <= len(ml) - 2:
                    x += 1
                    self.table.setItem(x, 0, QTableWidgetItem(ml[x]))
            header = self.table.horizontalHeader()
            header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)



            f.close()
        else:
            pass

def machineLearning(file):
    dft = pd.read_csv(file)

    dft['00'] = 1
    print(dft)


    X_train = np.array(dft.iloc[:, [-1, -4]])
    y_train = np.array(dft.iloc[:, -3])
    X_test = np.array(dft.iloc[:, [-1, -3]])
    X_plot = np.array(dft.iloc[:, -3])
    y_test = np.array(dft.iloc[:, -2])
    x_transpose = np.transpose(X_train)
    x_transpose_dot_x = x_transpose.dot(X_train)
    var1 = np.linalg.inv(x_transpose_dot_x)

    var2 = x_transpose.dot(y_train)
    theta = var1.dot(var2)
    prediction = np.dot(X_test, theta)
    arrList = prediction.tolist()
    R2_score = 1 - sum((prediction - y_test) ** 2) / sum((y_test - np.mean(y_test)) ** 2)
    results = [str(i) for i in arrList]
    plt.style.use('dark_background')
    plt.plot(X_plot, y_test, 'o')
    plt.plot(X_plot, prediction)
    plt.show()
    print(R2_score)
    return results




if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkgraystyle.load_stylesheet())
    ex = GUIapp()
    ex.show()
    sys.exit(app.exec_())