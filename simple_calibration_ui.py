from PyQt5.QtWidgets import (QApplication, QGraphicsView,
        QGraphicsPixmapItem, QGraphicsScene, QDesktopWidget)
from PyQt5.QtGui import QPainter, QPixmap
from PyQt5.QtCore import (QObject, QPointF,
        QPropertyAnimation, pyqtProperty)
import sys
import cv2
import collections
import math
import pyautogui
import os
import time

class Ball(QObject):
    def __init__(self):
        super().__init__()

        self.pixmap_item = QGraphicsPixmapItem(QPixmap("red_ball.png"))

    def _set_pos(self, pos):
        self.pixmap_item.setPos(pos)

    pos = pyqtProperty(QPointF, fset=_set_pos)


class Example(QGraphicsView):
    def __init__(self):
        super().__init__()

        self.initView()

    def initView(self):
        ag = QDesktopWidget().availableGeometry()
        sg = QDesktopWidget().screenGeometry()

        widget = self.geometry()
        x = ag.width()# - widget.width()
        y = ag.height()# - sg.height() - widget.height()
        print("Available: ", ag.width(), ag.height())
        print("Screen: ", sg.width(), sg.height())
        print("Widget: ", widget.width(), widget.height())

        screen = app.primaryScreen()
        size = screen.size()
        rect = screen.availableGeometry()
        self.resize(rect.width(), rect.height())
        self.setGeometry(0, 0, rect.width(), rect.height())

        self.ball = Ball()

        self.anim = QPropertyAnimation(self.ball, b'pos')
        self.anim.setDuration(20000)
        self.anim.setStartValue(QPointF(0, 0))

        self.anim.setKeyValueAt(0.2, QPointF(x / 2, -y / 2))
        self.anim.setKeyValueAt(0.4, QPointF(x / 2, y / 2))
        self.anim.setKeyValueAt(0.6, QPointF(-x / 2, y / 2))
        self.anim.setKeyValueAt(0.8, QPointF(-x / 2, -y / 2))

        self.anim.setEndValue(QPointF(0, 0))

        self.scene = QGraphicsScene(self)
        self.scene.setSceneRect(0, 0, 150, 150)
        self.scene.addItem(self.ball.pixmap_item)
        self.setScene(self.scene)

        self.setWindowTitle("Calibration")
        self.setRenderHint(QPainter.Antialiasing)
        screen = app.primaryScreen()
        size = screen.size()
        rect = screen.availableGeometry()
        self.resize(rect.width(), rect.height())
        self.setGeometry(0, 0, rect.width(), rect.height())

        self.anim.start()
        self.showFullScreen()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())

