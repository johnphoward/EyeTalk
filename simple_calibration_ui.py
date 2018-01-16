from PyQt5.QtWidgets import (QApplication, QGraphicsView,
        QGraphicsPixmapItem, QGraphicsScene, QDesktopWidget)
from PyQt5.QtGui import QPainter, QPixmap
from PyQt5.QtCore import (QObject, QPointF, QTimer,
        QPropertyAnimation, pyqtProperty)
import sys
from GazeDetector import GazeDetector
import pyautogui

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
        self.screen_width = ag.width()
        self.screen_height = ag.height()

        self.ball = Ball()
        self.ball_width = self.ball.pixmap_item.boundingRect().size().width()
        self.ball_height = self.ball.pixmap_item.boundingRect().size().height()

        self.anim = QPropertyAnimation(self.ball, b'pos')
        self.anim.setDuration(10000)
        self.anim.setStartValue(QPointF(0, 0))

        self.anim.setKeyValueAt(0.25, QPointF(self.screen_width - self.ball_width, 0))
        self.anim.setKeyValueAt(0.5, QPointF(self.screen_width - self.ball_width, self.screen_height - self.ball_height))
        self.anim.setKeyValueAt(0.75, QPointF(0, self.screen_height - self.ball_height))

        self.anim.setEndValue(QPointF(0, 0))

        self.scene = QGraphicsScene(self)
        self.scene.setSceneRect(0, 0, self.screen_width, self.screen_height)
        self.scene.addItem(self.ball.pixmap_item)
        self.setScene(self.scene)

        self.setWindowTitle("Calibration")
        self.setRenderHint(QPainter.Antialiasing)
        #self.resize(x, y)
        #self.setGeometry(0, 0, x, y)

        self.anim.start()
        self.showFullScreen()
        self.gaze = GazeDetector()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.sample)
        self.timer.start(1000)

    def sample(self):
        self.gaze.sample()
        print(self.ball.pixmap_item.scenePos().x(), self.ball.pixmap_item.scenePos().y())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())

