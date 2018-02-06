import math
import numpy as np
import atexit
import os
import subprocess as sp
from time import sleep
from ipc_reader import IPCReader


class GazeDetector:
    LEFT_EYE = 0
    RIGHT_EYE = 1

    def __init__(self, external_camera=False):

        # TODO: parameterize so that we can pass in the camera number
        self.cpp_proc = sp.Popen(['{prefix}/eyefinder_cpp/build/eyefinder'.format(prefix=os.getcwd())])

        # clean up IPC at end
        atexit.register(self.cleanup)
        sleep(2)
        self.active = (self.cpp_proc.poll() is not None)

    def sample(self):
        """
        Read an image from the video capture device and determine where the user is looking
        :return: a ndarray of probabilities for each label in the UI
        """

        # TODO: check if seen frame already

        features = self.sample_features()

        # print(features)

        # probabilities = self.calculate_location_probabilities_from_features(features)

        # return probabilities

    def cleanup(self):
        """
        Clean up the semaphore, shared memory, and kill the eye finder process
        """
        with IPCReader() as reader:
            reader.clean()
        self.cpp_proc.kill()

    @staticmethod
    def sample_features():
        """
        Read an image from the video capture device and return the extracted features of the image
        :return: a ndarray of shape(30) full of numerical features
        """
        with IPCReader() as reader:
            return np.asarray(reader.read())

    def calculate_location_probabilities_from_features(self, features):
        """
        Feed features through a machine learning algorithm to get probabilities
        :param features: a ndarray of shape(30) full of numerical features
        :return: a ndarray of probabilities for each label in the UI
        """
        raise NotImplementedError

    def train_location_classifier(self, data):
        """
        Train location classifier using data
        :param data: a ndarray of shape(N, 30) of N rows of numerical features
        """
        raise NotImplementedError


if __name__ == '__main__':
    tracker = GazeDetector()
    sleep(3)
    for i in range(100):
        sleep(0.05)
        tracker.sample()
