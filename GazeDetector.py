import math
import numpy as np
import atexit
import os
import subprocess as sp
from time import sleep
from eyefinderreader import EyeFinderReader


class GazeDetector:
    LEFT_EYE = 0
    RIGHT_EYE = 1

    def __init__(self, external_camera=False):

        # TODO: parameterize so that we can pass in the camera number
        self.cpp_proc = sp.Popen(['{prefix}/eyefinder_cpp/build/eyefinder'.format(prefix=os.getcwd())])

        # clean up IPC at end
        atexit.register(self.cleanup)

    def sample(self):
        """
        Read an image from the video capture device and determine where the user is looking
        :return: a ndarray of probabilities for each label in the UI
        """

        features = self.sample_features()

        print(features)

        # probabilities = self.calculate_location_probabilities_from_features(features)

        # return probabilities

    def cleanup(self):
        """
        Clean up the semaphore, shared memory, and kill the eye finder process
        """
        with EyeFinderReader() as reader:
            reader.clean()
        self.cpp_proc.kill()

    @staticmethod
    def sample_features():
        """
        Read an image from the video capture device and return the extracted features of the image
        :return: a ndarray of shape(30) full of numerical features
        """
        with EyeFinderReader() as reader:
            return np.asarray(reader.read())

    @staticmethod
    def _calculate_face_angles(face_points):
        """
        Calculate the angles of the face from the 68 points
        theta = rotation about vertical axis through center of head
            - assume nose length is constant/proportional to face width
        alpha = rotation about x axis running front to back through face (side-to-side head tilt)
        phi = rotation about y axis running horizontally through sides of head (as in nodding)
        :param face_points: dlib full_object_detection object
        :return: list of 2 facial angles
        """

        # first calculate theta
        left_cheek = face_points.part(1)
        right_cheek = face_points.part(16)
        nose_center = face_points.part(33)

        face_width = right_cheek.x - left_cheek.x
        nose_length = face_width / 4.0

        true_center_nose_x = left_cheek.x + face_width / 2.0
        nose_x_offset = nose_center.x - true_center_nose_x
        try:
            angle = math.atan(nose_length / abs(nose_x_offset))
            theta = (90 - math.degrees(angle)) * np.sign(nose_x_offset)
        except ZeroDivisionError:
            theta = 0.0

        # now calculate alpha
        vertical_cheek_offset = right_cheek.y - left_cheek.y
        angle = math.atan(abs(vertical_cheek_offset) / face_width)
        alpha = math.degrees(angle) * np.sign(vertical_cheek_offset)

        # TODO: determine how to calculate phi based on the points and expand vector to 31

        return [theta, alpha]

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
