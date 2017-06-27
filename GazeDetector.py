import cv2
import dlib
import numpy as np
from itertools import chain


class GazeDetector:
    LEFT_EYE = 0
    RIGHT_EYE = 1

    def __init__(self):
        self.video_feed = cv2.VideoCapture(0)

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    def sample(self):
        """
        Read an image from the video capture device and determine where the user is looking
        :return: a ndarray of probabilities for each label in the UI
        """
        read_success, frame = self.video_feed.read()

        img = self.preprocess_image(frame)

        features = self.extract_features(img)

        # probabilities = self.calculate_location_probabilities_from_features(features)

        # return probabilities

    def sample_features(self):
        """
        Read an image from the video capture device and return the extracted features of the image
        :return: a ndarray of shape(30) full of numerical features
        """
        read_success, frame = self.video_feed.read()

        img = self.preprocess_image(frame)

        return self.extract_features(img)

    def extract_features(self, img):
        """
        Given an image, find all of the features that we need to identify the location of the user's gaze
        :param img: an already-preprocessed image in the form of an ndarray of shape (l, w)
        :return: a ndarray of shape(30) full of numerical features
        """
        rectangles = self.detector(img)
        rect = self._select_main_face_rectangle(rectangles)

        facial_points = self.predictor(img, rect)

        img_features = self._build_feature_vector_from_points(facial_points)
        eye_points = img_features[:24]

        left_eye_img = self._get_eye_image(img, eye_points, self.LEFT_EYE)
        # img_features[24:26] = self.detect_eye_center(left_eye_img)
        cv2.imwrite('left_eye.png', left_eye_img)

        right_eye_img = self._get_eye_image(img, eye_points, self.RIGHT_EYE)
        # img_features[26:28] = self.detect_eye_center(right_eye_img)
        cv2.imwrite('right_eye.png', right_eye_img)

        all_points = list(facial_points.parts())

        for p in all_points:
            cv2.circle(img, (p.x, p.y), 1, 1, 2)

        cv2.imwrite('save_frame.png', img)

        return img_features

    @staticmethod
    def _get_eye_image(image, eye_coordinates, side):
        """
        Get the relevant pixels of the user's eyes from an image of the full face
        :param image: an image in the form of an ndarray of shape (l, w)
        :param eye_coordinates: a list of coordinates of alternating x and y of points from the user's eye
        :param side: constant LEFT_EYE or RIGHT_EYE (0 or 1) referring to which eye you are finding
        :return: an image in the form of a 2D ndarray of just the pixels around the user's eye
        """
        first_index = side * 12
        end_index = first_index + 12

        x_vals = eye_coordinates[first_index: end_index: 2].astype(int)
        x_min = x_vals.min()
        x_max = x_vals.max()

        y_vals = eye_coordinates[first_index + 1: end_index: 2].astype(int)
        y_min = y_vals.min() - 2
        y_max = y_vals.max() + 2

        return image[y_min: y_max, x_min: x_max]

    @staticmethod
    def _build_feature_vector_from_points(face_points):
        """
        Start building vector with following format:
        0-11 = x, y coordinates of left eye points
        12-23 = x, y coordinates of right eye points
        24, 25 = x, y coordinates of left eye center (will be zero in this array)
        26, 27 = x, y coordinates of right eye center (will be zero in this array)
        28-29 = face angles (will be zero in this array)
        :param face_points: dlib full_object_detection object
        :return: np ndarray of size 30 with the above format
        """
        vector = np.zeros(30)
        vector[0:24] = list(chain(*[(pt.x, pt.y) for pt in [face_points.part(n) for n in range(36, 48)]]))
        return vector

    @staticmethod
    def _select_main_face_rectangle(rectangles):
        """
        Determine which face to read from - will need to identify user from possibly multiple rectangles
        :param rectangles: a dlib rectangles object 
        :return: a single dlib rectangle object
        """
        # TODO: Make smarter
        return next(rect for rect in rectangles)

    @staticmethod
    def preprocess_image(image):
        """
        Carry out any image preprocessing necessary. Convert to grayscale
        :param image: image in form of ndarray of shape (l, w, 3)
        :return: image in form of ndarray of shape (l, w)
        """
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return grayscale

    def detect_eye_center(self, eye_image):
        """
        Given an image of an eye, identify the center point of the pupil using Timm-Barth algorithm
        :param eye_image: a 2D ndarray image of a user's eye
        :return: an x and y coordinate for the center of the eye in list form
        """
        raise NotImplementedError

    def calculate_location_probabilities_from_features(self, features):
        """
        Feed features through a machine learning algorithm to get probabilities
        :param features: a ndarray of shape(30) full of numerical features
        :return: a ndarray of probabilities for each label in the UI
        """
        raise NotImplementedError

    def train_location_classifier(self, data):

        raise NotImplementedError


if __name__ == '__main__':
    tracker = GazeDetector()
    for i in range(10):
        tracker.sample()
