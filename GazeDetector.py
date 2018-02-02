import cv2
import dlib
import math
import numpy as np
from itertools import chain, product
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD


class GazeDetector:
    LEFT_EYE = 0
    RIGHT_EYE = 1

    def __init__(self, external_camera=False):
        self.video_feed = cv2.VideoCapture(int(external_camera))

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

        self.neural_network = None
        self.init_model()

    def init_model(self):
        model = Sequential()
        model.add(Dense(20, input_dim=30, kernel_initializer='uniform', activation='relu'))
        model.add(Dense(20, kernel_initializer='uniform', activation='relu'))
        model.add(Activation("softmax"))
        model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01))
        self.neural_network = model

    def sample(self):
        """
        Read an image from the video capture device and determine where the user is looking
        :return: a ndarray of probabilities for each label in the UI
        """
        read_success, frame = self.video_feed.read()

        img = self.preprocess_image(frame)

        features = self.extract_features(img)

        probabilities = self.calculate_location_probabilities_from_features(features)

        return probabilities

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

        left_eye_img, left_eye_corner = self._get_eye_image(img, eye_points, self.LEFT_EYE)
        img_features[24:26] = self._detect_eye_center(left_eye_img, left_eye_corner, debug=True, debug_side='left')

        right_eye_img, right_eye_corner = self._get_eye_image(img, eye_points, self.RIGHT_EYE)
        img_features[26:28] = self._detect_eye_center(right_eye_img, right_eye_corner, debug=True, debug_side='right')

        img_features[28:] = self._calculate_face_angles(facial_points)

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
        :return: an image as an 2D ndarray of just the pixels around the user's eye, (x, y) tuple of corner of eye
        """
        first_index = side * 12
        end_index = first_index + 12

        x_vals = eye_coordinates[first_index: end_index: 2].astype(int)
        x_min = x_vals.min()
        x_max = x_vals.max()

        y_vals = eye_coordinates[first_index + 1: end_index: 2].astype(int)
        y_min = y_vals.min() - 2
        y_max = y_vals.max() + 2

        return image[y_min: y_max, x_min: x_max], (x_min, y_min)

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

    def _detect_eye_center(self, eye_image, eye_corner, debug=False, debug_side='left'):
        """
        Given an image of an eye, identify the center point of the pupil using Timm-Barth algorithm
        :param eye_image: a 2D ndarray image of a user's eye
        :return: an x and y coordinate for the center of the eye in list form
        """
        img_deriv_wrt_x = cv2.Sobel(eye_image, cv2.CV_64F, 1, 0, ksize=3)
        img_deriv_wrt_y = cv2.Sobel(eye_image, cv2.CV_64F, 0, 1, ksize=3)

        gradient_mags = cv2.magnitude(img_deriv_wrt_x, img_deriv_wrt_y)
        unit_gradient_x = np.nan_to_num(img_deriv_wrt_x / gradient_mags)
        unit_gradient_y = np.nan_to_num(img_deriv_wrt_y / gradient_mags)

        blurred = cv2.GaussianBlur(eye_image, (5, 5), 0, 0)
        pixel_weights = np.invert(blurred)
        image_rows, image_cols = eye_image.shape

        objective_values = np.zeros((image_rows, image_cols))

        for x_i, y_i in product(range(image_rows), range(image_cols)):
            row_values = np.arange(image_cols * 1.0) - y_i
            col_values = np.arange(image_rows * 1.0) - x_i
            displacement_x, displacement_y = np.meshgrid(row_values, col_values)
            disp_mags = cv2.magnitude(displacement_x, displacement_y)
            unit_disp_x = np.nan_to_num(displacement_x / disp_mags)
            unit_disp_y = np.nan_to_num(displacement_y / disp_mags)

            pixel_objective = pixel_weights * np.square(unit_disp_x * unit_gradient_x + unit_disp_y * unit_gradient_y)
            objective = np.sum(pixel_objective) / eye_image.size
            objective_values[x_i, y_i] = objective

        # set edges to 0 to disqualify them from consideration (function is always positive)
        objective_values[0, :] = 0
        objective_values[-1, :] = 0
        objective_values[:, 0] = 0
        objective_values[0, -1] = 0

        max_locs = np.unravel_index(objective_values.argmax(), objective_values.shape)

        if debug:
            # put red dot in center of eye and save to file
            new_eye = cv2.cvtColor(eye_image, cv2.COLOR_GRAY2RGB)
            new_eye[max_locs[0]][max_locs[1]] = [0, 0, 255]
            cv2.imwrite('output_{side}.png'.format(side=debug_side), new_eye)

        center_pixels = tuple(map(lambda x, y: x + y, max_locs, eye_corner))

        return center_pixels

    def calculate_location_probabilities_from_features(self, features):
        """
        Feed features through a machine learning algorithm to get probabilities
        :param features: a ndarray of shape(30) full of numerical features
        :return: a ndarray of probabilities for each label in the UI
        """

        prediction_vals = self.neural_network.predict(features)
        return prediction_vals

    def train_location_classifier(self, data, num_epochs=128):
        """
        Train location classifier using data
        :param data: a ndarray of shape(N, 30) of N rows of numerical features
        :param num_epochs: an integer number for how many iterations through all the data the training will do
        """

        # split data into X and Y (data and labels)

        training_data = None
        training_labels = None

        self.neural_network.fit(training_data, training_labels, num_epochs, batch_size=128)


if __name__ == '__main__':
    tracker = GazeDetector()
    for i in range(10):
        tracker.sample()
