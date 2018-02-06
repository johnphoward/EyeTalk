#include "eyefinder.h"

// ***** ***** ***** ***** ***** ***** ***** ***** ***** ***** ***** ***** *****
// *****
// ***** ***** ***** ***** ***** ***** ***** ***** ***** ***** ***** ***** *****
// *****
/*
PUBLIC
*/

_EF_::EyeFinder::EyeFinder(void) : cap(0) {
  // set the buffersize of cv::VideoCapture -> cap
  cap.set(CV_CAP_PROP_BUFFERSIZE, 3);

  // Initialize the Semaphore (POSIX semaphore)
  sem = sem_open(sem_name, O_CREAT | O_EXCL, 0666, 1);
  if (sem == SEM_FAILED) {
    std::cout << "*** *** sem_open failed!!!" << std::endl;
    std::cout << sem_name << ", errno: " << errno << std::endl;
    sem = sem_open(sem_name, O_CREAT, 0666, 1);
    sem_close(sem);
    exit(1);
  } else {
    printf("sem: %p\n", sem);
  }

  // Initialize the Shared Memory (System V shared Memory)
  // Setup shared memory
  if ((shmid = shmget(key, shared_size, IPC_EXCL | IPC_CREAT | 0666)) < 0) {
    std::cout << "*** *** shmget() failed!!! " << errno << std::endl;
    exit(1);
  } else { // GOLDEN!!!
    std::cout << "shmid: " << shmid << std::endl;
  }
  // Attached shared memory
  if ((shared_memory = (char *)shmat(shmid, NULL, 0)) == (char *)-1) {
    printf("Error attaching shared memory id");
    exit(1);
  } else { // GOLDEN!!!
    std::cout << "shared_memory: " << (unsigned long)shared_memory << std::endl;
  }
}
_EF_::EyeFinder::~EyeFinder(void) {
  // Free the Semaphore
  sem_unlink(sem_name);
  sem_close(sem);
  // Free the Shared Memory
  shmdt(shared_memory);
  shmctl(shmid, IPC_RMID, NULL);

  // Bye message
  std::cout << "\n\nBye~~~!\n" << std::endl;
}

// *****
// start
int _EF_::EyeFinder::start(void) {
  try {
    std::chrono::steady_clock::time_point begin, end;

    if (!cap.isOpened()) {
      std::cerr << "Unable to connect to camera" << std::endl;
      return 1;
    }

    // Load face detection and pose estimation models.
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    dlib::shape_predictor pose_model;
    dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

    // Grab and process frames until the main window is closed by the user.
#if DEBUG
    int i = 0;
    cv::namedWindow("LALALA", cv::WINDOW_AUTOSIZE);
    while (i++ < 100) {
#else
    while (true) {
#endif
      // Grab a frame
      cv::Mat temp;
      if (!cap.read(temp)) {
        break;
      }

      // Shrink the frame size
      cv::resize(temp, temp, cv::Size(), 0.5, 0.5);
      cv::cvtColor(temp, temp, CV_BGR2GRAY);

      dlib::cv_image<dlib::uint8> cimg(temp);

      // Detect faces
      std::vector<dlib::rectangle> faces = detector(cimg);

      // Find the pose of each face. (Only the first)
      std::vector<dlib::full_object_detection> shapes;
      for (unsigned long i = 0; i < 1 && i < faces.size(); ++i)
        shapes.push_back(pose_model(cimg, faces[i]));

      // guarantee that there are at least
      if (shapes.size()) {

        std::vector<long> facial_features_vec;

        // Left eye + Right eye points
        preCalculationPoints(shapes, facial_features_vec);

        // Find the face angle + pupils
        calculateFaceAngles(shapes, facial_features_vec);

        // calculatePupils(roi_l_mat, facial_features_vec);
        // calculatePupils(roi_r_mat, facial_features_vec);
        // Calling EyeLike's now
        calculatePupilsEL(shapes, facial_features_vec, temp);

        // Write out the Facial Features
        writeFacialFeaturesToShm(facial_features_vec);
      } else {
        writeBadFacialFeaturesToShm(); // or skip it
      }
    }
  } catch (dlib::serialization_error &e) {
    std::cout << "You need dlib's default face landmarking model file to run "
                 "this example."
              << std::endl;
    std::cout << "You can get it from the following URL: " << std::endl;
    std::cout
        << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        << std::endl;
    std::cout << std::endl << e.what() << std::endl;
  } catch (std::exception &e) {
    std::cout << e.what() << std::endl;
  }
  return 0;
}

// ***** ***** ***** ***** ***** ***** ***** ***** ***** ***** ***** ***** *****
// *****
// ***** ***** ***** ***** ***** ***** ***** ***** ***** ***** ***** ***** *****
// *****
/*
PRIVATE
*/

// *****
// setMinAndMax
std::tuple<long, long, long, long> _EF_::EyeFinder::setMinAndMax(
    int start, int end,
    const std::vector<dlib::full_object_detection> &shapes) {
  // min_x, min_y, max_x, max_y
  std::tuple<long, long, long, long> tp{LONG_MAX, LONG_MAX, LONG_MIN, LONG_MIN};

  for (int i = start; i <= end; ++i) {
    auto x = shapes[0].part(i).x();
    auto y = shapes[0].part(i).y();
    std::get<0>(tp) = std::min(std::get<0>(tp), x);
    std::get<1>(tp) = std::min(std::get<1>(tp), y);
    std::get<2>(tp) = std::max(std::get<2>(tp), x);
    std::get<3>(tp) = std::max(std::get<3>(tp), y);
  }

  return tp;
}

// *****
// getROI
cv::Rect _EF_::EyeFinder::getROI(std::tuple<long, long, long, long> &tp,
                                 cv::Mat frame) {
#if DEBUG
  std::cout << "_EF_::EyeFinder::getROI" << std::get<0>(tp) << " "
            << std::get<1>(tp) << " " << std::get<2>(tp) << " "
            << std::get<3>(tp) << std::endl;
#endif
  auto start_x = std::max(std::get<0>(tp) - 10, long(0));
  auto start_y = std::max(std::get<1>(tp) - 10, long(0));
  auto size_x =
      std::min(std::get<2>(tp) - std::get<0>(tp) + 15, frame.cols - start_x);
  auto size_y =
      std::min(std::get<3>(tp) - std::get<1>(tp) + 15, frame.rows - start_y);
  return cv::Rect(start_x, start_y, size_x, size_y);
}

// ***** ***** ***** ***** ***** ***** ***** ***** ***** ***** ***** ***** *****
// *****
//
// Calculation!!! (finding face angles, pupils, and writing out to shared
// memory)
//
// ***** ***** ***** ***** ***** ***** ***** ***** ***** ***** ***** ***** *****
// *****

// *****
// preCalculationPoints
//  0-11 = x, y coordinates of left eye points
//  12-23 = x, y coordinates of right eye points
void _EF_::EyeFinder::preCalculationPoints(
    const std::vector<dlib::full_object_detection> &shapes,
    std::vector<long> &facial_features_vec) {

  for (int i = 36; i <= 47; ++i) {
    auto shp = shapes[0].part(i);
    facial_features_vec.push_back(shp.x());
    facial_features_vec.push_back(shp.y());
  }
}

// *****
// calculateFaceAngles()
void _EF_::EyeFinder::calculateFaceAngles(
    const std::vector<dlib::full_object_detection> &shapes,
    std::vector<long> &facial_features_vec) {
  auto left_cheek = shapes[0].part(1);
  auto right_cheek = shapes[0].part(16);
  auto nose_center = shapes[0].part(33);

  float face_width = right_cheek.x() - left_cheek.x();
  float nose_length = face_width * 0.25;

  float true_center_nose_x = left_cheek.x() + face_width / 2.0;
  float nose_x_offset = nose_center.x() - true_center_nose_x;

  float theta = 0.0;
  if (nose_x_offset != 0.0) {
    float angle = atan(nose_length / std::abs(nose_x_offset));

    theta = (90 - angle * (180.0 / M_PI));
    if (signbit(nose_x_offset))
      theta *= -1;
#if DEBUG
    std::cout << "angle: " << angle << std::endl;
    std::cout << "theta: " << theta << std::endl;
#endif
  }

  float vertical_cheek_offset = right_cheek.y() - left_cheek.y();
  float angle = atan(std::abs(vertical_cheek_offset) / face_width);
  float alpha = angle * (180.00 / M_PI);
  if (signbit(vertical_cheek_offset))
    alpha *= -1;

  // TODO: handle types for theta and alpha
  facial_features_vec.push_back(theta);
  facial_features_vec.push_back(alpha);
}

// *****
// calculatePupils() a.k.a. Timm-Barth Algorithm
void _EF_::EyeFinder::calculatePupils(cv::Mat src,
                                      std::vector<long> &facial_features_vec) {

  cv::Mat src_blur, src_blur_inv;
  int scale = 1;
  int delta = 0;
  int ddepth = CV_64F;

  cv::GaussianBlur(src, src_blur, cv::Size(5, 5), 0, 0, cv::BORDER_DEFAULT);
  src_blur_inv =
      cv::Mat::ones(src_blur.size(), src_blur.type()) * 255 - src_blur;

  cv::Mat grad_x, grad_y;
  cv::Sobel(src, grad_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT);
  cv::Sobel(src, grad_y, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT);

  cv::Mat mag(grad_x.size(), grad_x.type());
  cv::magnitude(grad_x, grad_y, mag);
  cv::divide(grad_x, mag, grad_x);
  cv::divide(grad_y, mag, grad_y);

  cv::Mat pixel_combination_x(src_blur_inv.size(), CV_16S);
  cv::Mat pixel_combination_y(src_blur_inv.size(), CV_16S);

  int row_LIMIT = src_blur_inv.rows;
  int col_LIMIT = src_blur_inv.cols;

  double max_objective = -777777.777;
  int max_r = 0, max_c = 0;

  for (int pixel_r = 10; pixel_r < row_LIMIT - 10; ++pixel_r) {
    for (int pixel_c = 10; pixel_c < col_LIMIT - 10; ++pixel_c) {
      double obj_val = 0.0;
      for (int i = 0; i < row_LIMIT; ++i) {
        cv::Vec3b *r_grad_x = grad_x.ptr<cv::Vec3b>(i);
        cv::Vec3b *r_grad_y = grad_y.ptr<cv::Vec3b>(i);
        cv::Vec3b *r_src_gray_blur_inv = src_blur_inv.ptr<cv::Vec3b>(i);
        for (int j = 0; j < col_LIMIT; ++j) {
          int x_val = i - pixel_r;
          int y_val = j - pixel_c;
          if ((x_val | y_val) == 0)
            break;
          double magnitude = sqrt(x_val * x_val + y_val * y_val);
          double unit_x = x_val / magnitude;
          double unit_y = y_val / magnitude;

          double dot = (unit_x * r_grad_x[j][0] + unit_y * r_grad_y[j][0]);
          double dot2 = dot * dot;

          double pixel_val = r_src_gray_blur_inv[j][0] * dot2;
          obj_val += pixel_val;
        }
      }

      if (obj_val > max_objective) {
        max_r = pixel_r;
        max_c = pixel_c;
        max_objective = obj_val;
      }
    }
  }

  facial_features_vec.push_back((long)max_r);
  facial_features_vec.push_back((long)max_c);
}

// *****
// calculatePupilsEL() a.k.a. Timm-Barth Algorithm using EyeLike
void _EF_::EyeFinder::calculatePupilsEL(
    const std::vector<dlib::full_object_detection> &shapes,
    std::vector<long> &facial_features_vec, cv::Mat temp) {

  // left eye + right eye
  std::tuple<long, long, long, long> l_tp =
      EyeFinder::setMinAndMax(36, 41, shapes);
  std::tuple<long, long, long, long> r_tp =
      EyeFinder::setMinAndMax(42, 47, shapes);
  std::tuple<long, long, long, long> face_tp =
      EyeFinder::setMinAndMax(1, 16, shapes);

  // ROI for left eye + right eye
  cv::Rect roi_l = EyeFinder::getROI(l_tp, temp);
  cv::Rect roi_r = EyeFinder::getROI(r_tp, temp);
  cv::Rect roi_face = EyeFinder::getROI(face_tp, temp);

  // Display eye tracking on the screen
  cv::Mat roi_l_mat, roi_r_mat, roi_face_mat;
  roi_l_mat = temp(roi_l);
  roi_r_mat = temp(roi_r);
  roi_face_mat = temp(roi_face);

  cv::Mat temp_clone = temp.clone();
  cv::GaussianBlur(temp_clone, temp_clone, cv::Size(0, 0),
                   0.005 * roi_face.width);
  cv::Point leftPupil = findEyeCenter(temp_clone, roi_l, "Left Eye");
  cv::Point rightPupil = findEyeCenter(temp_clone, roi_r, "Right Eye");

  int real_leftPupil_x = leftPupil.x + roi_l.x;
  int real_leftPupil_y = leftPupil.y + roi_l.y;
  int real_rightPupil_x = rightPupil.x + roi_r.x;
  int real_rightPupil_y = rightPupil.y + roi_r.y;

  facial_features_vec.push_back((long)real_leftPupil_x);
  facial_features_vec.push_back((long)real_leftPupil_y);
  facial_features_vec.push_back((long)real_rightPupil_x);
  facial_features_vec.push_back((long)real_rightPupil_y);

#if DEBUG_TB
  cv::circle(temp, cv::Point(real_leftPupil_x, real_leftPupil_y), 1,
             cv::Scalar(255, 255, 255, 255));
  cv::circle(temp, cv::Point(real_rightPupil_x, real_rightPupil_y), 1,
             cv::Scalar(255, 255, 255, 255));
  // cv::resize(temp, temp, cv::Size(300,300));
  cv::imshow("LALALA", temp);
  cv::waitKey(1);
#endif
}

// *****
// writeFacialFeaturesToShm()
//  Will be to shared memory with semaphore synchronization
//  0-11 = x, y coordinates of left eye points
//  12-23 = x, y coordinates of right eye points
//  24, 25 = x, y coordinates of left eye center (timm)
//  26, 27 = x, y coordinates of right eye center (timm)
//  28-29 = face angles (another code)
void _EF_::EyeFinder::writeFacialFeaturesToShm(
    const std::vector<long> &facial_features_vec) {
  int i = 0;
  sem_wait(sem);

  // add an ID to the frame first
  memcpy(shared_memory + sizeof(long) * i, &frame_id, sizeof(long));
  i = 1;

  // now copy every value in vec over
  for (const auto num : facial_features_vec) {
    memcpy(shared_memory + sizeof(long) * i, &num, sizeof(long));
    i++;
  }

  // TODO: check if implementation has to change to represent face angles as
  // floats

  sem_post(sem);
  frame_id = (frame_id + 1) % 100;
}

void _EF_::EyeFinder::writeBadFacialFeaturesToShm(void) {}
