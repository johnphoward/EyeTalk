// The contents of this file are in the public domain. See
// LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This example program shows how to find frontal human faces in an image and
    estimate their pose.  The pose takes the form of 68 landmarks.  These are
    points on the face such as the corners of the mouth, along the eyebrows, on
    the eyes, and so forth.


    This example is essentially just a version of the
   face_landmark_detection_ex.cpp example modified to use OpenCV's VideoCapture
   object to read from a camera instead of files.


    Finally, note that the face detector is fastest when compiled with at least
    SSE2 instructions enabled.  So if you are using a PC with an Intel or AMD
    chip then you should enable at least SSE2 instructions.  If you are using
    cmake to compile this program you can enable them by using one of the
    following commands when you create the build project:
        cmake path_to_dlib_root/examples -DUSE_SSE2_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_SSE4_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_AVX_INSTRUCTIONS=ON
    This will set the appropriate compiler options for GCC, clang, Visual
    Studio, or the Intel compiler.  If you are using another compiler then you
    need to consult your compiler's manual to determine how to enable these
    instructions.  Note that AVX is the fastest but requires a CPU from at least
    2011.  SSE4 is the next fastest and is supported by most current machines.
*/

#include "eyefinder.h"

/*
Art by Herman Hiddema
            _
           / )
        ,-(,' ,---.
       (,-.\,' `  _)-._
          ,' `(_)_)  ,-`--.
         /          (      )
        /            `-.,-'|
       /                |  /
       |               ,^ /
      /                   |
      |                   /
     /       PUBLIC      /
     |                   |
     |                   |
    /                    \
  ,.|                    |
(`\ |                    |
(\  |   --.      /  \_   |
 (__(   ___)-.   | '' )  /)
hh   `---...\\\--(__))/-'-'
*/

_EF_::EyeFinder::EyeFinder(void) : cap(0) {
  // set the buffersize of cv::VideoCapture -> cap
  cap.set(CV_CAP_PROP_BUFFERSIZE, 3);

  // Initialize the Semaphore (POSIX semaphore)
  sem = sem_open(sem_name, O_CREAT | O_EXCL, 0666, 1);
  if (sem == SEM_FAILED) {
    std::cout << "*** *** sem_open failed!!!" << std::endl;
    std::cout << sem_name << ", errno: " << errno <<std::endl;
    sem = sem_open(sem_name, O_CREAT, 0666, 1);
    sem_close(sem);
    exit(1);
  } else {
    printf("sem: %p\n", sem);
  }

  // Initialize the Shared Memory (System V shared Memory)
  // Setup shared memory
  if ((shmid = shmget(key, shared_size, IPC_EXCL | IPC_CREAT | 0666)) < 0) {
    std::cout << "*** *** shmget() failed!!! "<< errno << std::endl;
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
    int i = 0;
    while (true)
    {
      // Grab a frame
      cv::Mat temp;
      if (!cap.read(temp)) {
        break;
      }

      // Shrink the frame size
      cv::resize(temp, temp, cv::Size(), 0.5, 0.5);
      cv::cvtColor(temp, temp, CV_BGR2GRAY);

      // Turn OpenCV's Mat into something dlib can deal with.  Note that this
      // just wraps the Mat object, it doesn't copy anything.  So cimg is only
      // valid as long as temp is valid.  Also don't do anything to temp that
      // would cause it to reallocate the memory which stores the image as that
      // will make cimg contain dangling pointers.  This basically means you
      // shouldn't modify temp while using cimg.
      dlib::cv_image<dlib::uint8> cimg(temp);

      // Detect faces
      std::vector<dlib::rectangle> faces = detector(cimg);


      // Find the pose of each face. (Only the first)
      std::vector<dlib::full_object_detection> shapes;
      for (unsigned long i = 0; i < 1 && i < faces.size(); ++i)
        shapes.push_back(pose_model(cimg, faces[i]));

      // guarantee that there are at least
      if (shapes.size()) {
        // left eye + right eye
        std::tuple<long, long, long, long> l_tp =
            EyeFinder::setMinAndMax(36, 41, shapes);
        std::tuple<long, long, long, long> r_tp =
            EyeFinder::setMinAndMax(42, 47, shapes);

        // ROI for left eye + right eye
        cv::Rect roi_l = EyeFinder::getROI(l_tp, temp);
        cv::Rect roi_r = EyeFinder::getROI(r_tp, temp);

        // Display eye tracking on the screen
        cv::Mat roi_l_mat, roi_r_mat;
        roi_l_mat = temp(roi_l);
        roi_r_mat = temp(roi_r);

        std::vector<std::pair<long, long>> facial_features_vec;

        // Left eye + Right eye points
        preCalculationPoints(facial_features_vec, shapes);

        // Find the face angle + pupils
        calculateFaceAngles();

        calculatePupils(roi_l_mat, facial_features_vec);
        calculatePupils(roi_r_mat, facial_features_vec);

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
// ***** ***** ***** ***** ***** ***** ***** ***** ***** ***** ***** ***** *****
// *****

// ***** ***** ***** ***** ***** ***** ***** ***** ***** ***** ***** ***** *****
// *****
// ***** ***** ***** ***** ***** ***** ***** ***** ***** ***** ***** ***** *****
// *****
// ***** ***** ***** ***** ***** ***** ***** ***** ***** ***** ***** ***** *****
// *****

// ***** ***** ***** ***** ***** ***** ***** ***** ***** ***** ***** ***** *****
// *****
// ***** ***** ***** ***** ***** ***** ***** ***** ***** ***** ***** ***** *****
// *****

// ***** ***** ***** ***** ***** ***** ***** ***** ***** ***** ***** ***** *****
// *****

/*
Art by Herman Hiddema
            _
           / )
        ,-(,' ,---.
       (,-.\,' `  _)-._
          ,' `(_)_)  ,-`--.
         /          (      )
        /            `-.,-'|
       /                |  /
       |               ,^ /
      /                   |
      |                   /
     /       PRIVATE     /
     |                   |
     |                   |
    /                    \
  ,.|                    |
(`\ |                    |
(\  |   --.      /  \_   |
 (__(   ___)-.   | '' )  /)
hh   `---...\\\--(__))/-'-'
*/

// *****
// setMinAndMax
std::tuple<long, long, long, long> _EF_::EyeFinder::setMinAndMax(
    int start, int end, std::vector<dlib::full_object_detection> &shapes) {
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
    std::vector<std::pair<long, long>> &facial_features_vec,
    const std::vector<dlib::full_object_detection> &shapes) {

  for (int i = 36; i <= 47; ++i) {
    auto shp = shapes[0].part(i);
    facial_features_vec.push_back(std::make_pair(shp.x(), shp.y()));
  }
}

// *****
// calculateFaceAngles()
void _EF_::EyeFinder::calculateFaceAngles(void) {usleep(100);}

// *****
// calculatePupils() a.k.a. Timm-Barth Algorithm
void _EF_::EyeFinder::calculatePupils(cv::Mat src, std::vector<std::pair<long, long>> &facial_features_vec) {

  cv::Mat src_blur, src_blur_inv;
  int scale = 1;
  int delta = 0;
  int ddepth = CV_64F;

  cv::GaussianBlur( src, src_blur, cv::Size(5,5), 0, 0, cv::BORDER_DEFAULT );
  src_blur_inv = cv::Mat::ones(src_blur.size(), src_blur.type()) * 255 - src_blur;

  cv::Mat grad_x, grad_y;
  cv::Sobel( src, grad_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT );
  cv::Sobel( src, grad_y, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT );

  cv::Mat mag(grad_x.size(), grad_x.type());
  cv::magnitude(grad_x, grad_y, mag);
  cv::divide(grad_x, mag, grad_x);
  cv::divide(grad_y, mag, grad_y);

  cv::Mat pixel_combination_x(src_blur_inv.size(), CV_16S);
  cv::Mat pixel_combination_y(src_blur_inv.size(), CV_16S);

  int row_LIMIT = src_blur_inv.rows;
  int col_LIMIT = src_blur_inv.cols;

  double max_objective = -777777.777;
  int max_r=0, max_c=0;

  for (int pixel_r = 10; pixel_r < row_LIMIT-10; ++pixel_r)
  {
    for (int pixel_c = 10; pixel_c < col_LIMIT-10; ++pixel_c)
    {
      double obj_val = 0.0;
      for (int i = 0; i < row_LIMIT; ++i)
      {
        cv::Vec3b* r_grad_x = grad_x.ptr<cv::Vec3b>(i);
        cv::Vec3b* r_grad_y = grad_y.ptr<cv::Vec3b>(i);
        cv::Vec3b* r_src_gray_blur_inv = src_blur_inv.ptr<cv::Vec3b>(i);
        for (int j = 0; j < col_LIMIT; ++j)
        {
          int x_val = i - pixel_r;
          int y_val = j - pixel_c;
          if ((x_val | y_val) == 0) break;
          double magnitude = sqrt(x_val*x_val + y_val*y_val);
          double unit_x = x_val/magnitude;
          double unit_y = y_val/magnitude;

          double dot = (unit_x*r_grad_x[j][0] + unit_y*r_grad_y[j][0]);
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

  facial_features_vec.push_back(std::make_pair((long)max_r, (long)max_c));
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
    const std::vector<std::pair<long, long>> &facial_features_vec) {
    int i = 0;
    sem_wait(sem);

    for (const auto pr : facial_features_vec) {
        long x = pr.first, y = pr.second;
        memcpy(shared_memory+sizeof(long)*i, &x, sizeof(long));
        memcpy(shared_memory+sizeof(long)*(i+1), &y, sizeof(long));
        i+=2;
    }

    sem_post(sem);
}



void _EF_::EyeFinder::writeBadFacialFeaturesToShm(void) {}
