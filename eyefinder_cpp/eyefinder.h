#ifndef EYEFINDER_H
#define EYEFINDER_H

//#include <dlib/gui_widgets.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>

#include "opencv2/imgproc/imgproc.hpp"

#include <cmath>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// C headers
// link with -lpthread
#include <fcntl.h>
#include <semaphore.h>
#include <stdlib.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

// EyeLike
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include <iostream>
#include <math.h>
#include <queue>
#include <stdio.h>

#include "constants.h"
#include "findEyeCenter.h"
#include "findEyeCorner.h"

#define DEBUG 0
#define DEBUG_TB 1
#define MACRO_START (begin = std::chrono::steady_clock::now())
#define MACRO_END (end = std::chrono::steady_clock::now())
#define MACRO_P_DIFF(MSG)                                                      \
  (std::cout << MSG                                                            \
             << std::chrono::duration_cast<std::chrono::microseconds>(end -    \
                                                                      begin)   \
                    .count()                                                   \
             << std::endl)
namespace _EF_ {

class EyeFinder final {

public:
  EyeFinder(void);
  ~EyeFinder(void);
  // Deleted so blocked from use
  EyeFinder(const EyeFinder &other) = delete;
  EyeFinder(EyeFinder &&other) = delete;
  EyeFinder &operator=(const EyeFinder &other) = delete;
  EyeFinder &operator=(EyeFinder &&other) = delete;

  int start(void);

private:
  const bool showmain = false;
  const bool showeyes = false;
  const bool clear = true;
  static const unsigned int NUM_WINS = 3;

  cv::VideoCapture cap;

  sem_t *sem;
  const char *sem_name = "/capstone";

  long frame_id = 0;

  int shmid;
  char *shared_memory;
  const key_t key = 123456;
  const unsigned int shared_size = sizeof(long) + 30 * sizeof(long);

  // std::vector<long> facial_features_vec;
  // change to python list later with boost, look below in start()

  std::tuple<long, long, long, long>
  setMinAndMax(int start, int end,
               const std::vector<dlib::full_object_detection> &shapes);
  cv::Rect getROI(std::tuple<long, long, long, long> &tp, cv::Mat frame);

  void
  preCalculationPoints(const std::vector<dlib::full_object_detection> &shapes,
                       std::vector<long> &facial_features_vec);
  void
  calculateFaceAngles(const std::vector<dlib::full_object_detection> &shapes,
                      std::vector<long> &facial_features_vec);
  void calculatePupils(cv::Mat src, std::vector<long> &facial_features_vec);
  void calculatePupilsEL(const std::vector<dlib::full_object_detection> &shapes,
                         std::vector<long> &facial_features_vec, cv::Mat temp);
  void writeFacialFeaturesToShm(const std::vector<long> &facial_features_vec);
  void writeBadFacialFeaturesToShm(void);
};
}; // namespace _EF_
#endif
