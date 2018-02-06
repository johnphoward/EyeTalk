#!/bin/bash

if [ ! -d "build" ]; then
  # Control will enter here if build directory doesn't exist.
  mkdir -p build && cd build && cmake .. -DUSE_AVX_INSTRUCTIONS=ON && make -j && cd ..
else
  cd build && make -j && cd ..
fi

if [ ! -f "./build/shape_predictor_68_face_landmarks.dat" ]; then
  cd build && \
  wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 && \
  bzip2 -d shape_predictor_68_face_landmarks.dat.bz2 && \
  cd ..
fi

# cd build && ./eyefinder

# if [[ "$OSTYPE" == "linux-gnu" ]]; then
#         # ...
# elif [[ "$OSTYPE" == "darwin"* ]]; then
#         # Mac OSX
# elif [[ "$OSTYPE" == "cygwin" ]]; then
#         # POSIX compatibility layer and Linux environment emulation for Windows
# elif [[ "$OSTYPE" == "msys" ]]; then
#         # Lightweight shell and GNU utilities compiled for Windows (part of MinGW)
# elif [[ "$OSTYPE" == "win32" ]]; then
#         # I'm not sure this can happen.
# elif [[ "$OSTYPE" == "freebsd"* ]]; then
#         # nothing
# else
#         # nothing
# fi
