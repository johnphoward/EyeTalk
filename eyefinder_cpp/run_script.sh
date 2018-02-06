#!/bin/bash

if [ ! -d "build" ]; then
  if [ "$1" == "debug" ]; then
    mkdir -p build && cd build && cmake .. -DDEBUG=1 -DDEBUG_TB=1 -DUSE_AVX_INSTRUCTIONS=ON \
    && make -j && cd .. && echo "$1"
    cp ../shape_predictor_68_face_landmarks.dat .
  else
    # Control will enter here if build directory doesn't exist.
    mkdir -p build && cd build && cmake .. -DUSE_AVX_INSTRUCTIONS=ON && make -j && cd ..
  fi
else
  cd build && make -j && cd ..
fi

if [ "$1" == "debug" ]; then
  ./build/eyefinder
fi
