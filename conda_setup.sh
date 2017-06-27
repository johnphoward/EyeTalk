#!/bin/sh

conda create -n eyetalk python=3.5 numpy -y
source activate eyetalk
conda install -c menpo opencv3=3.1.0 dlib=19.4 -y
