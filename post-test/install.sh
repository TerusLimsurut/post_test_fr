#!/bin/bash

# post-eval/install.sh

conda create -n post_env python=3.6 pip -y
source activate post_env

pip install numpy==1.16.3
pip install pyarmor==5.4.3