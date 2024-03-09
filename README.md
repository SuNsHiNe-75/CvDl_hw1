#### Introduction
This is the first assignment of the Computer Vision and Deep Learning course at NCKU, aiming to implement techniques such as Camera Calibration, Augmented Reality, Stereo Disparity Map, SIFT, and Training a CIFAR10 Classifier Using VGG19 with BN.

#### Establishing Environment
I set up the development environment using Virtualenv, and downloaded relevant packages as follows:
```shell
virtualenv Env_hw1 --python=python3.11.6
python -m pip install --upgrade pip

pip install opencv-contrib-python

pip install matplotlib

pip install pyqt5
pip install pyqt5-tools

pip install pytorch
pip install torchvision
pip install torchsummary
pip install tensorboard
pip install pillow

pip install keras
pip install tensorflow
```

#### Run
In the Q1toQ5 folder:  
```shell
python main.py
```

#### Notice
- Due to the large size of the trained model, it has not been uploaded to this repository. If you want to run the fifth question of this project, you will need to train the model yourself.
- If you want to use the pyqt5 package, you may need to configure some things in your IDE (for example, install relevant extensions in VSCode).
- After the UI window pops up, there are corresponding images/videos needed to be imported first for each question. Pay attention to the execution order.
