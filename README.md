## Robust Facial Landmark Detection via a Fully-Convolutional Local-Global Context Network

* Facial landmarks are used to localize and represent salient regions of the face, such as: Eyes, Eyebrows, Nose, Mouth, Jawline
* Challenges: different shapes, poses, lighting conditions, occlusions, etc.
* Uses of landmark detection: face alignment, head pose estimation, face swapping, blink detection and much more.
* Following is an iPython Notebook implementation of paper:
  * Robust Facial Landmark Detection via a Fully-Convolutional Local-Global Context Network, Proceedings of the International Conference on Computer Vision and Pattern Recognition (CVPR), IEEE, 2018 - Daniel Merget, Matthias Rock, Gerhard Rigoll
  * Link: https://www.mmk.ei.tum.de/fileadmin/w00bqn/www/Verschiedenes/cvpr2018.pdf
* Code is written in brainscript and also Python using Microsoft CNTK and for post-processing Matlab is used.
* Code and other details can be found [here](https://www.mmk.ei.tum.de/cvpr2018/).

### Important notes from the paper
* Fully convolutional NN are good at modeling local features, but it results to constrained receptive field (local context).
* To overcome this there are many ways: cascades/pooling etc. This paper proposes a new approach to use channel-wise/kernel convolution and dilated convolution (global context) to achieve the same with better accuracy than several SOTA methods. It introduces global context into a fully-convolutional neural network directly.
* Major Contributions:
  * Uses Kernel Convolution directly within the network
  * Uses Dilated Convolutions to increase receptive field
  * Doesn’t depend on prior face detections
  * Input image is directly mapped to heatmap based tensor allowing the network to be accurate and robust

### To run the code
* main.py is meant to be run on Google Colab with Python3 & GPU runtime
* main.ipynb is to see working output of main.py

### Our Work-Plan
* Our plan is to port their code to python so that it can be run on Google Colab for our experimentation and then later make it production ready
* Steps to be followed:
  1. Port CNTK neural network code to Keras
  2. Port Matlab post-processing code to Python
  3. Train the network using 300-W dataset
  4. Make it work for multiple-faces (images from wild)
