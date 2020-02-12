# Face Recogniton Resnet-50 model ported to TensorFlow Lite for Android

* This code makes use of tensorflow pb file converted to tflite:

curl https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_0.50_128_frozen.tgz | tar xzv -C /tmp
tflite_convert   --output_file=/tmp/foo.tflite   --graph_def_file=/tmp/mobilenet_v1_0.50_128/frozen_graph.pb   --input_arrays=input   --output_arrays=MobilenetV1/Predictions/Reshape_1

tflite_convert --output_file=gvFR.tflite --graph_def_file=09-02_02-45.pb --input_arrays=input --output_arrays=embedding --input_shapes=10,224,224,3

* Implement of Android NDK to load custom Face Recognition tflite models instead of the classifier example from Google TensorFlow example

* Images below shows face extract 512-D features and similarity between Face One and Two.

<p align="center">
  <img src="https://github.com/joytsay/Android-TensorFlow-Lite-Example/blob/master/assets/Screenshot_1581501908.png?raw=true" width="250">
  <img src="https://github.com/joytsay/Android-TensorFlow-Lite-Example/blob/master/assets/Screenshot_1581501919.png?raw=true" width="250">
  <img src="https://github.com/joytsay/Android-TensorFlow-Lite-Example/blob/master/assets/Screenshot_1581501930.png?raw=true" width="250">
</p>
<br>
<br>

# Android TensorFlow Lite Machine Learning Example

[![Mindorks](https://img.shields.io/badge/mindorks-opensource-blue.svg)](https://mindorks.com/open-source-projects)
[![Mindorks Community](https://img.shields.io/badge/join-community-blue.svg)](https://mindorks.com/join-community)
[![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.svg?v=102)](https://opensource.org/licenses/Apache-2.0)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/amitshekhariitbhu/Android-TensorFlow-Lite-Example/blob/master/LICENSE)

##  About Android TensorFlow Lite Machine Learning Example
* This is an example project for integrating [TensorFlow Lite](https://www.tensorflow.org/mobile/tflite/) into Android application
* This project include an example for object detection for an image taken from camera using TensorFlow Lite library.

# [Read this article. It describes everything about TensorFlow Lite for Android.](https://letslearnai.com/2018/03/17/android-tensorflow-lite-machine-learning-example.html)

<p align="center">
  <img src="https://raw.githubusercontent.com/amitshekhariitbhu/Android-TensorFlow-Lite-Example/master/assets/keyboard_example.png" width="250">
  <img src="https://raw.githubusercontent.com/amitshekhariitbhu/Android-TensorFlow-Lite-Example/master/assets/pen_example.png" width="250">
  <img src="https://raw.githubusercontent.com/amitshekhariitbhu/Android-TensorFlow-Lite-Example/master/assets/wallet_example.png" width="250">
</p>
<img src=https://raw.githubusercontent.com/amitshekhariitbhu/Android-TensorFlow-Lite-Example/master/assets/sample_combined.png >
<br>
<br>

### Find this project useful ? :heart:
* Support it by clicking the :star: button on the upper right of this page. :v:

### Credits
* The classifier example has been taken from Google TensorFlow example.

[Check out Mindorks awesome open source projects here](https://mindorks.com/open-source-projects)

### License
```
   Copyright (C) 2018 MINDORKS NEXTGEN PRIVATE LIMITED

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```

### Contributing to Android TensorFlow Lite Machine Learning Example
Just make pull request. You are in!
