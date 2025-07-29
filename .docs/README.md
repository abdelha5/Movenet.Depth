# Movenet.Depth
This is a modification of the pytorch implementation of Movenet "Movenet.Pytorch" created by @fire717 to include depth data of coco images. Depth is extracted using the state-of-the-art monocular depth estimation model "MiDas".
![midas](../data/imgs/midas.jpg)

## Intro
![start](../data/imgs/three_pane_aligned.gif)

MoveNet is an ultra fast and accurate model that detects 17 keypoints of a body.
This is A Pytorch implementation of MoveNet from Google. Further tuned to include Depth data.

Google just release pre-train models(tfjs or tflite), which cannot be converted to some CPU inference framework such as NCNN,Tengine,MNN,TNN, and we can not add our own custom data to finetune, so there is this repo.

## Results
This version of the model that was trained using depth data has proven to be able to outperform the original model by running more accurate keypoint predictions on image frames with more complex movements. We ran a comparison between both models on some frames as shown below.
#### Some samples
![throw](../data/imgs/Frisbee_throw.png)
![catch](../data/imgs/catch.png)


## Resources
1. [Blog:Next-Generation Pose Detection with MoveNet and TensorFlow.js](https://blog.tensorflow.org/2021/05/next-generation-pose-detection-with-movenet-and-tensorflowjs.html
)
2. [model card](https://storage.googleapis.com/movenet/MoveNet.SinglePose%20Model%20Card.pdf)
3. [TFHub：movenet/singlepose/lightning
](https://tfhub.dev/google/movenet/singlepose/lightning/4
)
