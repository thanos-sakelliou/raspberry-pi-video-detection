# raspberry-pi-video-detection
video/image object detection/recognition with old algorithms and new DL models. Raspberry Pi 4 Model B

![raspberry Pi 4](https://github.com/thanos-sakelliou/raspberry-pi-video-detection/blob/master/2019_Sakelliou_img_2.png)

The project "object-detection-mobilenet" is for dectecting objects with a webcome connected to the Raspberry Pi, using the latest(Jan 2020) tensorflow 2.0 lite, with the pretrained mobile model MobileNetV3-SSD (including coralTPU version). We used a tweaked model from the repo: https://github.com/leigh-johnson/rpi-deep-pantilt

command to run: python3 detect.py detect
-If not used with the text option, it opens the defult mode, which is simple application of the model on the webcam(320x320).
-If a Hi-res video is used, then, a bacground substraction method is used, to detect moving objects and then feed them to the model.This way, it is possible to run the detection on larger images. (ex.1280x720) 

Options:

  --loglevel TEXT  Run object detection without pan-tilt controls. Pass
                   --loglevel=DEBUG to inspect FPS.
                   
  --edge-tpu       Accelerate inferences using Coral USB Edge TPU
  
  --video TEXT     use the video.mp4 stored at the same folder, instead of a
                   camera. Use BG_SUB for static background substraction method, or
                   GAUSSIAN for Gaussian mixture dynamic background substraction.
                   
  --help           Show this message and exit.
