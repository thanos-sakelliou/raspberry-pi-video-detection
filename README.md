# raspberry-pi-video-detection
video/image object detection/recognition with old algorithms and new DL models. Raspberry Pi 4 Model B

![raspberry Pi 4](https://github.com/thanos-sakelliou/raspberry-pi-video-detection/blob/master/2019_Sakelliou_img_2.png)

The project "object-detection-mobilenet" is for dectecting objects with a webcome connected to the Raspberry Pi, using the latest(Jan 2020) tensorflow 2.0 lite, with the pretrained mobile model MobileNetV3-SSD (including coralTPU version). We used a tweaked model from the repo: https://github.com/leigh-johnson/rpi-deep-pantilt

Default command (showing FPS): python3 detect.py detect --loglevel INFO
