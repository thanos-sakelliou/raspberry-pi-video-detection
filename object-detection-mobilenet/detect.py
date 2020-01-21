# -*- coding: utf-8 -*-

"""start object detection"""
import logging
import sys
import time
import click

import numpy as np
import cv2

from ssd_mobilenet_v3_coco import SSDMobileNet_V3_Small_Coco_PostProcessed, SSDMobileNet_V3_Coco_EdgeTPU_Quant
from opencv.movement_segmentation import gaussian_seg, background_substraction


@click.group()
def cli():
    pass


def run_detect(vs, model):
    LOGLEVEL = logging.getLogger().getEffectiveLevel()

    start_time = time.time()
    fps_counter = 0
    stream_name = "Stream object detection"
    
    while True:
        read, frame = vs.read()
        
        if frame is None:
            break
        
        #model requires 320x320 and has [352, 288]
        required_res = (320, 320)
        
        #FOR UPSCALE
#         scale_percent = required_res[0] * 100/ frame.shape[0]   
#         height = required_res[0]
#         width = int(frame.shape[1] * scale_percent / 100)
#         # resize image
#         resized_frame = cv2.resize(frame, (width, height), interpolation = cv2.INTER_AREA)
#         #crop frame to correct dimensions
#         cropped_frame = resized_frame[:, 0:required_res[1]]


        #DOWNSCALE AND PATCHING

        scale_percent = required_res[0] * 100/ frame.shape[1]
        
        width = required_res[0]
        height = int(frame.shape[0] * scale_percent / 100)
        
        # resize image
        resized_frame = cv2.resize(frame, (width, height), interpolation = cv2.INTER_AREA)
       
        #crop frame to correct dimensions
        bordersize = required_res[1] - height

        cropped_frame = cv2.copyMakeBorder(
            resized_frame,
            top=0,
            left=0,
            right=0,
            bottom=bordersize,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )
        
        prediction = model.predict(cropped_frame)
        overlay = model.create_overlay(
            cropped_frame, prediction)

        cv2.imshow(stream_name, overlay)
        
        key = cv2.waitKey(5) & 0xFF
        # if the `q` key is pressed, break from the lop
        if key == ord("q"):
            break

        
        if LOGLEVEL <= logging.INFO:
            fps_counter += 1
            if (time.time() - start_time) > 1:
                fps = fps_counter / (time.time() - start_time)
                logging.info(f'FPS: {fps}')
                fps_counter = 0
                start_time = time.time()
                
def run_detect_on_video(vs, model, motion_detection_type):
    LOGLEVEL = logging.getLogger().getEffectiveLevel()

    start_time = time.time()
    fps_counter = 0
    stream_name = "Stream object detection"
    
    min_area = 500
    rescale_width = 640
    skip = 0
    
    if motion_detection_type == "GAUSSIAN":
        #INIT
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        substractor = cv2.bgsegm.createBackgroundSubtractorGMG()

        substractor.setBackgroundPrior(0.65) # Background prior probability:
        substractor.setDecisionThreshold(0.75) # Decision threshold
        substractor.setDefaultLearningRate(0.02) #Learning rate
        substractor.setSmoothingRadius(25) #smoothing radius
        
        #repeat first 20 frames
        for i in range(18):
            read, frame = vs.read()
            
        while i<200:
            # resize image
            scale_percent = rescale_width * 100/ frame.shape[1]
            width = rescale_width
            height = int(frame.shape[0] * scale_percent / 100)
            frame_copy = frame.copy()
            resized_frame = cv2.resize(frame_copy, (width, height), interpolation = cv2.INTER_AREA)
            
            move_rect = gaussian_seg(resized_frame, kernel, substractor, min_area)
            i +=1
    
    if motion_detection_type == "BG_SUB":
        #skip first 20 frames
        for i in range(18):
            read, first_frame = vs.read()
        
        # resize image
        scale_percent = rescale_width * 100/ first_frame.shape[1]
        width = rescale_width
        height = int(first_frame.shape[0] * scale_percent / 100)
        frame_copy = first_frame.copy()
        first_frame = cv2.resize(frame_copy, (width, height), interpolation = cv2.INTER_AREA)
        
        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        first_frame = cv2.GaussianBlur(first_frame, (21, 21), 0)
    
    while True:
        read, frame = vs.read()
        #take only every 3rd frame (still makes delay)
        skip += 1
        if skip < 3:
            continue
        else:
            skip = 0
            
        if frame is None:
            break
        
        #model requires 320x320
        required_res = (320, 320)    
        
        #DOWNSCALE FOR MOVEMENT SEGMENTATION
        # resize image
        scale_percent = rescale_width * 100/ frame.shape[1]
        width = rescale_width
        height = int(frame.shape[0] * scale_percent / 100)
        frame_copy = frame.copy()
        resized_frame = cv2.resize(frame_copy, (width, height), interpolation = cv2.INTER_AREA)
        
         
        #movement segmentation - get rectangles of moving objects
        if motion_detection_type == "BG_SUB":
            move_rect = background_substraction(resized_frame, first_frame, min_area)
        elif motion_detection_type =="GAUSSIAN":
            move_rect = gaussian_seg(resized_frame, kernel, substractor, min_area)

        #scale it for the original frame
        move_rect = np.array(move_rect) * 100/int(scale_percent)
        
        if move_rect.any():
            for rect in move_rect:
               
                #resize frame to correct dimensions for model
                missing_pixels_height = required_res[1] - abs(int(rect[1] - rect[3]))
                missing_pixels_width = required_res[0] - abs(int(rect[0] - rect[2]))
#                 print("rect", rect)
#                 print("missing pixels", missing_pixels_height, missing_pixels_width)
                
                y1 = int(rect[1] - missing_pixels_height/2)
                if y1 < 0:
                    remainder = abs(y1)
                    y1 = 0
                elif y1 > frame.shape[0]:
                    remainder = y1 - frame.shape[0]
                    y1 = frame.shape[0]
                else:
                    remainder = 0

                y2 = int(rect[3] + missing_pixels_height/2) + remainder
                if y2 < 0:
                    remainder = abs(y2)
                    y2 = 0
                elif y2 > frame.shape[0]:
                    remainder = y2 - frame.shape[0]
                    y2 = frame.shape[0]
                else:
                    remainder = 0
                y1 += remainder
                
                x1 = int(rect[0] - missing_pixels_width/2)
                if x1 < 0:
                    remainder = abs(x1)
                    x1 = 0
                elif x1 > frame.shape[1]:
                    remainder = x1 - frame.shape[1]
                    x1 = frame.shape[1]
                else:
                    remainder = 0
                    
                x2 = int(rect[2] + missing_pixels_width/2) + remainder
                if x2 < 0:
                    remainder = abs(x2)
                    x2 = 0
                elif x2 > frame.shape[1]:
                    remainder = x2 - frame.shape[1]
                    x2 = frame.shape[1]
                else:
                    remainder = 0
                x1 += remainder
                
                
                ready_img_part = frame[y1:y2, x1:x2]
                
#                 print('dims', y1,y2,x1,x2)         
                
                #FIX
                if (ready_img_part.shape[0] ==320) and (ready_img_part.shape[1] ==320):
                    
                    prediction = model.predict(ready_img_part)
                    overlay = model.create_overlay(
                        ready_img_part, prediction)
                    frame[y1:y2, x1:x2] = overlay
#                 else:
#                     print("not processed frame: ", ready_img_part.shape)
            
        cv2.imshow(stream_name, frame)
        
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key is pressed, break from the lop
        if key == ord("q"):
            break

        
        if LOGLEVEL <= logging.INFO:
            fps_counter += 1
            if (time.time() - start_time) > 1:
                fps = fps_counter / (time.time() - start_time)
                logging.info(f'FPS: {fps}')
                fps_counter = 0
                start_time = time.time()


@cli.command()
@click.option('--loglevel', required=False, type=str, default='WARNING', help='Run object detection without pan-tilt controls. Pass --loglevel=DEBUG to inspect FPS.')
@click.option('--edge-tpu', is_flag=True, required=False, type=bool, default=False, help='Accelerate inferences using Coral USB Edge TPU')
@click.option('--video', required=False, type=str, default='', help='use the video.mp4 stored at the same folder, instead of a camera. Use BG_SUB for background substraction method, or GAUSSIAN for Gaussian segmentation')
def detect(loglevel, edge_tpu, video):
    level = logging.getLevelName(loglevel)
    logging.getLogger().setLevel(level)

    if edge_tpu:
        model = SSDMobileNet_V3_Coco_EdgeTPU_Quant()
    else:
        model = SSDMobileNet_V3_Small_Coco_PostProcessed()

#     framerate=30
    # Open camera
    if  video == 'GAUSSIAN' or video == 'BG_SUB':
        vs = cv2.VideoCapture("video.mp4")
        
        width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
        hight = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("VIDEO WIDTH, HIGHT:", width, hight)
        
        try:
            run_detect_on_video(vs, model, video)
        except KeyboardInterrupt:
            vs.release()
            
    else:
        vs = cv2.VideoCapture(0) 
        if not vs.isOpened():
            print("Cannot open camera")
            exit()

        # Init resolution
        
        #webcam has specific resolutions,  run 'v4l2-ctl --list-formats-ext'
        resolution = tuple([352, 288])
        vs.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        vs.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

        try:
            run_detect(vs, model)
        except KeyboardInterrupt:
            vs.release()


@cli.command()
@click.option('--loglevel', required=False, type=str, default='WARNING', help='List all valid classification labels')
def list_labels(loglevel):
    level = logging.getLevelName(loglevel)
    logging.getLogger().setLevel(level)
    model = SSDMobileNet_V3_Small_Coco_PostProcessed()
    print('You can detect / track the following objects:')
    print([x['name'] for x in model.category_index.values()])



def main():
    cli()


if __name__ == "__main__":
    main()

