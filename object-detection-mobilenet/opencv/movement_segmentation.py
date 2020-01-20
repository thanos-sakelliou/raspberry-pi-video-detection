import imutils
import cv2
 
    
def gaussian_seg(frame, kernel, substractor, min_area=500):

    mask = substractor.apply(frame)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    dilated_mask = cv2.dilate(mask, None, iterations=2)
 
    cnts = cv2.findContours(dilated_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    move_rect = []
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        
        (x, y, w, h) = cv2.boundingRect(c)
        move_rect.append([x, y, x+w, y+h])
        cv2.rectangle(frame, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
  
#     cv2.imshow("Dilated mask", dilated_mask)
#     cv2.imshow("Mask", mask)
    cv2.imshow("Movement detection feed", frame)

    return move_rect


def background_substraction(frame, first_frame, min_area=500):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # compute the absolute difference between the current frame and
    # first frame
    frameDelta = cv2.absdiff(first_frame, gray)
    thresh = cv2.threshold(frameDelta, 5, 255, cv2.THRESH_BINARY)[1]

    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    move_rect = []
    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < min_area:
            continue

        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        move_rect.append([x, y, x+w, y+h])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


    # show the frame and record if the user presses a key
    cv2.imshow("Movement detection feed", frame)
#     cv2.imshow("Thresh", thresh)
#     cv2.imshow("Frame Delta", frameDelta)

    return move_rect
 

