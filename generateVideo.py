import cv2
import os
import grabber
import time
import numpy as np

def main():
    windowSize = 416
    g = grabber.Grabber(bbox=(int(1920/2-windowSize/2),int(1080/2-windowSize/2),int(1920/2+windowSize/2),int(1080/2+windowSize/2)))
    source = np.empty((windowSize, windowSize, 3), dtype='uint8')
    image_folder = 'log'
    video_name = 'video.avi'

    video = cv2.VideoWriter(video_name, 0, 1, (416,416))

    prev = 0
    while(1):
        g.grab(source) #screenshot
        #bgr = source[...,::-1] #convert to rgb (necessary for YOLOv3)
        #curr = time.time()
        if True:#curr - prev >= 1.6667:
            #prev = curr
            video.write(source)
            

    cv2.destroyAllWindows()
    video.release()

if __name__ == "__main__":
    main()
