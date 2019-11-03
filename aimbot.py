import time
from matplotlib import pyplot
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import grabber
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs
from threading import Thread

import win32api
import win32con
import win32process

import ctypes
from ctypes import c_long, POINTER, sizeof, c_int
from ctypes.wintypes import DWORD

'''
Flags and YOLOv3 Tensorflow Implementation cloned from https://github.com/zzh8829/yolov3-tf2
'''
flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')

#C struct redefinitions 
class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", c_long),
        ("dy", c_long),
        ("mouseData", DWORD),
        ("dwFlags", DWORD),
        ("time", DWORD),
        ("dwExtraInfo", POINTER(c_long)),
    ]

class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", DWORD),
        ("mi", MOUSEINPUT),
    ]

'''
Input types
'''
move = INPUT()

move.type = 0
move.mi.dx = 0
move.mi.dy = 0
move.mi.mouseData = 0
move.mi.time = 0
move.mi.dwExtraInfo = None
move.mi.dwFlags = 0x001 #mouse moved flag

down = INPUT() #Left mouse down

down.type = 0
down.mi.dx = 0
down.mi.dy = 0
down.mi.mouseData = 0
down.mi.time = 0
move.mi.dwExtraInfo = None
down.mi.dwFlags = 0x001 | 0x002 #mouse moved | mouse down flag

up = INPUT() #left mouse up

up.type = 0
up.mi.dx = 0
up.mi.dy = 0
up.mi.mouseData = 0
up.mi.time = 0
move.mi.dwExtraInfo = None
up.mi.dwFlags = 0x001 | 0x004 #mouse moved | mouse up flag
'''
End of declaring input types
'''

def main(_argv):
    '''
    Set process to high priority for best performance
    '''
    pid = win32api.GetCurrentProcessId()
    handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
    win32process.SetPriorityClass(handle, win32process.HIGH_PRIORITY_CLASS)
    
    '''
    Setting up variables and tools for screenshotting
    '''
    windowSize = int(416)
    source = np.empty((windowSize, windowSize, 3), dtype='uint8')
    #bbox = (top left coord, bottom right coord)
    g = grabber.Grabber(bbox=(int(1920/2-windowSize/2),int(1080/2-windowSize/2),int(1920/2+windowSize/2),int(1080/2+windowSize/2)))
    bgr = source = np.empty((windowSize, windowSize, 3), dtype='uint8')
    detection = np.empty((windowSize, windowSize, 3), dtype='uint8')
    
    '''
    Load YOLOv3 model and weights, use tiny for better fps but lower accuracy
    '''
    #yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    yolo = YoloV3(classes=FLAGS.num_classes)

    #yolo.load_weights("./checkpoints/yolov3-tiny.tf")
    yolo.load_weights("./checkpoints/yolov3.tf")
    print('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    print('classes loaded')
    
    '''
    Network doesn't compute quickly on first forward propagation
    '''
    r = np.random.random((1, 320, 320, 3)).astype(np.float32)
    yolo.predict(r)
    
    print("Ready")
    delta = 1
    while True:
        prev = np.zeros(shape=(1,2))
        while win32api.GetAsyncKeyState(win32con.VK_MENU): #alt is being held
            t1 = time.time()
            g.grab(source) #screenshot

            bgr = source[...,::-1] #convert to rgb (necessary for YOLOv3)
            
            bgr = tf.divide(bgr,255) #scale pixel values from 0 to 1
            bgr = tf.image.resize(bgr, (416,416)) #resize to proper input dimensions
            imgCapture = tf.expand_dims(bgr, 0) #array of one image

            #results of prediction
            boxes, scores, classes, nums = yolo.predict(imgCapture)
            #mouse is released from previous clicking action
            res = ctypes.windll.user32.SendInput(1, ctypes.pointer(up), ctypes.sizeof(up))

            #if a person is detected and confidence is > 50%
            if scores[0][0] > 0.5 and classes[0][0] == 0.0:
                #detections are denormalized from 0..1 to 0..416
                detection = tf.multiply(boxes[0][0], windowSize)
                
                if(not np.any(prev)): #prev tuple is empty
                    ctypes.windll.user32.BlockInput(True) #need to fix (not working for some reason)
                    '''
                    get prev detection to calculate velocity and aim ahead
                    prev = middle of x values, head value
                    Note: yolo tiny cannot detect head for at long distances, detects upper chest instead
                    '''
                    prev = [int((detection[0] + detection[2])/2 - int(windowSize/2)), int((detection[1] * 1.1) - int(windowSize/2))]
                else: #else prev has already been set, go ahead with aiming
                    ctypes.windll.user32.BlockInput(False) #not working
                    '''
                    x and y values are the distances from the origin (center of screen) to the aim points
                    vx and vy are estimated using distance covered in last detection and time expended in previous detection
                    '''
                    x = int((detection[0] + detection[2])/2 - int(windowSize/2))
                    y = int((detection[1] * 1.1)) - int(windowSize/2)
                    #y = int((detection[1] + detection[3])/2 - 208)
                    vx = int((x - prev[0])*(2*delta))
                    vy = int((y - prev[1])*(2*delta))

                    '''
                    Setting mouse move inputs
                    Sending inputs
                    Setting prev array of 0's
                    '''
                    move.mi.dx = x + vx #416/2
                    move.mi.dy = y + vy #416/2

                    #mouse move
                    res = ctypes.windll.user32.SendInput(1, ctypes.pointer(move), ctypes.sizeof(move))
                    #mouse click (will be held down and burst until mouse up before next detection)
                    res = ctypes.windll.user32.SendInput(1, ctypes.pointer(down), ctypes.sizeof(down))
                    
                    #prev = [x, y]
                    prev = np.zeros(shape=(1,2))

            
            #Debugging info (time and fps)
            t2 = time.time()
            delta = t2-t1
            fps = 1/delta
            logging.info(fps)
            
            #source = draw_outputs(source, (boxes, scores, classes, nums), class_names)
            #cv2.imwrite("./log/" + str(t2)+'.jpg', source)
            #cv2.imshow('image',source)
            #cv2.waitKey(1)

        #if mouse is still being held, release it
        if win32api.GetAsyncKeyState(win32con.VK_LBUTTON):
            res = ctypes.windll.user32.SendInput(1, ctypes.pointer(up), ctypes.sizeof(up))
        #sleep 10 ms to prevent CPU hogging
        time.sleep(0.01)
        ctypes.windll.user32.BlockInput(False) #not working
        

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
