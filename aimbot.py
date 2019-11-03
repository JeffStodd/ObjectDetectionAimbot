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

flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')

# C struct redefinitions 
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

move = INPUT()

move.type = 0
move.mi.dx = 0
move.mi.dy = 0
move.mi.mouseData = 0
move.mi.time = 0
move.mi.dwExtraInfo = None
move.mi.dwFlags = 0x001

down = INPUT()

down.type = 0
down.mi.dx = 0
down.mi.dy = 0
down.mi.mouseData = 0
down.mi.time = 0
move.mi.dwExtraInfo = None
down.mi.dwFlags = 0x001 | 0x002

up = INPUT()

up.type = 0
up.mi.dx = 0
up.mi.dy = 0
up.mi.mouseData = 0
up.mi.time = 0
move.mi.dwExtraInfo = None
up.mi.dwFlags = 0x001 | 0x004

def main(_argv):
    pid = win32api.GetCurrentProcessId()
    handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
    win32process.SetPriorityClass(handle, win32process.HIGH_PRIORITY_CLASS)

    windowSize = int(416)
    source = np.empty((windowSize, windowSize, 3), dtype='uint8')
    g = grabber.Grabber(bbox=(int(1920/2-windowSize/2),int(1080/2-windowSize/2),int(1920/2+windowSize/2),int(1080/2+windowSize/2)))
    
    #yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    yolo = YoloV3(classes=FLAGS.num_classes)

    #yolo.load_weights("./checkpoints/yolov3-tiny.tf")
    yolo.load_weights("./checkpoints/yolov3.tf")
    print('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    print('classes loaded')
    
    bgr = source = np.empty((windowSize, windowSize, 3), dtype='uint8')
    detection = np.empty((windowSize, windowSize, 3), dtype='uint8')

    r = np.random.random((1, 320, 320, 3)).astype(np.float32)
    yolo.predict(r)
    
    print("Ready")
    delta = 1
    while True:
        prev = np.zeros(shape=(1,2))
        while win32api.GetAsyncKeyState(win32con.VK_MENU): #alt is being held
            t1 = time.time()
            g.grab(source)

            #t = time.time()
            bgr = source[...,::-1] #convert to rgb
            
            #print(time.time() - t)
            bgr = tf.divide(bgr,255) #replace source with bgr
            bgr = tf.image.resize(bgr, (416,416))
            imgCapture = tf.expand_dims(bgr, 0) #replace source with bgr
            
            
            boxes, scores, classes, nums = yolo.predict(imgCapture)
            res = ctypes.windll.user32.SendInput(1, ctypes.pointer(up), ctypes.sizeof(up))
            
            if scores[0][0] > 0.5 and classes[0][0] == 0.0:
                detection = tf.multiply(boxes[0][0], windowSize)
                
                if(not np.any(prev)): #prev tuple is empty
                    ctypes.windll.user32.BlockInput(True)
                    prev = [int((detection[0] + detection[2])/2 - int(windowSize/2)), int((detection[1] * 1.1) - int(windowSize/2))]
                else:
                    ctypes.windll.user32.BlockInput(False)
                    x = int((detection[0] + detection[2])/2 - int(windowSize/2))
                    y = int((detection[1] * 1.1)) - int(windowSize/2)
                    #y = int((detection[1] + detection[3])/2 - 208)
                    vx = int((x - prev[0])*(2*delta))
                    vy = int((y - prev[1])*(2*delta))
                    
                    move.mi.dx = x + vx #416/2
                    move.mi.dy = y + vy #416/2
                    
                    res = ctypes.windll.user32.SendInput(1, ctypes.pointer(move), ctypes.sizeof(move))
                    res = ctypes.windll.user32.SendInput(1, ctypes.pointer(down), ctypes.sizeof(down))
                    
                    #prev = [x, y]
                    prev = np.zeros(shape=(1,2))
                
                #print(classes[0][0])
                #print(ctypes.FormatError(ctypes.GetLastError()))

            
                
            t2 = time.time()
            delta = t2-t1
            fps = 1/delta
            logging.info(fps)
            
            #source = draw_outputs(source, (boxes, scores, classes, nums), class_names)
            #cv2.imwrite("./log/" + str(t2)+'.jpg', source)
            #cv2.imshow('image',source)
            #cv2.waitKey(1)

        if win32api.GetAsyncKeyState(win32con.VK_LBUTTON):
            res = ctypes.windll.user32.SendInput(1, ctypes.pointer(up), ctypes.sizeof(up))
        time.sleep(0.01)
        ctypes.windll.user32.BlockInput(False)
        

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
