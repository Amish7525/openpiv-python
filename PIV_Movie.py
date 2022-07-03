#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 10:34:21 2022

@author: alekmishra
"""

from openpiv import tools, scaling, pyprocess, validation, filters,preprocess
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib import pyplot as plt

PIVimg = []

# load each frame of .avi file into an array
frames = []
video = cv2.VideoCapture("/Users/alekmishra/Library/CloudStorage/Box-Box/SMAD2 Project_Video Analysis_AM/Processed Videos/C4/B3_C4_18/B3_C4_18_Fast.avi")
j = 0
while True:
    read, frame= video.read()
    if not read:
        break
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frames.append(grayFrame)
    print ("DONE " + str(j))
    j += 1
    
frames = np.array(frames)

'''
k = 0

while k <=166:
    plt.imshow(frames[k], interpolation='nearest')
    plt.show()
    k += 1
'''



i = 1

while i <= 166:
    frame_a  = frames[i-1]
    frame_b  = frames[i]

    u, v, sig2noise = pyprocess.extended_search_area_piv( frame_a, frame_b, \
        window_size=32, overlap=16, dt=0.02, search_area_size=64, sig2noise_method='peak2peak' )

    print(u,v,sig2noise)

    x, y = pyprocess.get_coordinates( image_size=frame_a.shape, search_area_size=64, overlap=16 )
    u, v, mask = validation.sig2noise_val( u, v, sig2noise, threshold = 1.3 )
    u, v, mask = validation.global_val( u, v, (-1000, 2000), (-1000, 1000) )
    u, v = filters.replace_outliers( u, v, method='localmean', max_iter=10, kernel_size=2)
    x, y, u, v = scaling.uniform(x, y, u, v, scaling_factor = 96.52 )
    tools.save(x, y, u, v, mask, '../data/test1/test_data.vec' )
    tools.display_vector_field('../data/test1/test_data.vec', scale=75, width=0.0035)
    print(i)
    i += 1

   

   
    