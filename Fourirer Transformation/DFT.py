# -*- coding: utf-8 -*-
"""
Created on Thu May 14 17:37:47 2020

@author: Fuso
"""
import numpy as np
import imageio, math

#MAIN

#read image
filename = str(input()).rstrip()
input_img = imageio.imread(filename)

method = int(input()) #select method
save = int(input()) #1(save final_img) 0(dont save)

img = np.array(input_img)
img = img.astype(np.int32) #casting para realizar as funcoes

#creates matrix for final image
t1, t2 = img.shape
final_img = np.zeros((t1,t2), dtype=np.float)
