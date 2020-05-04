# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 15:35:18 2020

@author: Fuso

Beatriz Campos de Almeida de Castro Monteiro
NUSP: 9778619
Bacharel em Ciência da Computação - ICMC, USP São Carlos
SCC0251/5830— Prof. Moacir Ponti
2020.01

github link: https://github.com/biamont/PDI/tree/master/Image%20Enhancement%20and%20Filtering

Assignment 2 :  Image Enhancement and Filtering

"""

import numpy as np
import imageio, math

#CREATES A BORDER OF n/2 0s around the img n=3 t=3 a = b = som = 1
def padding(f, kernel):
    
    n, m = f.shape
    
    kr, kc = kernel.shape
    
    centerR = int((kr-1)/2) # find middle row of kernel
    centerC = int((kc-1)/2) #find middle collun of kernel

    pad_filter = np.zeros((n+(kr-1),m+(kc-1)), dtype=np.float)

    for i in range(centerR, n+centerR):
        for j in range(centerC,m+centerC):
            pad_filter[i][j] = f[i-centerR][j-centerC]
    
    return pad_filter

def spatialGaussian(n, gs, center):
    #creates a nxn matrix of 1s
    w = np.ones((n,n), dtype=np.float )
    
    for i in range(n):
        for j in range(n):
            x = math.sqrt(math.pow((i-center),2)+math.pow((j-center),2))
            p1 = 1/(2*(math.pi)*math.pow(gs,2))
            p2 = -(math.pow(x,2)/(2*math.pow(gs,2)))
            w[i][j] = math.pow(p1,p2)
        
    return w

def rangeGaussian(gr, x):

    p1 = 1/(2*np.pi*math.pow(gr,2))
    p2 = (-(math.pow(x,2)))/(2*math.pow(gr,2))
    final = float(p1*np.exp(p2))

    return final

def convolution__bilateral_f(f, ws, gr, x, y, cr, cc):
    n, m = ws.shape #dimensions of w
    a = int((n-1)/2)
    b = int((m-1)/2)
    
    #copy img region centered at x, y
    region_f = np.zeros((n,m), dtype=np.float)
    region_f = f[(x-a):(x+(a+1)), (y-b):(y+(b+1))]
    
    Wp, If = 0.0, 0.0
    
    for i in range(n):
        for j in range(m):
            ngb = region_f[i][j]
            x = region_f[i][j] - region_f[cr][cc]
            
            gri = rangeGaussian(gr, x) #cria filtro range gaussian
            Wi = gri*float(ws[i][j])
            Wp = Wp+float(Wi)
            If = If+(Wi*ngb)    
 
    value = int(If/Wp)
    
    return value
 
def convolution_point(f, kernel, x, y):

    n, m = kernel.shape #dimensions of w
    a = int((n-1)/2)
    b = int((m-1)/2)
    
    #copy img region centered at x, y
    region_f = np.zeros((n,m), dtype=np.float)
    region_f = f[(x-a):(x+(a+1)), (y-b):(y+(b+1))]
    
    If = 0.0
    s1, s2 = region_f.shape 

    for i in range(n):
        for j in range(m):
            Ii = region_f[i][j]
            If = If + (Ii * kernel[i, j])   
    
    return If
    
def bilateral_filter(img, final_img):
    n = int(input())
    gs = float(input())
    gr = float(input())
    

    center = int((n-1)/2) # find middle row of filter
  
    #creates spatial gaussian
    ws = spatialGaussian(n, gs, center)
    
    pad_img = padding(img, ws) #creates a border of zereos so convolution can be applied
    
    som = int((n-1)/2)
    
    #appling filter by convolution with ws and gr
    for i in range(som, t1+som):
        for j in range(som, t2+som):
            final_img[i-som][j-som] = convolution__bilateral_f(pad_img, ws, gr, i, j, center, center)
            
    return final_img

def unsharp_mask(img, final_img):
    c = float(input())
    k = int(input())
    
    #defining kernel filters
    kernel1 = np.matrix([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    kernel2 = np.matrix([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
    
    nr,nc = kernel1.shape# since the shape of kernel 1 is the same as karnel 2, it doesnt matter which you'll use
    centerR = int((nr-1)/2) # find middle row of kernel
    centerC = int((nc-1)/2) #find middle collun of kernel
    

    #creating padding with +centerR 0's on top and bottom of img and +centerC on right and left of img
    if k == 1:
        pad_img = padding(img, kernel1)
    if k== 2:
        pad_img = padding(img, kernel2) 
    
    #applying the convolution with the chosed kernel filter
    for i in range(centerR, t1+centerR):
        for j in range(centerC, t2+centerC):
            if k == 1:
                final_img[i-centerR][j-centerC] = convolution_point(pad_img, kernel1, i, j)
            else:
                if k == 2:
                    final_img[i-centerR][j-centerC] = convolution_point(pad_img, kernel2, i, j)
                else:
                    print('Invalid Kernel number')
    
    #scaling and adding
    min_f_i = np.min(final_img)
    max_f_i = np.max(final_img)
   
    for i in range(t1):
        for j in range(t2):
            final_img[i,j] = ((final_img[i,j]-min_f_i)*255)/(max_f_i - min_f_i)
            
            final_img[i,j] = (final_img[i,j]*c) + img[i,j]

    #scaling again
    min_f_i = np.min(final_img)
    max_f_i = np.max(final_img)
	
    for i in range(t1):
        for j in range(t2):
            final_img[i,j] = ((final_img[i,j]-min_f_i)*255)/(max_f_i - min_f_i)
            
    return final_img

def viganette_filter(img, final_img):
    
    delta_r = float(input())  
    delta_c = float(input())
    
    #creating 1D gaussian kernel for the size of rows (gr) and colluns (gc) of img
    gr = np.zeros((1, t1), dtype=np.float)
    gc = np.zeros((1, t2), dtype=np.float)
      
    a = int((t1-1)/2)
    b = int((t2-1)/2)
    
    #defining each element as the distance between the element end the array's center
    for i in range(t1):
        gr[0][i] = rangeGaussian(delta_r, i-a)

    for i in range(t2):
        gc[0][i] = rangeGaussian(delta_c, i-b)
        
    #multiplying gr(as a collun) and gc
    Mrc = np.matmul(gr.T, gc)
   
    #multiplying element by element the original img and Mrc
    for i in range(t1):
        for j in range(t2):
            final_img[i][j] = img[i][j]*Mrc[i][j]
    
    #scaling       
    min_f_i = np.min(final_img)
    max_f_i = np.max(final_img)
	
    for i in range(t1):
        for j in range(t2):
            final_img[i,j] = ((final_img[i,j]-min_f_i)*255)/(max_f_i - min_f_i)
            
    return final_img    
    
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


#BILATERAL FILTER
if method == 1:
    final_img = bilateral_filter(img, final_img)

#UNSHARP MASK USING LAPLACIAN FILTER
if method == 2:
    final_img = unsharp_mask(img, final_img)

#VIGANETTE FILTER
            
if method == 3:
    final_img = viganette_filter(img, final_img)
    

#COMPARING IMG AND IMG_FINAL
total = 0.0

for i in range(t1):
	for j in range(t2):
		total += (final_img[i,j] - float(img[i,j]))**2
    
total = np.sqrt(total)

print(round(total,4))

final_img = final_img.astype(np.uint8) #transforms image bacj to its original format (uint8)
    
if save == 1:
    imageio.imwrite('output_img.png', final_img)
  