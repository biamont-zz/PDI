# -*- coding: utf-8 -*-
"""
Created on Thu May 14 18:04:03 2020

@author: Fuso
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 14 17:37:47 2020

@author: Fuso
"""
import numpy as np
import imageio, time, warnings

def DFT2D(f):
    F = np.zeros(f.shape, dtype=np.complex64)
    n,m = f.shape
    
    x = np.arange(n).reshape(n,1)
    y = np.arange(m).reshape(1, m)
    
    for u in np.arange(n):
        for v in np.arange(m):
           F[u,v] = np.sum(f[x,y]*np.exp((-1j*2*np.pi)*(((u*x/n)+((v*y)/m)))))
    
    return F/np.sqrt(n*m)

def INV_DFT2D(f):
    
    F = np.zeros(f.shape, dtype=np.complex64)
    n,m = f.shape
    
    x = np.arange(n).reshape(n,1)
    y = np.arange(m).reshape(1, m)
    
    for u in np.arange(n):
        for v in np.arange(m):
            
            F[u,v] = np.sum(f[x,y]*np.exp((1j*2*np.pi)*(((u*x/n)+((v*y)/m)))))
    
    return F/np.sqrt(m*n)

def find_2ndMax(f):
    
    p1 = np.max(f)
#    print(p1)
    n,m = f.shape
    
    p2 = 0
    
    for i in range(n):
        for j in range(m):
            if np.real(np.abs(f[i][j])) > p2 and np.real(np.abs(f[i][j])) != p1:
                p2 = np.real(np.abs(f[i][j]))

    return p2
                
            
#MAIN

#read image
filename = str(input()).rstrip()
input_img = imageio.imread(filename)

T = float(input()) #coeficient

img = np.array(input_img)
img = img.astype(np.float32) #casting para realizar as funcoes

start = time.time()

#creates matrix for final image
n,m = img.shape
fourier_img = np.zeros((n,m), dtype=np.float)

fourier_img = DFT2D(img)

p2 = find_2ndMax(fourier_img)
#print(p2)
cnt = 0

for i in range(n):
    for j in range(m):
        if abs(fourier_img[i][j]) < (p2*T):
            fourier_img[i][j] = 0
            cnt += 1

fourier_img = INV_DFT2D(fourier_img)

#imageio.imwrite('output_img.png', fourier_img.astype(np.uint8))

warnings.filterwarnings("ignore")
         
print("Threshold=%.4f" % (p2*T))
print("Filtered Coefficients=%d" % (cnt))
print("Original Mean=%.2f" % (np.mean(img)))
print("New Mean=%.2f" % (np.mean(fourier_img)))



end = time.time()
elapsed = end - start
    
#print("Running time: "+ str(elapsed) + "sec")
