
"""
Created on Thu May 14 17:37:47 2020

Beatriz Campos de Almeida de Castro Monteiro
NUSP: 9778619
Bacharel em Ciência da Computação - ICMC, USP São Carlos
SCC0251/5830— Prof. Moacir Ponti
2020.01

Short Assignment 1: Filtering in Fourier Domain

github link: https://github.com/biamont/PDI/tree/master/Fourirer%20Transformation

@author: Fuso
"""
import numpy as np
import imageio, time, warnings

#Fourier function
def DFT2D(f):
    F = np.zeros(f.shape, dtype=np.complex64)
    n,m = f.shape
    
    #creates row array for with values in range 0 to n for x and collum array with values 0 to m for y
    x = np.arange(n).reshape(n,1)
    y = np.arange(m).reshape(1, m)
    
    #optmizes a 4 for loop to calculate the fourier function
    for u in np.arange(n):
        for v in np.arange(m):
           F[u,v] = np.sum(f[x,y]*np.exp((-1j*2*np.pi)*(((u*x/n)+((v*y)/m)))))
    
    #normalizes result
    return F/np.sqrt(n*m)

#Inverse Fourier function
def INV_DFT2D(f):
    
    F = np.zeros(f.shape, dtype=np.complex64)
    n,m = f.shape
    
    #creates row array for with values in range 0 to n for x and collum array with values 0 to m for y
    x = np.arange(n).reshape(n,1)
    y = np.arange(m).reshape(1, m)
    
    #optmizes a 4 for loop to calculate the inverse fourier function
    for u in np.arange(n):
        for v in np.arange(m):        
            F[u,v] = np.sum(f[x,y]*np.exp((1j*2*np.pi)*(((u*x/n)+((v*y)/m)))))
    
    #normalizes result
    return F/np.sqrt(m*n)

def find_2ndMax(f):
    
    p1 = np.max(f)
    n,m = f.shape
    
    p2 = 0
    
    #run throught matrix checking for the second highest absolut value
    for i in range(n):
        for j in range(m):
            if np.real(np.abs(f[i][j])) > p2 and np.real(np.abs(f[i][j])) != p1:
                p2 = np.real(np.abs(f[i][j]))

    return p2
                
            
#MAIN

#reads image
filename = str(input()).rstrip()
input_img = imageio.imread(filename)

T = float(input()) #coeficient

img = np.array(input_img)
img = img.astype(np.float32) #casting para realizar as funcoes

start = time.time()

#creates matrix for image with fourier filter
n,m = img.shape
fourier_img = np.zeros((n,m), dtype=np.float)

#gets image with fourier filter
fourier_img = DFT2D(img)

#gets second peak of the fourier image
p2 = find_2ndMax(fourier_img)

cnt = 0 #counter

#turns to 0 all the values in fourier image below p2*T
for i in range(n):
    for j in range(m):
        if abs(fourier_img[i][j]) < (p2*T):
            fourier_img[i][j] = 0
            cnt += 1

#gets absolute valoue from image result of inverse fourier transformation
fourier_img = abs(INV_DFT2D(fourier_img))

#ignore warnings -> to make it possible to post in run.codes
warnings.filterwarnings("ignore")
         
print("Threshold=%.4f" % (p2*T))
print("Filtered Coefficients=%d" % (cnt))
print("Original Mean=%.2f" % (np.mean(img)))
print("New Mean=%.2f" % (np.mean(fourier_img)))

#calculates running time of the program
end = time.time()
elapsed = end - start
    
#print("Running time: "+ str(elapsed) + "sec")
