import cv2
import numpy as np
from argparse import ArgumentParser
from scipy.fftpack import fft

imagem = cv2.imread('0338.TIF', 0)
cv2.imshow('Imagem original', imagem)
copia = np.array(imagem, dtype=float)
#print(imagem)

d0 = 50

a = np.arange(-np.floor(imagem.shape[0]/2), np.ceil(imagem.shape[0]/2))
b = np.arange(-np.floor(imagem.shape[1]/2), np.ceil(imagem.shape[1]/2))

a, b = np.meshgrid(b, a)

h = np.exp(-((a**2) + (b**2))/(2*d0**2))

cv2.imshow('a', h)
#print(len(h))

fa = np.fft.fft2(imagem)
fa = np.fft.fftshift(fa)
fb = fa * h
#fb2 = np.array(fb, dtype=np.uint8)
#cv2.imshow('fb', fb2)
fb = np.fft.ifftshift(fb)
c = np.fft.ifft2(fb)
c = np.real(c)
c = np.clip(c, 0, 255)
c = np.uint8(np.floor(c))

cv2.imshow('mascara', c)

sub = copia - c
subcopia = np.array(sub, dtype=np.uint8)
cv2.imshow('sub', subcopia)

add = copia + (4*sub)
add = np.array(add, dtype=np.uint8)
cv2.imshow('add', add)

cv2.waitKey(0)
