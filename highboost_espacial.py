import cv2
import numpy as np
from argparse import ArgumentParser

imagem = cv2.imread('0340.TIF', 0)
cv2.imshow('Imagem original', imagem)
cv2.imwrite('original.jpg', imagem)

print("Altura: %d pixels" % (imagem.shape[0]))
print("Largura: %d pixels" % (imagem.shape[1]))

tam_mascara = 5		#int(input("Entre com o tamanho da mascara: "))
peso_mascara = tam_mascara * tam_mascara

mascara = np.arange(peso_mascara, dtype=float)
mascara = mascara.reshape(tam_mascara, tam_mascara)

index_mascara = tam_mascara/2
index_mascara = int(index_mascara)

x = np.arange(-index_mascara, index_mascara + 1)
y = np.arange(-index_mascara, index_mascara + 1)
x, y = np.meshgrid(y, x)

d0 = 20

mascara = np.exp(-((x**2) + (y**2))/(2*d0**2))/peso_mascara

copia2 = np.array(imagem, dtype=float)
copia = np.array(imagem, dtype=float)

i = 0
j = 0
colunas = []
linhas = []

#Percorrendo a matriz original para cálculo das médias
while i+tam_mascara <= imagem.shape[0]:
	while j+tam_mascara <= imagem.shape[1]:
		aux = imagem[i:tam_mascara+i, j:tam_mascara+j]
		aux = aux * mascara
		aux = aux.sum()
		#print(aux)
		colunas.append(aux)
		j += 1
	linhas.append(colunas)
	colunas = []
	i += 1
	j = 0


borrada = np.array(linhas, dtype=float)
copia[index_mascara:(imagem.shape[0]-index_mascara), index_mascara:(imagem.shape[1]-index_mascara)] = borrada

borrada = np.array(copia, dtype=np.uint8)
cv2.imshow('Imagem borrada', borrada) 

sub = copia2[::1, ::1] - copia[::1, ::1]
cv2.imshow('Imagem subtraida', sub)

nitida = copia2 + (2*sub)
nitida = np.real(nitida)
nitida = np.clip(nitida, 0, 255)
nitida = np.uint8(np.floor(nitida))

cv2.imshow('Imagem adicionada', nitida)
cv2.imwrite('hb_espacial.jpg',nitida)

cv2.waitKey(0)