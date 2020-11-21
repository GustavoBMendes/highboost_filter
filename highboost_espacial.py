import cv2
import numpy as np
from argparse import ArgumentParser

imagem = cv2.imread('0340.TIF', 0)
cv2.imshow('Imagem original', imagem)

print("Altura: %d pixels" % (imagem.shape[0]))
print("Largura: %d pixels" % (imagem.shape[1]))

tam_mascara = 5#int(input("Entre com o tamanho da mascara: "))
peso_mascara = tam_mascara * tam_mascara

mascara = np.arange(peso_mascara, dtype=float)
mascara = mascara.reshape(tam_mascara, tam_mascara)
mascara[::1, ::1] = 1/peso_mascara

index_mascara = tam_mascara/2
index_mascara = int(index_mascara)

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
sub2 = np.array(sub, dtype=np.uint8)
cv2.imshow('Imagem subtraida', sub)

add = copia2 + (4*sub)
add = np.real(add)
add = np.clip(add, 0, 255)
add = np.uint8(np.floor(add))

cv2.imshow('Imagem adicionada', add)

cv2.waitKey(0)
