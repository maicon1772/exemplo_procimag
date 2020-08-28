# -*- coding: cp1252 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt

def showImage(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()

def main():
    #carrega um classificador de um arquivo
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    #carrega um vídeo
    #cap = cv2.VideoCapture("rafinha.mp4")

    img1 = cv2.imread("imagensrostos/rosto2.jpg")
    img2 = cv2.imread("imagensrostos/rostocommasc1.jpg")
    img3 = cv2.imread("imagensrostos/rosto1.jpg")
    img4 = cv2.imread("imagensrostos/rostocommasc2.jpg")

    #while(True):
    #carrega o frame de vídeo
    #frameExiste, frame = cap.read()

    #chegou ao último frame ou houve erro? então sair!
    #if(frameExiste == False):
    #    cap.release()
    #    return

    #somente funciona com tons de cinza
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    gray4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)

    #detecta faces de diferentes tamanhos no frame de vídeo
    #primeiro parâmetro: a imagem

    #segundo parâmetro: especifica o quanto a imagem reduz
    #de tamanho durante a verificação

    #terceiro parâmetro: especifica quantos vizinhos
    #cada candidato a retângulo retêm
    faces1 = face_cascade.detectMultiScale(gray1, 1.3, 5)
    faces2 = face_cascade.detectMultiScale(gray2, 1.3, 5)
    faces3 = face_cascade.detectMultiScale(gray3, 1.3, 5)
    faces4 = face_cascade.detectMultiScale(gray4, 1.3, 5)

    #para cada face detectada
    for (x, y, w, h) in faces1:
        #Cortar imagem
        crop_img1 = img1[int(y+h*0.55):int(y+h*0.90), int(x+w*0.35):int(x+w*0.65)]
        #desenhe um retângulo (imagem, posição inicial, final, cor, espessura)
        img1 = cv2.rectangle(img1,(x,y),(x+w,y+h),(0,0,255),5)

    #para cada face detectada
    for (x, y, w, h) in faces2:
        #Cortar imagem
        crop_img2 = img2[int(y+h*0.55):int(y+h*0.90), int(x+w*0.35):int(x+w*0.65)]
        #desenhe um retângulo (imagem, posição inicial, final, cor, espessura)
        img2 = cv2.rectangle(img2,(x,y),(x+w,y+h),(0,0,255),5)
    #para cada face detectada
    for (x, y, w, h) in faces3:
        #Cortar imagem
        crop_img3 = img3[int(y+h*0.55):int(y+h*0.90), int(x+w*0.35):int(x+w*0.65)]
        #desenhe um retângulo (imagem, posição inicial, final, cor, espessura)
        img3 = cv2.rectangle(img3,(x,y),(x+w,y+h),(0,0,255),5)
    #para cada face detectada
    for (x, y, w, h) in faces4:
        #Cortar imagem
        crop_img4 = img4[int(y+h*0.55):int(y+h*0.90), int(x+w*0.35):int(x+w*0.65)]
        #desenhe um retângulo (imagem, posição inicial, final, cor, espessura)
        img4 = cv2.rectangle(img4,(x,y),(x+w,y+h),(0,0,255),5)

    #Histograma
    #color = ('b','g','r')
    #for i,col in enumerate(color):
    #    histr = cv2.calcHist([crop_img1],[i],None,[256],[0,256])
    #    plt.plot(histr,color = col)
    #    plt.xlim([0,256])
    #plt.show()
    # create a mask
    mask = np.zeros(crop_img1.shape[:2], np.uint8)
    mask[100:300, 100:400] = 255
    masked_img = cv2.bitwise_and(crop_img1,crop_img1,mask = mask)
    # Calculate histogram with mask and without mask
    # Check third argument for mask
    hist_full = cv2.calcHist([crop_img1],[0],None,[256],[0,256])
    hist_mask = cv2.calcHist([crop_img1],[0],mask,[256],[0,256])
    plt.subplot(421), plt.imshow(crop_img1)
    #plt.subplot(222), plt.imshow(mask,'gray')
    #plt.subplot(223), plt.imshow(masked_img, 'gray')
    plt.subplot(422), plt.plot(hist_full), plt.plot(hist_mask)
    #plt.xlim([0,256])
    #plt.show()

    # create a mask
    mask = np.zeros(crop_img2.shape[:2], np.uint8)
    mask[100:300, 100:400] = 255
    masked_img = cv2.bitwise_and(crop_img2,crop_img2,mask = mask)
    # Calculate histogram with mask and without mask
    # Check third argument for mask
    hist_full = cv2.calcHist([crop_img2],[0],None,[256],[0,256])
    hist_mask = cv2.calcHist([crop_img2],[0],mask,[256],[0,256])
    plt.subplot(423), plt.imshow(crop_img2)
    #plt.subplot(222), plt.imshow(mask,'gray')
    #plt.subplot(223), plt.imshow(masked_img, 'gray')
    plt.subplot(424), plt.plot(hist_full), plt.plot(hist_mask)
    #plt.xlim([0,256])
    #plt.show()

    # create a mask
    mask = np.zeros(crop_img3.shape[:2], np.uint8)
    mask[100:300, 100:400] = 255
    masked_img = cv2.bitwise_and(crop_img3,crop_img3,mask = mask)
    # Calculate histogram with mask and without mask
    # Check third argument for mask
    hist_full = cv2.calcHist([crop_img3],[0],None,[256],[0,256])
    hist_mask = cv2.calcHist([crop_img3],[0],mask,[256],[0,256])
    plt.subplot(425), plt.imshow(crop_img3)
    #plt.subplot(222), plt.imshow(mask,'gray')
    #plt.subplot(223), plt.imshow(masked_img, 'gray')
    plt.subplot(426), plt.plot(hist_full), plt.plot(hist_mask)
    #plt.xlim([0,256])
    #plt.show()

    # create a mask
    mask = np.zeros(crop_img4.shape[:2], np.uint8)
    mask[100:300, 100:400] = 255
    masked_img = cv2.bitwise_and(crop_img4,crop_img4,mask = mask)
    # Calculate histogram with mask and without mask
    # Check third argument for mask
    hist_full = cv2.calcHist([crop_img4],[0],None,[256],[0,256])
    hist_mask = cv2.calcHist([crop_img4],[0],mask,[256],[0,256])
    plt.subplot(427), plt.imshow(crop_img4)
    #plt.subplot(222), plt.imshow(mask,'gray')
    #plt.subplot(223), plt.imshow(masked_img, 'gray')
    plt.subplot(428), plt.plot(hist_full), plt.plot(hist_mask)
    plt.xlim([0,256])
    plt.show()


    #showImage(img1)
    #showImage(img2)
    #showImage(img3)
    #showImage(img4)


    #cv2.imshow("deteccao", img1)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break

    cv2.destroyAllWindows()

#if __name__ == "__main__":
main()
