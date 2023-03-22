#!/usr/bin/python
# -*- coding: utf-8 -*-

# Este NÃO é um programa ROS

from __future__ import print_function, division 

import cv2
import os,sys, os.path
import numpy as np

print("Rodando Python versão ", sys.version)
print("OpenCV versão: ", cv2.__version__)
print("Diretório de trabalho: ", os.getcwd())

# Arquivos necessários
model = os.path.join(os.getcwd(), "MobileNetSSD_deploy.caffemodel")
proto = os.path.join(os.getcwd(), "MobileNetSSD_deploy.prototxt.txt")

# Baixe o arquivo em https://github.com/Insper/robot20/blob/master/media/cow_wolf.mp4
video = "cow_wolf.mp4"

def detect(frame):
    """
        Recebe - uma imagem colorida
        Devolve: objeto encontrado
    """
    image = frame.copy()
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and
    # predictions
    print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()

    results = []

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence


        if confidence > CONFIDENCE:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # display the prediction
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            print("[INFO] {}".format(label))
            cv2.rectangle(image, (startX, startY), (endX, endY),
                COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

            results.append((CLASSES[idx], confidence*100, (startX, startY),(endX, endY) ))

    # show the output image
    return image, results

if __name__ == "__main__":

    # Inicializa a aquisição da webcam
    cap = cv2.VideoCapture(video)

    # cria a rede neural
    net = cv2.dnn.readNetFromCaffe(proto, model)

    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]   

    CONFIDENCE = 0.7
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))


    print("Se a janela com a imagem não aparecer em primeiro plano dê Alt-Tab")

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if ret == False:
            #print("Codigo de retorno FALSO - problema para capturar o frame")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            #sys.exit(0)

        # Our operations on the frame come here
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        saida, resultados = detect(frame)
        # results.append((CLASSES[idx], confidence*100, (startX, startY),(endX, endY) ))
        # make a bounding box around the objects identified as wolfs
        l_wolf = []; l_cow = []
        for r in resultados:
            coordinates = [r[2], r[3]]
            if r[0] == "cow":
                l_cow.append(coordinates)
            elif r[0] == "horse" or r[0] == "sheep":
                l_wolf.append(coordinates)
        if len(l_wolf) > 0:
            x_min = min([wolf[0][0] for wolf in l_wolf])
            y_min = min([wolf[0][1] for wolf in l_wolf])
            x_max = max([wolf[1][0] for wolf in l_wolf])
            y_max = max([wolf[1][1] for wolf in l_wolf])
            cv2.rectangle(saida, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
            wolf_bounding_box = (x_min, y_min, x_max, y_max)
            if len(l_cow) > 0:
                for cow in l_cow:
                    if cow[0][0] >= wolf_bounding_box[0] and cow[0][1] >= wolf_bounding_box[1] and cow[1][0] <= wolf_bounding_box[2] and cow[1][1] <= wolf_bounding_box[3]:
                        cv2.putText(saida, "EM PERIGO", (700, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                    else:
                        cv2.putText(saida, "Nao em perigo", (700, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

        # NOTE que em testes a OpenCV 4.0 requereu frames em BGR para o cv2.imshow
        cv2.imshow('imagem', saida)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

