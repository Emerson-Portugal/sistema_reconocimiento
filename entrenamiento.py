## entrenamiento.py
import cv2
import os
import numpy as np

dataPath = r'C:\Users\TI-ElmersonP\Documents\codigo_BHC\asistencia_facial\data'
peopleList = os.listdir(dataPath)

labels = []
facesData = []
label = 0

for nameDir in peopleList:
    personPath = dataPath + '/' + nameDir

    for fileName in os.listdir(personPath):
        labels.append(label)
        facesData.append(cv2.imread(personPath + '/' + fileName, 0))
        image = cv2.imread(personPath + '/' + fileName, 0)  # Este valor no se está utilizando en el entrenamiento

    label = label + 1


# Use cv2.face.createEigenFaceRecognizer() for both OpenCV 3.x and 4.x
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

print("Training....")
face_recognizer.train(facesData, np.array(labels))
face_recognizer.write('ModelFaceFrontaData.xml')
