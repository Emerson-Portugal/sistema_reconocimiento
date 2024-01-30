import cv2
import os
import numpy as np

dataPath = r'C:\Users\TI-ElmersonP\Documents\codigo_BHC\asistencia_facial\data'
peopleList = os.listdir(dataPath)

labels = []
facesData = []
label = 0

for nameDir in peopleList:
    personPath = os.path.join(dataPath, nameDir)

    for fileName in os.listdir(personPath):
        labels.append(label)
        image = cv2.imread(os.path.join(personPath, fileName), cv2.IMREAD_GRAYSCALE)
        facesData.append(image)

    label += 1

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

print("Training....")
face_recognizer.train(facesData, np.array(labels))
face_recognizer.write('ModelFaceFrontaDataTest.xml')
