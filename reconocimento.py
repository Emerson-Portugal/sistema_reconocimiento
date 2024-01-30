import cv2
import os

dataPath = r'C:\Users\TI-ElmersonP\Documents\codigo_BHC\asistencia_facial\data'
imagePaths = os.listdir(dataPath)
print('imagePath=', imagePaths)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('ModelFaceFrontaDataTest.xml')

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()

    faces = faceClassif.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in faces:
        rostro = auxFrame[y:y+h, x:x+w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        result = face_recognizer.predict(rostro)

        print("resultado", result[1])


        if result[1] < 100:
            cv2.putText(frame, 'Confianza: {:.2f}'.format(result[1]), (x, y-45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(frame, '{}'.format(imagePaths[result[0]]), (x, y-25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        else:
            cv2.putText(frame, 'DESCONOCIDO', (x,y-20), 2, 0.8, (0,0,255),1,cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255),2)

    cv2.imshow('frame', frame)
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
