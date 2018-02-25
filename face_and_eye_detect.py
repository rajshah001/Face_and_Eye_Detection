import numpy as np  
import cv2

face_cascade = cv2.CascadeClassifier(r'C:\Users\Raj\Desktop\VS Code\opencv\haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier(r'C:\Users\Raj\Desktop\VS Code\opencv\haarcascade_eye_tree_eyeglasses.xml')

#smile_cascade = cv2.CascadeClassifier(r'C:\Users\Raj\Desktop\VS Code\opencv\haarcascade_smile.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.putText(frame, 'Face', (x, y-10), font, 1, (0,255,255), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,255), 3)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_color)

        for (ex,ey,ew,eh) in eyes:
            cv2.putText(roi_color, 'Eye', (ex, ey-10), font, 0.5, (0,255,0), 1, cv2.LINE_AA)
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 3)

        # smile = smile_cascade.detectMultiScale(roi_gray)

        # for (sx,sy,sw,sh) in smile:
        #     cv2.rectangle(roi_color, (sx,sy), (sx+sw, sy+sh), (0,0,255), 3)

    cv2.imshow('frame', frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
