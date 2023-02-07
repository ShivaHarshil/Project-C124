import cv2
import numpy as np
import tensorflow as tf
model = tf.keras.models.load_model("keras_model.h5")
vid = cv2.VideoCapture(0)
  
while(True):
    ret, frame = vid.read()
    img=cv2.resize(frame,(224,224))
    prediction = model.predict(img)
    print(prediction)
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    
    if key == 32:
        break
vid.release()
cv2.destroyAllWindows()
