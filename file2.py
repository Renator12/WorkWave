#EMOTION RECOGNITION AND MUSIC RECOMMENDER.Select timer and it sets a timedown timer which takes a picture after that time frame
#and emotions are found according to it

import cv2
import time
from fer import FER
import matplotlib.pyplot as plt
import operator

import matplotlib.image as mpimg
# SET THE COUNTDOWN TIMER
# for simplicity we set it to 3
# We can also take this as input
def piccapture(TIMER,name1='selfie'):
    # Open the camera
    cap = cv2.VideoCapture(0)
    while True:
        # Read and display each frame
        ret, img = cap.read()
        cv2.imshow('a', img)
        # check for the key pressed
        k = cv2.waitKey(125)
        # set the key for the countdown
        # to begin. Here we set q
        # if key pressed is q
        if k == ord('q'):
            prev = time.time()
            while TIMER >= 0:
                ret, img = cap.read()
                # Display countdown on each frame
                # specify the font and draw the
                # countdown using puttext
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, str(TIMER),
                            (200, 250), font,
                            7, (0, 255, 255),
                            4, cv2.LINE_AA)
                cv2.imshow('a', img)
                cv2.waitKey(125)
                # current time
                cur = time.time()
                # Update and keep track of Countdown
                # if time elapsed is one second
                # then decrease the counter
                if cur - prev >= 1:
                    prev = cur
                    TIMER = TIMER - 1
            else:
                ret, img = cap.read()
                # Display the clicked frame for 2
                # sec.You can increase time in
                # waitKey also
                cv2.imshow('a', img)
                # time for which image displayed
                cv2.waitKey(2000)
                # Save the frame
                cv2.imwrite(f'{name1}.jpg', img)
                cap.release()
                cv2.destroyAllWindows()
                # HERE we can reset the Countdown timer
                # if we want more Capture without closing
                # the camera
        # Press Esc to exit
        elif k == 27:
            break

input_image = cv2.imread("selfie.jpg")
emotion_detector = FER()
val=emotion_detector.detect_emotions(input_image)
dict1=val[0]['emotions']
print(dict1)
emotion=max(dict1, key=dict1.get)

