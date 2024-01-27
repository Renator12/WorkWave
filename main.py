import cv2
import time
import math as m
import mediapipe as mp
mp_pose=mp.solutions.pose
pose=mp_pose.Pose()
mp_hollistic=mp.solutions.holistic

def findDistance(x1,y1,x2,y2):
    dist2=m.sqrt((x2-x1)**2+(y2-y1)**2)
    return dist2
def findAngle(x1,y1,x2,y2):
    theta = m.acos((y2 - y1) * (-y1) / (m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
    degree = int(180 / m.pi) * theta
    return degree

def Warning():
    print('YOUR POSTURE IS INCORRECT')

#Frame counters
goodframes=0
badframes=0
font=cv2.FONT_HERSHEY_SIMPLEX
blue = (255, 127, 0)
red = (50, 50, 255)
green = (127, 255, 0)
dark_blue = (127, 20, 0)
light_green = (127, 233, 100)
yellow = (0, 255, 255)
pink = (255, 0, 255)
mp_drawing=mp.solutions.drawing_utils
BG_COLOR=(192,192,192)
#CV2 PART
cap=cv2.VideoCapture(0)
fps=int(cap.get(cv2.CAP_PROP_FPS))
width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_size=(width,height)

while cap.isOpened():
    success,image=cap.read()
    if not success:
        print('No frame')
        break
    h,w=image.shape[:2]
    keypoints=pose.process(image)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    lm=keypoints.pose_landmarks
    lmPose=mp_pose.PoseLandmark

    l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w) #normalizing wrt w
    l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h) ##normalizing wrt w
    # Right shoulder
    r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w) #normalize
    r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h) #normalize
    # Left ear.
    l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w) #normalize
    l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h) #normalize

    r_ear_x = int(lm.landmark[lmPose.RIGHT_EAR].x * w) #normalize
    r_ear_y = int(lm.landmark[lmPose.RIGHT_EAR].y * h) #normalize
    # NOSE.
    l_nose_x = int(lm.landmark[lmPose.NOSE].x * w)  #normalize
    l_nose_y = int(lm.landmark[lmPose.NOSE].y * h) #normalize
    # Calculate angles.
    l_neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
    print(l_neck_inclination)
    l_bend_inclination = findAngle(l_nose_x, l_nose_y, l_shldr_x, l_shldr_y)
    r_neck_inclination = findAngle(r_shldr_x, r_shldr_y, r_ear_x, r_ear_y)
    r_bend_inclination = findAngle(l_nose_x, l_nose_y, r_shldr_x, r_shldr_y)
    shldr_inclination = findAngle(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)


    #-------------------------

    cv2.circle(image, (l_shldr_x, l_shldr_y), 7, yellow, -1)
    cv2.circle(image, (r_shldr_x, r_shldr_y), 7, yellow, -1)
    cv2.circle(image, (l_ear_x, l_ear_y), 7, yellow, -1)
    cv2.circle(image, (r_ear_x, r_ear_y), 7, yellow, -1)
    # mp_drawing.plot_landmarks(keypoints.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
    z_nose = lm.landmark[lmPose.NOSE].z * w
    print(f'z: {lm.landmark[lmPose.NOSE].z * w}')
    if z_nose < -1100:
        cv2.putText(image, 'YOU ARE VERY CLOSER TO THE SCREEN', (10, 150), font, 0.9, red, 2)

    # Let's take y - coordinate of P3 100px above x1,  for display elegance.
    # Although we are taking y = 0 while calculating angle between P1,P2,P3.
    cv2.circle(image, (l_shldr_x, l_shldr_y - 100), 7, yellow, -1)
    cv2.circle(image, (r_shldr_x, r_shldr_y - 100), 7, pink, -1)
    cv2.circle(image, (l_nose_x, l_nose_y), 7, yellow, -1)

    # Similarly, here we are taking y - coordinate 100px above x1. Note that
    # you can take any value for y, not necessarily 100 or 200 pixels.
    cv2.circle(image, (l_nose_x, l_nose_y - 100), 7, yellow, -1)

    # Put text, Posture and angle inclination.
    # Text string for display.
    angle_text_left = 'Neck_left : ' + str(int(l_neck_inclination)) + '  Bend_left : ' + str(int(l_bend_inclination))
    angle_text_right = 'Neck_right : ' + str(int(r_neck_inclination)) + '  Bend_right : ' + str(int(r_bend_inclination))
    angle_text_string = 'Neck : ' + str(int(l_neck_inclination + r_neck_inclination)) + '  Bend : ' + str(
        int(l_bend_inclination + r_bend_inclination))
    angle_text_shldr = 'Shoulder : ' + str(int(shldr_inclination))
    neck = l_neck_inclination + r_neck_inclination
    bend = l_bend_inclination + r_bend_inclination
    x_m_point = int((l_shldr_x + r_shldr_x) / 2)
    y_m_point = int((l_shldr_y + r_shldr_y) / 2)

    cv2.imshow('my im',image)
cap.release()


