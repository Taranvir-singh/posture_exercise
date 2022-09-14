#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 12:49:19 2022

@author: pinkal
"""

import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose



def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle 
    return angle 

path = 'video/v3.mp4'
cap = cv2.VideoCapture(path)

counter = 0 
stage = None

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            print("There is some problem while opening the frame")
            break
        else:
            frame = cv2.resize(frame, (700,700))  
            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(frame)
            frame.flags.writeable=True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        
        left_shoulder = [result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x, result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
        right_shoulder = [result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
        left_hip = [result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x, result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y]
        right_hip = [result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x, result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y]
        left_knee = [result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x, result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y]
        right_knee = [result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x, result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y]

        angle = calculate_angle(left_shoulder,left_hip,left_knee)
        right_angle = calculate_angle(right_shoulder,right_hip,right_knee)

        # Curl counter logic
        if angle > 160:
            stage = "Stand"  #
        if angle < 90 and stage =='Stand':
            stage="Sit"
            counter +=1
            print(counter)
            
        

        cv2.rectangle(frame, (0,0), (225,73), (245,117,16), -1)
        cv2.putText(frame, str(stage), (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (255,255,0), 2, cv2.LINE_AA)
        cv2.putText(frame, str(counter), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (255,255,0), 2, cv2.LINE_AA)
        
        
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        cv2.imshow("video", frame)
        
cap.release()
cv2.destroyAllWindows()
    





