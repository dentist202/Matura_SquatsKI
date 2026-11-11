# ============================================================
# Start: python main.py
#
# Video ODER Live: immer zwei Stellen umschalten.
# Was man benutzen will: ohne # . Das andere: mit # davor.
#
# VIDEO benutzen:
#   cap = cv2.VideoCapture("video/gym2_sp.mp4")
#   # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#   ...
#   timestamp_ms = int(frame_index * (1000 / fps))
#   # timestamp_ms = int((time.time() - start_time) * 1000)
#
# LIVE (Webcam) benutzen:
#   # cap = cv2.VideoCapture("video/gym2_sp.mp4")
#   cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#   ...
#   # timestamp_ms = int(frame_index * (1000 / fps))
#   timestamp_ms = int((time.time() - start_time) * 1000)
#
# Taste "q" beendet das Programm.
# ============================================================


import cv2                        #webcam + drawing on the image
import mediapipe as mp            #pose detection
import numpy as np                #math: vector, angle calculation
import matplotlib.pyplot as plt   #3D skeleton
from mpl_toolkits.mplot3d import Axes3D   #enables 3D plotting
import math #vector length
import pandas as pd
import time
import winsound

from mediapipe.tasks import python
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions


 
# ============================================================
#                          CONSTANTS
# ============================================================


# 33 body points:

# 0 - nose
# 11 - left shoulder
# 12 - right shoulder
# 13 - left elbow
# 14 - right elbow
# 15 - left wrist
# 16 - right wrist
# 23 - left hip
# 24 - right hip
# 25 - left knee
# 26 - right knee
# 27 - left ankle
# 28 - right ankle

LEFT_SHOULDER = 11 
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
RIGHT_HEEL = 30
RIGHT_FOOT_INDEX = 32


# Pairs of points for SKELETON:
POSE_CONNECTIONS = [
    (12, 24),                       # torso (shoulder + hip)
    (24, 26), (26, 28),             # leg   (hip + knee + ankle)
    (28, 30), (30, 32), (32, 28),   # foot  (ankle + heel + toe)
]
 

# Videos im Ordner "video" (Name oben bei cap eintragen):
#   richtig.mp4                  - 7 richtige Kniebeugen, alle grün
#   gym2_sp.mp4                  - 4 richtige Kniebeugen
#   fehler_zu_wenig_tief.mp4     - zu wenig tief      -> "Go deeper"
#   fehler_oberkoerper_vorne.mp4 - Rumpf zu weit vorne -> "Chest up"
#   fehler_fersen.mp4            - Fersen heben ab    -> "Keep heels down"
#   fehler_huefte_zuerst.mp4     - Hüfte zuerst hoch  -> "Hips too fast"
#   fehler_gewicht_zehen.mp4     - Gewicht auf Zehen  -> "Weight on toes"
# In jedem Fehlervideo ist eine Kniebeuge richtig


# ============================================================
#                             CAMERA / VIDEO 
# ============================================================

cap = cv2.VideoCapture("video/gym2_sp.mp4")
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # Open the webcam (0 = default camera)


cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


# ============================================================
#  SETUP: model(pose_landmarker.task),
# ============================================================
 
 

# Mode_Setup (Video). 
# Video mode tracks landmarks across frames with timestamps (points are more predictable)
base_options = python.BaseOptions(model_asset_path='pose_landmarker_full.task')
options = PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=mp.tasks.vision.RunningMode.VIDEO, #VIDEO instead of IMAGE
)
pose_landmarker = PoseLandmarker.create_from_options(options)
 



# ============================================================
#  OPENCV
# ============================================================


 
# cv2.setWindowProperty("Squats with AI", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.namedWindow("Squats with AI", cv2.WINDOW_AUTOSIZE)
fps = cap.get(cv2.CAP_PROP_FPS)

# <<3D-Matplotlib>>

# # 3D window for the skeleton
# plt.ion()                                 
# fig = plt.figure()                         # an empty figure without Axes
# ax = fig.add_subplot(111, projection="3d")
 
 # <<3D-Matplotlib>>
 
# ============================================================
#                        FUNCTIONS
# ============================================================
 
def point_above(point, distance):
    # only y coordinate changes
    # media pipe Y is negativ, so we ahve to - distance
    return [point[0], point[1] - distance, point[2]]

def calculate_height(world_landmarks):
    #Зріст= сума довжин сегментів вздовж тіла (права сторона):
    #гомілковостоп- коліно -стегно - плече - око - верх голови
    
    segments_right_side = [
        (28, 26),  # shin:        ankle → knee
        (26, 24),  # thigh:       knee → hip
        (24, 12),  # torso:       hip → shoulder
        (12, 0),   # neck+head:   shoulder → nose
    ]

    segments_left_side = [
        (27, 25),  # shin:        ankle → knee
        (25, 23),  # thigh:       knee → hip
        (23, 11),  # torso:       hip → shoulder
        (11, 0),   # neck+head:   shoulder → nose
        ]
    
    total_left_side = 0
    total_right_side = 0
    for p1, p2 in segments_right_side:
        start = np.array([world_landmarks[p1].x, 
                          world_landmarks[p1].y, 
                          world_landmarks[p1].z])
        end = np.array([world_landmarks[p2].x, 
                        world_landmarks[p2].y, 
                        world_landmarks[p2].z])
        total_right_side += np.linalg.norm(end - start)
    
    total_right_side += 0.10

    for p1, p2 in segments_left_side:
        start = np.array([world_landmarks[p1].x, 
                          world_landmarks[p1].y, 
                          world_landmarks[p1].z])
        end = np.array([world_landmarks[p2].x, 
                        world_landmarks[p2].y, 
                        world_landmarks[p2].z])
        total_left_side += np.linalg.norm(end - start)
    
    total_left_side += 0.10


    return total_left_side, total_right_side


def calculate_angle(point_a, point_b, point_c):

    point_a = np.array(point_a)
    point_b = np.array(point_b)
    point_c = np.array(point_c)
 
    vector_ba = point_a - point_b
    vector_bc = point_c - point_b
 
    #skalarprodukt
    cosine_angle = np.dot(vector_ba, vector_bc) / (
        np.linalg.norm(vector_ba) * np.linalg.norm(vector_bc)
    )

    # cos can be only between -1 nd 1
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
 
    return np.degrees(np.arccos(cosine_angle))


def point_above_2d(point, distance):
    # у пікселях y росте вниз, тому "вгору" = менший y
    return [point[0], point[1] - distance, 0]

def calculate_angle_2d(point_a, point_b, point_c):
    return calculate_angle(
        [point_a[0], point_a[1], 0],
        [point_b[0], point_b[1], 0],
        [point_c[0], point_c[1], 0],
    )

def landmarks_2d_to_dict(landmarks_2d, frame_w, frame_h):
    positions = {}
    for index, landmark in enumerate(landmarks_2d):
        positions[index] = [landmark.x * frame_w, landmark.y * frame_h]
    return positions


 
# ============================================================
#                        Landmarks
# ============================================================

#store landmarks in a dict {index:[x,y,z]}
def landmarks_to_dict(world_landmarks):

    positions = {}

    for index, landmark in enumerate(world_landmarks):

        positions[index] = [landmark.x, landmark.y, landmark.z]
    return positions
 


#to draw 2d skeleton on the screen (landmark(0-1)*screen_resolution)
def draw_skeleton_2d(frame, landmarks):

    right_points = {11, 13, 15, 23, 25, 27}

    for point_1, point_2 in POSE_CONNECTIONS:

        if point_1 in right_points or point_2 in right_points:
            color = (255, 255, 255)

        else:
            color = (41, 35, 35)
            
        start = (int(landmarks[point_1].x * frame.shape[1]),
                 int(landmarks[point_1].y * frame.shape[0]))
        
        end = (int(landmarks[point_2].x * frame.shape[1]),
               int(landmarks[point_2].y * frame.shape[0]))
        cv2.line(frame, start, end, color, 4)


def length_3d(world_landmarks, point_1, point_2):
        start = (float(world_landmarks[point_1].x),
                 float(world_landmarks[point_1].y),
                 float(world_landmarks[point_1].z))

        
        end = (float(world_landmarks[point_2].x),
                float(world_landmarks[point_2].y),
                float(world_landmarks[point_2].z))

        length_3d = math.dist(start, end)

        return length_3d

def length_2d(landmarks_2d, point_1, point_2, frame_w, frame_h):
    start = (
        float(landmarks_2d[point_1].x * frame_w),
        float(landmarks_2d[point_1].y * frame_h)
    )

    end = (
        float(landmarks_2d[point_2].x * frame_w),
        float(landmarks_2d[point_2].y * frame_h)
    )

    return math.dist(start, end)

 # 2D

def calculate_height_2d(landmarks_2d, frame_w, frame_h):

    segments_right_side = [
        (28, 26),
        (26, 24),
        (24, 12),
        (12, 0)
    ]

    segments_left_side = [
        (27, 25),
        (25, 23),
        (23, 11),
        (11, 0)
    ]

    total_right_side = 0
    total_left_side = 0

    for p1, p2 in segments_right_side:
        total_right_side += length_2d(
            landmarks_2d, p1, p2, frame_w, frame_h
        )

    for p1, p2 in segments_left_side:
        total_left_side += length_2d(
            landmarks_2d, p1, p2, frame_w, frame_h
        )

    return total_left_side, total_right_side



def csv_list(all_frames, frame_index, timestamp_ms,
             knee_angle_left_side, knee_angle_right_side,
             torso_angle_left_side, torso_angle_right_side,
             knee_angle_left_side_2d, knee_angle_right_side_2d,
             torso_angle_left_side_2d, torso_angle_right_side_2d,
             current_height_left_side, current_height_right_side,
             world_landmarks, landmarks_2d, frame_w, frame_h,current_height_left_side_2d, current_height_right_side_2d,
             thigh_left_2d, thigh_right_2d,
             shin_left_2d, shin_right_2d,
             shoulder_width_2d, hip_width_2d, time_in_box_ms, time_sideways_ms, app_state, rep_phase, rep_count):

    row = {
        "frame": frame_index,
        "time_ms": timestamp_ms,
        "knee_angle_left_side": knee_angle_left_side,
        "knee_angle_right_side": knee_angle_right_side,
        "torso_angle_left_side": torso_angle_left_side,
        "torso_angle_right_side": torso_angle_right_side,
        "knee_angle_left_side_2d": knee_angle_left_side_2d,
        "knee_angle_right_side_2d": knee_angle_right_side_2d,
        "torso_angle_left_side_2d": torso_angle_left_side_2d,
        "torso_angle_right_side_2d": torso_angle_right_side_2d,

        # <<3D-Values>>
        "thigh_left_3d":     length_3d(world_landmarks, 23, 25),
        "thigh_right_3d":    length_3d(world_landmarks, 24, 26),
        "shin_left_3d":      length_3d(world_landmarks, 25, 27),
        "shin_right_3d":     length_3d(world_landmarks, 26, 28),
        "shoulder_width_3d": length_3d(world_landmarks, 11, 12),
        "hip_width_3d":      length_3d(world_landmarks, 23, 24),
        "height_left_3d": current_height_left_side,
        "height_right_3d": current_height_right_side,
        # <<3D-Values>>

        "thigh_left_2d": thigh_left_2d,
        "thigh_right_2d": thigh_right_2d,
        "shin_left_2d": shin_left_2d,
        "shin_right_2d": shin_right_2d,
        "shoulder_width_2d": shoulder_width_2d,
        "hip_width_2d": hip_width_2d,
        "height_left_2d": current_height_left_side_2d,
        "height_right_2d": current_height_right_side_2d,
        "time_in_box_ms": time_in_box_ms,
        "time_sideways_ms": time_sideways_ms,
        "app_state": app_state,
        "rep_phase": rep_phase,
        "rep_count": rep_count
    }

    for i in range(33):
        row[f"landmark_x{i}"] = world_landmarks[i].x
        row[f"landmark_y{i}"] = world_landmarks[i].y
        row[f"landmark_z{i}"] = world_landmarks[i].z
        row[f"3d_landmark_{i}_visibility"] = world_landmarks[i].visibility
        row[f"2d_landmark_{i}_visibility"] = landmarks_2d[i].visibility

        #2D Pixel -coordinates
        
        row[f"landmark_2d_x{i}"] = landmarks_2d[i].x * frame_w
        row[f"landmark_2d_y{i}"] = landmarks_2d[i].y * frame_h


    all_frames.append(row)

 
# ============================================================
#                        POSITION CHECK
# ============================================================


def person_in_box_check(frame, frame_w, frame_h, landmarks_2d, timestamp_ms, in_box_timer_start_ms):

    x1 = frame_w *0.30 #0.35 center
    x2 = frame_w *0.65 #0.65 center 
    y1 = frame_h *0.01
    y2 = frame_h *0.99

    required_points = {
        0: "NOSE",
        1: "LEFT_EYE_INNER",
        4: "RIGHT_EYE_INNER",
        7: "LEFT_EAR",
        8: "RIGHT_EAR",
        11: "LEFT_SHOULDER",     # left shoulder
        12: "RIGHT_SHOULDER",      # right shoulder
        23: "LEFT_HIP",   # left hip
        24: "RIGHT_HIP",    # right hip
        25: "LEFT_KNEE",   # left knee
        26: "RIGHT_KNEE",    # right knee
        27: "LEFT_ANKLE",   # left ankle
        28: "RIGHT_ANKLE",    # right ankle
        29: "LEFT_HEEL",
        30: "RIGHT_HEEL",
        31: "LEFT_FOOT_INDEX",
        32: "RIGHT_FOOT_INDEX",
    
    }

    for i in required_points:
        user_position_x = landmarks_2d[i].x * frame_w
        user_position_y = landmarks_2d[i].y * frame_h

        if x1<user_position_x<x2 and y1<user_position_y<y2:
            # is_person_in_box = True
            all_points_in_box_state = True

        else:
            all_points_in_box_state = False

            break

    if all_points_in_box_state:
        if in_box_timer_start_ms == 0:
            in_box_timer_start_ms = timestamp_ms
        time_in_box_ms = timestamp_ms - in_box_timer_start_ms
    else:
        in_box_timer_start_ms= 0
        time_in_box_ms = 0


    return all_points_in_box_state, in_box_timer_start_ms, time_in_box_ms


def side_check(landmarks_2d, timestamp_ms, side_timer_start_ms):
    
    if landmarks_2d[25].visibility < 0.75:
        if side_timer_start_ms == 0:
            side_timer_start_ms = timestamp_ms
        person_time_sideways = timestamp_ms - side_timer_start_ms
    else:
        side_timer_start_ms =0
        person_time_sideways = 0


    return side_timer_start_ms, person_time_sideways


def grade(value, green_limit, yellow_limit):
    if value <= green_limit:
        return "green"
    elif value <= yellow_limit:
        return "yellow"
    else:
        return "red"

# Feedback, Startbildschirm und Resultat-Tabelle

def rep_feedback(result):
    colors = [result["K2_color"], result["K4_color"], result["K5_color"], result["K6_color"], result["K7_color"]]

    if "red" in colors:
        worst = "red"

    elif "yellow" in colors:
        worst = "yellow"

    else:
        return "green", "Good squat"

    if result["K2_color"] == worst:
        return worst, "Go deeper"
    
    if result["K4_color"] == worst:
        return worst, "Chest up"
    
    if result["K5_color"] == worst:
        return worst, "Keep heels down"
        
    if result["K6_color"] == worst:
        return worst, "Hips too fast"

    if result["K7"] > 0:
        return worst, "Weight on toes"
    else:
        return worst, "Weight on heels" 

def rep_text(result):
    text = ""

    if result["K2_color"] != "green":
        text = text + "Go deeper. "
    if result["K4_color"] != "green":
        text = text + "Chest up. "
    if result["K5_color"] != "green":
        text = text + "Keep heels down. "
    if result["K6_color"] != "green":
        text = text + "Hips too fast. "

    if result["K7_color"] == "yellow" or result["K7_color"] == "red":
        if result["K7"] > 0:
            text = text + "Weight on toes. "
        else:
            text = text + "Weight on heels. "

    if text == "":
        text = "Good squat."

    return text

def show_results(rep_results):
    colors = {"green": (0, 140, 0), "yellow": (0, 170, 220), "red": (0, 0, 180)}
    black = (0, 0, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    screen = np.full((50 * (len(rep_results) + 2), 1200, 3), 255, dtype=np.uint8)

    cv2.putText(screen, f"You did {len(rep_results)} squats.", (30, 50), font, 1.2, black, 3)

    y = 60
    for result in rep_results:
        y = y + 50
        worst, tip = rep_feedback(result)
        cv2.putText(screen, f"Squat {result['rep']}: {rep_text(result)}", (30, y), font, 0.9, colors[worst], 2)

    cv2.imshow("Results", screen)
    cv2.waitKey(0)

def on_mouse(event, x, y, flags, param):
    global start_clicked
    if event == cv2.EVENT_LBUTTONDOWN and 450 < x < 750 and 770 < y < 860:
        start_clicked = "yes"




def show_start_screen():
    screen = np.full((900, 1200, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    black = (0, 0, 0)

    rules = [
        "Before you start:",
        "1. Find a free space. Your whole body must be visible.",
        "2. Wear shorts or tight leggings. No wide trousers.",
        "3. Use an empty barbell on your shoulders, no plates.",
        "4. This app is for healthy beginners.",
        "",
        "How it works:",
        "1. Step into the green box and hold still.",
        "2. Turn your right side to the camera and hold still.",
        "3. Do your squats. After each squat you see a colour and a tip.",
        "4. Step out of the box to finish and see your results.",
    ]

    y = 70
    for line in rules:
        cv2.putText(screen, line, (50, y), font, 0.9, black, 2)
        y = y + 60

    cv2.rectangle(screen, (450, 770), (750, 860), (0, 140, 0), -1)
    cv2.putText(screen, "START", (525, 830), font, 1.5, (255, 255, 255), 4)

    cv2.setMouseCallback("Squats with AI", on_mouse)

    while start_clicked == "no":
        cv2.imshow("Squats with AI", screen)
        cv2.waitKey(30)


# ============================================================
#                             MAIN LOOP
# ============================================================
start_clicked = "no"
all_frames = []



def main():

    show_start_screen()


    app_state = "SETUP_BOX"
    SIDE_REQUIRED_MS = 3000
    BOX_REQUIRED_MS = 1500
    LEFT_BOX_END_MS = 2000

    PHASE_TITLES = {
    "SETUP_BOX":  "Step 1/3: Green box",
    "SETUP_SIDE": "Step 2/3: Side view",
    "TRAINING":   "Step 3/3: Squats",
    "RESULTS":    "Finished",
}

    REP_DOWN_ANGLE = 110
    REP_UP_ANGLE = 150
    STAND_ANGLE = 160


    frame_index = 0    #count frames() timestamp
    graph_frame = 0  

    #Box, Sideway, Leave (Timers)
    in_box_timer_start_ms= 0 
    side_timer_start_ms = 0
    person_insde_box_time = 0
    time_sideways_ms = 0
    left_box_timer_start_ms = 0

    #reps
    rep_count= 0
    rep_phase = "UP"

    # werte zu jedem Frame (nur beim stehen)
    stand_heel_x = []
    stand_heel_y = []
    stand_toe_x = []
    stand_toe_y= []
    stand_thigh_length = []

    # werte zu jedem Frame (nur kniebeuge)
    rep_knee_angles = []
    rep_k2_values = []
    rep_torso_angles = []
    rep_heel_y = []
    rep_shoulder_x = []
    rep_hip_angles = []
    rep_shin_angles = []

    # alle resultaten
    rep_results = []

    feedback_color = "green"
    feedback_text = ""

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Camera error")
            break

        frame_h, frame_w = frame.shape[:2]
        is_person_in_box = False
        panel_message = "Step into the green box"
        panel_is_green = False  
        time_in_box_ms = 0
        time_sideways_ms = 0


        #OpenCV(BGR) convert to MediaPipe(RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
 
        
        frame_index += 1



# ============================================================
#                             CAMERA / VIDEO timestamp
# ============================================================

        timestamp_ms = int(frame_index * (1000 / fps))  # Video
        # timestamp_ms = int((time.time() - start_time) * 1000) # Live






# ============================================================
#                             MAIN LOOP
# ============================================================
        results = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
 
        #2D: draw skeleton on the screen + landmarks on the scrren
        if results.pose_landmarks:
            landmarks_2d = results.pose_landmarks[0]
            draw_skeleton_2d(frame, landmarks_2d)


            # Points on the right side of the body

            # right side of the body in MediaPipe
            right_points = {12, 24, 26, 28, 30, 32}

            for index, landmark in enumerate(landmarks_2d):
                if index in right_points:


                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 7, (255, 200, 0), -1)

        #         csv_list()


 

        #3D: draw skeleton on the graph + right knee angle
        if results.pose_world_landmarks and results.pose_landmarks:
            world_landmarks = results.pose_world_landmarks[0]
            landmarks_2d = results.pose_landmarks[0]
            landmark_positions = landmarks_to_dict(world_landmarks)

            current_height_left_side, current_height_right_side   = calculate_height(world_landmarks)

            graph_frame += 1                        
            n = 1 # to change in what frequency should points be displayed

            # <<3D-Matplotlib>>
            # if graph_frame % n == 0:                
                # draw_skeleton_3d(ax, world_landmarks)  
                
            # <<3D-Matplotlib>>

            

            knee_angle_right_side = calculate_angle(
                landmark_positions[RIGHT_HIP],
                landmark_positions[RIGHT_KNEE],
                landmark_positions[RIGHT_ANKLE],
            )

            knee_angle_left_side = calculate_angle(
                            landmark_positions[LEFT_HIP],
                            landmark_positions[LEFT_KNEE],
                            landmark_positions[LEFT_ANKLE],
                        )
            
            vertical_point_right_side = point_above(landmark_positions[RIGHT_HIP], distance = 0.5)
            vertical_point_left_side = point_above(landmark_positions[LEFT_HIP], distance = 0.5)

            torso_angle_right_side = calculate_angle(
                landmark_positions[RIGHT_SHOULDER],
                landmark_positions[RIGHT_HIP],                                  
                vertical_point_right_side, 
            )

            torso_angle_left_side = calculate_angle(
                landmark_positions[LEFT_SHOULDER],
                landmark_positions[LEFT_HIP],
                vertical_point_left_side,
            )

            # 2D

            frame_w = frame.shape[1]
            frame_h = frame.shape[0]
            landmark_positions_2d = landmarks_2d_to_dict(landmarks_2d, frame_w, frame_h)

            knee_angle_right_side_2d = calculate_angle_2d(
                landmark_positions_2d[RIGHT_HIP],
                landmark_positions_2d[RIGHT_KNEE],
                landmark_positions_2d[RIGHT_ANKLE],
            )
            knee_angle_left_side_2d = calculate_angle_2d(
                landmark_positions_2d[LEFT_HIP],
                landmark_positions_2d[LEFT_KNEE],
                landmark_positions_2d[LEFT_ANKLE],
            )

            vertical_point_right_side_2d = point_above_2d(landmark_positions_2d[RIGHT_HIP], distance=frame_h / 2)
            vertical_point_left_side_2d = point_above_2d(landmark_positions_2d[LEFT_HIP], distance=frame_h / 2)

            torso_angle_right_side_2d = calculate_angle_2d(
                landmark_positions_2d[RIGHT_SHOULDER],
                landmark_positions_2d[RIGHT_HIP],
                vertical_point_right_side_2d,
            )
            torso_angle_left_side_2d = calculate_angle_2d(
                landmark_positions_2d[LEFT_SHOULDER],
                landmark_positions_2d[LEFT_HIP],
                vertical_point_left_side_2d,
            )

            thigh_left_2d = length_2d(landmarks_2d, 23, 25, frame_w, frame_h)
            thigh_right_2d = length_2d(landmarks_2d, 24, 26, frame_w, frame_h)

            shin_left_2d = length_2d(landmarks_2d, 25, 27, frame_w, frame_h)
            shin_right_2d = length_2d(landmarks_2d, 26, 28, frame_w, frame_h)

            shoulder_width_2d = length_2d(landmarks_2d, 11, 12, frame_w, frame_h)
            hip_width_2d = length_2d(landmarks_2d, 23, 24, frame_w, frame_h)

            current_height_left_side_2d, current_height_right_side_2d = calculate_height_2d(
                landmarks_2d, frame_w, frame_h)





            is_person_in_box, in_box_timer_start_ms, time_in_box_ms = person_in_box_check(frame, frame_w, frame_h, landmarks_2d, timestamp_ms, in_box_timer_start_ms)

            if app_state == "SETUP_BOX":
                if is_person_in_box:
                    panel_message = f"Hold still 1.5s: {time_in_box_ms/1000:.1f} s"
                    panel_is_green = True

                if time_in_box_ms >= BOX_REQUIRED_MS:
                    app_state = "SETUP_SIDE"
                    side_timer_start_ms = 0

            elif app_state == "SETUP_SIDE":
                if not is_person_in_box:
                    app_state = "SETUP_BOX"
                else:
                    side_timer_start_ms, time_sideways_ms = side_check(landmarks_2d, timestamp_ms, side_timer_start_ms)

                    if landmarks_2d[LEFT_KNEE].visibility < 0.75:
                        panel_message = f"Stay sideways 3s: {time_sideways_ms/1000:.1f} s"
                        panel_is_green = True

                        if time_sideways_ms >= SIDE_REQUIRED_MS:
                            app_state = "TRAINING"
                    else:
                        panel_message = "Right side to camera"
                        

            elif app_state == "TRAINING":
                #werte zu jedem Frames

                #[x,y] a, b, c, d = [1, 2, 3, 4]  -> # a = 1, b = 2, c = 3, d = 4
                
                heel_x, heel_y = landmark_positions_2d[RIGHT_HEEL] #k5, k7
                toe_x, toe_y = landmark_positions_2d[RIGHT_FOOT_INDEX] #k7
                shoulder_x = landmark_positions_2d[RIGHT_SHOULDER][0] #k7


                hip_angle = calculate_angle_2d(landmark_positions_2d[RIGHT_SHOULDER], landmark_positions_2d[RIGHT_HIP], landmark_positions_2d[RIGHT_KNEE])
                
                ankle = landmark_positions_2d[RIGHT_ANKLE]
                shin_angle = calculate_angle_2d(landmark_positions_2d[RIGHT_KNEE], ankle, point_above_2d(ankle, frame_h / 2))

                if rep_phase == "UP" and knee_angle_right_side_2d > STAND_ANGLE:
                    stand_heel_x.append(heel_x)
                    stand_heel_y.append(heel_y)
                    stand_toe_x.append(toe_x)
                    stand_toe_y.append(toe_y)
                    stand_thigh_length.append(thigh_right_2d)

                if rep_phase == "UP" and knee_angle_right_side_2d < REP_DOWN_ANGLE:
                    rep_phase = "DOWN"

                    #Kniebeugephase, tiefster Punkt (k1,k2, k3,k4), ab tiefstem Punkt bis Ende: k6,  ganze phase: k5, k7
                    rep_knee_angles = []
                    rep_k2_values = []
                    rep_torso_angles = []
                    rep_heel_y = []
                    rep_shoulder_x = []
                    rep_hip_angles = []
                    rep_shin_angles = []


                elif rep_phase == "DOWN":
                    rep_knee_angles.append(knee_angle_right_side_2d)
                    height_difference = landmark_positions_2d[RIGHT_KNEE][1] - landmark_positions_2d[RIGHT_HIP][1]
                    k2_value = height_difference / np.median(stand_thigh_length[-45:]) * 100
                    rep_k2_values.append(k2_value)
                    rep_torso_angles.append(torso_angle_right_side_2d)
                    
                    rep_heel_y.append(heel_y)
                    rep_shoulder_x.append(shoulder_x)
                    rep_hip_angles.append(hip_angle)
                    rep_shin_angles.append(shin_angle)

                    if knee_angle_right_side_2d > REP_UP_ANGLE: #Kniebeuge: ab 110 nach unten, bis 150 nach oben
                        rep_phase = "UP"
                        rep_count += 1

                        bottom = int(np.argmin(rep_k2_values)) #index für den tiefsten Punkt
                        k2_bottom = float(np.median(rep_k2_values[bottom- 2:bottom+3])) #5 indexes in den tiefsten Punkt -> 5 aufeinanderfolgende werte -> median
                        
                        torso_smooth = pd.Series(rep_torso_angles).rolling(5, center=True, min_periods=1).median().values #smooth torso angles: median over 5 frames for each frame, removes single jumps

                        
                        k4_bottom = float(torso_smooth[bottom])
                        k6_bottom = float(torso_smooth[bottom:].max()- torso_smooth[bottom])
                        

                        #Letzte 45 Bilder (ca. 1,5 s bei 30 fps): nur der Stand direkt vor dieser Kniebeuge

                        heel_x_stand = np.array(stand_heel_x[-45:]) 
                        heel_y_stand = np.array(stand_heel_y[-45:])
                        toe_x_stand= np.array(stand_toe_x[-45:])
                        toe_y_stand= np.array(stand_toe_y[-45:])
                        thigh_stand= np.array(stand_thigh_length[-45:])

                        heel_smooth =pd.Series(rep_heel_y).rolling(5, center=True, min_periods=1).median().values
                        
                        k5_max = float((np.median(heel_y_stand)- heel_smooth.min())/np.median(thigh_stand)* 100)

                        foot_length = np.median(np.hypot(toe_x_stand - heel_x_stand, toe_y_stand - heel_y_stand))
                        foot_mid = np.median((heel_x_stand + toe_x_stand) / 2)
                        

                        shoulder_smooth = pd.Series(rep_shoulder_x).rolling(5, center=True, min_periods=1).median().values
                        k7_values = (shoulder_smooth - foot_mid) / foot_length * 100 # [5,4,3] - 2 = [3,2,1]

                        k7_max = float(k7_values[np.argmax(np.abs(k7_values))]) #grösste Abweichung von der Fussmitte, mit Vorzeichen

                        k1_bottom = float(np.median(rep_knee_angles[bottom- 2:bottom + 3]))# nur als zahl 
                        k3_bottom = float(np.median(rep_hip_angles[bottom- 2:bottom + 3])) - k1_bottom# nur als zahl


                        # Schienbeinneigung unten: ab 45° (sehr tief) wird K7 nur als Zahl gezeigt
                        shin_bottom = float(np.median(rep_shin_angles[bottom- 2:bottom+ 3]))

                        if shin_bottom< 45:
                            k7_color = grade(abs(k7_max), 25,40)
                        else:
                            k7_color = "number"


                        rep_results.append({
                            "rep": rep_count,
                            "K1": k1_bottom,
                            "K2": k2_bottom, "K2_color": grade(k2_bottom, 15, 30),
                            "K3": k3_bottom,
                            "K4": k4_bottom, "K4_color": grade(k4_bottom, 40, 45),
                            "K5": k5_max, "K5_color": grade(k5_max, 10, 20),
                            "K6": k6_bottom, "K6_color": grade(k6_bottom, 5, 10),
                            "K7": k7_max, "K7_color": k7_color,
                        })

                        print(rep_results[-1])
                        feedback_color, feedback_text = rep_feedback(rep_results[-1])
                        if feedback_color == "red":
                            winsound.Beep(600, 200)

                panel_message = f"Squats: {rep_count}"
                panel_is_green = True
                




            
            if app_state == "TRAINING":
                csv_list(
                    all_frames, frame_index, timestamp_ms,

                    knee_angle_left_side, knee_angle_right_side,
                    torso_angle_left_side, torso_angle_right_side,

                    knee_angle_left_side_2d, knee_angle_right_side_2d,
                    torso_angle_left_side_2d, torso_angle_right_side_2d,

                    current_height_left_side, current_height_right_side,

                    world_landmarks, landmarks_2d, frame_w, frame_h,

                    current_height_left_side_2d, current_height_right_side_2d,

                    thigh_left_2d, thigh_right_2d,
                    shin_left_2d, shin_right_2d,
                    shoulder_width_2d, hip_width_2d,
                    time_in_box_ms, time_sideways_ms, 
                    app_state, rep_phase, rep_count
                )

        else:
            in_box_timer_start_ms = 0
            side_timer_start_ms = 0

        if app_state == "TRAINING":
            if is_person_in_box:
                left_box_timer_start_ms = 0
            else:
                if left_box_timer_start_ms == 0:
                    left_box_timer_start_ms = timestamp_ms
                if timestamp_ms - left_box_timer_start_ms >= LEFT_BOX_END_MS:
                    app_state = "RESULTS"

        if app_state == "RESULTS":
            break
            
            # cv2.putText(frame, f"Right Knee Angle: {knee_angle_right_side:.1f}", (50, 100),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # cv2.putText(frame, f"Right Torso Lean Angle: {torso_angle_right_side:.1f}", (50, 200),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # cv2.putText(frame, f"Height_Left_Side: {current_height_left_side*100:.0f} cm", (50, 300),
            #                                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # cv2.putText(frame, f"Height_Right_Side: {current_height_right_side*100:.0f} cm", (50, 400),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


        if app_state == "TRAINING" and feedback_color == "yellow":
            panel_color = (0, 170, 220)#yellow
        
        elif app_state == "TRAINING" and feedback_color == "red":
            panel_color = (0, 0, 180)#red
        
        elif panel_is_green:
            panel_color = (0, 140, 0)#green
        
        else:
            panel_color = (0, 0, 180)#red

        # Panel (Overlay)

        if app_state == "SETUP_BOX" or app_state == "SETUP_SIDE":
            cv2.rectangle(frame, (int(frame_w * 0.30), int(frame_h * 0.01)), (int(frame_w * 0.65), int(frame_h * 0.99)), (0, 255, 0), 2)

        PANEL_X0 = int(frame_w * 0.65)
        text_size = frame_w / 1920
        thickness = max(2, int(4 * text_size))
        

        overlay = frame.copy()
        cv2.rectangle(overlay, (PANEL_X0, 0), (frame_w, frame_h), panel_color, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        cv2.putText(frame, PHASE_TITLES[app_state], (PANEL_X0 + 20, int(100 * text_size)), cv2.FONT_HERSHEY_SIMPLEX, 1.3 * text_size, (255, 255, 255), thickness)



        cv2.putText(frame, panel_message, (PANEL_X0 + 20, int(200 * text_size)), cv2.FONT_HERSHEY_SIMPLEX, 1.3 * text_size, (255, 255, 255), thickness)

        if app_state == "TRAINING":
            cv2.putText(frame, feedback_text, (PANEL_X0 + 20, int(340 * text_size)), cv2.FONT_HERSHEY_SIMPLEX, 2.0 * text_size, (255, 255, 255), thickness + 1)


        display_frame = cv2.resize(frame, None, fx=900 / frame_h, fy=900 / frame_h,
                                   interpolation=cv2.INTER_AREA)
        cv2.imshow("Squats with AI", display_frame)

        # "Q" to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    pd.DataFrame(all_frames).to_csv("results.csv", index=False)
    print(f"Saved {len(all_frames)} frames")

    show_results(rep_results)
    
    cap.release()
    cv2.destroyAllWindows()
 
 
main()
 