import matplotlib.pyplot as plt
import pandas as ps
from matplotlib.widgets import Slider, RadioButtons
import numpy as np

POSE_CONNECTIONS = [
    (11, 12),                       # shoulders (left shoulder + right shoulder)
    (11, 13), (13, 15),             # left arm  (left shoulder + left elbow + left wrist)
    (12, 14), (14, 16),             # right arm (the same for right)
    (11, 23), (12, 24), (23, 24),   # torso     ((left shoulder + left hip) + (right shoulder + right hip) + (left hip + right hip))
    (23, 25), (25, 27),             # left leg  (left hip + left knee + left ankle)
    (24, 26), (26, 28),             # right leg (the same for right)
]

# << === CLAUDE === >>
template_positions = {
    11: (2, 8),     # left shoulder
    12: (-2, 8),      # right shoulder
    13: (3.5, 6.5), # left elbow
    14: (-3.5, 6.5),  # right elbow
    15: (4.5, 5),   # left wrist
    16: (-4.5, 5),    # right wrist
    23: (1.5, 4),   # left hip
    24: (-1.5, 4),    # right hip
    25: (1.7, 2),   # left knee
    26: (-1.7, 2),    # right knee
    27: (1.8, 0),   # left ankle
    28: (-1.8, 0),    # right ankle
}

# <<// === CLAUDE === >>

# 2D: draw skeleton on the screen IMPLEMENTED FROM MAIN.PY
def draw_skeleton_2d_visibility(template_positions, db, i):

    right_points = {12, 14, 16, 24, 26, 28}

    for point_1, point_2 in POSE_CONNECTIONS:

        if point_1 in right_points or point_2 in right_points:
            ax1.plot([template_positions[point_1][0],template_positions[point_2][0]], # only x from point 1 and 2
                     [template_positions[point_1][1],template_positions[point_2][1]], color = "black") # only y from point 1 and 2

        else:
            ax1.plot([template_positions[point_1][0],template_positions[point_2][0]], # only x from point 1 and 2
                                 [template_positions[point_1][1],template_positions[point_2][1]], color = "red") # only y from point 1 and 2            

    # << === CLAUDE === >>

    for idx, (x, y) in template_positions.items():
        v = db[f"2d_landmark_{idx}_visibility"][i]
        ax1.text(x + 0.1, y, f"({v:.2f})", fontsize=14)

    ax1.set_xlim(-6, 6)
    ax1.set_ylim(-1, 9)
    
    # plt.pause(0.001)
    # <<// === CLAUDE === >>
current_graph_type = "Kniewinkel"


def draw_skeleton_3d_visibility(template_positions, db, i):

    right_points = {12, 14, 16, 24, 26, 28}

    for point_1, point_2 in POSE_CONNECTIONS:

        if point_1 in right_points or point_2 in right_points:
            ax2.plot([template_positions[point_1][0],template_positions[point_2][0]], # only x from point 1 and 2
                     [template_positions[point_1][1],template_positions[point_2][1]], color = "black") # only y from point 1 and 2

        else:
            ax2.plot([template_positions[point_1][0],template_positions[point_2][0]], # only x from point 1 and 2
                                 [template_positions[point_1][1],template_positions[point_2][1]], color = "red") # only y from point 1 and 2            

    
    # << === CLAUDE === >>
    for idx, (x, y) in template_positions.items():
        v = db[f"3d_landmark_{idx}_visibility"][i]
        ax2.text(x + 0.1, y, f"({v:.2f})", fontsize=14)

    ax2.set_xlim(-6, 6)
    ax2.set_ylim(-1, 9)

    # plt.pause(0.001)
    # <<// === CLAUDE === >>
current_graph_type = "Kniewinkel"


def switch_graph(graph_type):

    ax2.set_visible(False)

    global current_graph_type, point, point2, slider
    
    current_graph_type = graph_type

   
    if graph_type == "Kniewinkel":
        ax1.cla()
        ax2.cla()
        ax2.set_visible(True)
        ax2.axis("on")
        ax1.set_position([0.18, 0.3, 0.23, 0.5])
        ax2.set_position([0.45, 0.3, 0.23, 0.5])
        ax3.set_position([0.70, 0.3, 0.23, 0.5])
        ax1.set_ylabel("Kniewinkel (°)")
        ax1.set_title("Rechter Kniewinkel (3D)")
        ax1.set_xlabel("Zeit (s)")
        ax1.plot(time_sec,db["knee_angle_right_side"])
        ax1.grid(True)
        point, = ax1.plot(time_sec[0],db["knee_angle_right_side"][0], "ro")

        ax2.set_title("Kniewinkel (2D)")
        ax2.set_xlabel("Zeit (s)")
        ax2.plot(time_sec, db["knee_angle_right_side_2d"])
        ax2.grid(True)
        point2, = ax2.plot(time_sec[0], db["knee_angle_right_side_2d"][0], "ro")

    elif graph_type == "Torsowinkel":
        ax1.cla()
        ax2.cla()
        ax2.set_visible(True)
        ax2.axis("on")
        ax1.set_position([0.18, 0.3, 0.23, 0.5])
        ax2.set_position([0.45, 0.3, 0.23, 0.5])
        ax3.set_position([0.70, 0.3, 0.23, 0.5])
        ax1.set_ylabel("Torsowinkel (3D) (°)")
        ax1.set_title("Rechter Torsowinkel")
        ax1.set_xlabel("Zeit (s)")
        ax1.plot(time_sec,db["torso_angle_right_side"])
        ax1.grid(True)
        point, = ax1.plot(time_sec[0],db["torso_angle_right_side"][0], "ro")

        ax2.set_title("Torsowinkel (2D) (°)")
        ax2.set_xlabel("Zeit (s)")
        ax2.plot(time_sec, db["torso_angle_right_side_2d"])
        ax2.grid(True)
        point2, = ax2.plot(time_sec[0], db["torso_angle_right_side_2d"][0], "ro")
        
    elif graph_type == "Segmentlängen":
        ax1.cla()
        ax2.cla()
        ax2.set_visible(True)
        ax2.axis("on")
        ax1.set_ylabel("Länge (m)")
        ax1.set_title("Segmentlängen")
        ax1.set_xlabel("Zeit (s)")

        ax1.set_position([0.18, 0.3, 0.23, 0.5])
        ax2.set_position([0.45, 0.3, 0.23, 0.5])
        ax3.set_position([0.70, 0.3, 0.23, 0.5])
        
        ax1.plot(time_sec, db["thigh_left_3d"], label="Oberschenkel L")
        ax1.plot(time_sec, db["thigh_right_3d"], label="Oberschenkel R")
        ax1.plot(time_sec, db["shin_left_3d"], label="Unterschenkel L")
        ax1.plot(time_sec, db["shin_right_3d"], label="Unterschenkel R")
        ax1.plot(time_sec, db["shoulder_width_3d"], label="Schulterbreite")
        ax1.plot(time_sec, db["hip_width_3d"], label="Hüftbreite")

        ax2.set_ylabel("Pixel (px)")
        ax2.set_title("Segmentlängen")
        ax2.set_xlabel("Zeit (s)")

        # ax2.plot(time_sec, db["thigh_left_2d"], label="Oberschenkel L")
        ax2.plot(time_sec, db["thigh_right_2d"], label="Oberschenkel R")
        ax2.legend()
        # ax2.set_xlim(20, 50)
        # ax2.set_ylim(190, 250)
        # ax2.plot(time_sec, db["shin_left_2d"], label="Unterschenkel L")
        # ax2.plot(time_sec, db["shin_right_2d"], label="Unterschenkel R")
        # ax2.plot(time_sec, db["shoulder_width_2d"], label="Schulterbreite")
        # ax2.plot(time_sec, db["hip_width_2d"], label="Hüftbreite")


        ax1.grid(True)
        
        point, = ax1.plot(time_sec[0],db["thigh_left_3d"][0], "ro")
        point2, = ax2.plot(time_sec[0],db["thigh_left_2d"][0], "ro")

    elif graph_type == "Alle_Daten":
        ax1.cla()
        ax1.set_title("Alle Daten")
        draw_skeleton_3d(ax1, ax2, ax3, db, int(slider.val))

    elif graph_type == "2D VS 3D Visibility":
        ax1.cla()
        ax2.cla()

        ax1.set_position([0.18, 0.3, 0.22, 0.6])
        ax2.set_position([0.45, 0.3, 0.22, 0.6])
        ax1.axis("off")
        ax2.axis("off")
        ax2.set_visible(True)
        ax1.set_title("2D points Visibility: Rot = linke Seite|Schwarz = rechte Seite")
        ax2.set_title("3D points Visibility")

        # << === CLAUDE === >>
        draw_skeleton_2d_visibility(template_positions, db, int(slider.val))
        # <<// === CLAUDE === >>

        draw_skeleton_3d_visibility(template_positions, db, int(slider.val))
    fig.canvas.draw_idle()

 # SLIDER TIME
# << === CLAUDE === SLIDER TIME >>
def update_slider_text(i):
    # i = номер рядка в CSV (позиція слайдера)
    frame_nr = db["frame"][i]         # справжній номер кадру з CSV
    zeit = time_sec[i]                # час у секундах
    slider.valtext.set_text(f"{frame_nr}  |  {zeit:.2f} s")
# <<// === CLAUDE === >>


def slider_upt(val):

    global current_graph_type, point2

    i = int(slider.val)
    update_slider_text(i)

    if current_graph_type == "Kniewinkel":
        i = int(slider.val)
        point.set_data([time_sec[i]],[db["knee_angle_right_side"][i]])
        point2.set_data([time_sec[i]], [db["knee_angle_right_side_2d"][i]])
        draw_skeleton_3d(ax1, ax2, ax3, db, i)

    elif current_graph_type == "Torsowinkel":
        i = int(slider.val)
        point.set_data([time_sec[i]],[db["torso_angle_right_side"][i]])
        point2.set_data([time_sec[i]], [db["torso_angle_right_side_2d"][i]])
        draw_skeleton_3d(ax1, ax2, ax3, db, i)

    elif current_graph_type == "Segmentlängen":
        i = int(slider.val)
        point.set_data([time_sec[i]], [db["thigh_left_3d"][i]])
        point2.set_data([time_sec[i]], [db["thigh_left_2d"][i]])
        draw_skeleton_3d(ax1, ax2, ax3, db, i)

    elif current_graph_type == "Alle_Daten":
        i = int(slider.val)
        draw_skeleton_3d(ax1, ax2, ax3, db, i)

        # << === CLAUDE === >>
    elif current_graph_type == "2D VS 3D Visibility":
        i = int(slider.val)
        ax1.cla()
        ax2.cla()

        ax1.axis("off")
        ax2.axis("off")

        ax1.set_title("2D points Visibility")
        ax2.set_title("3D points Visibility")

        draw_skeleton_2d_visibility(template_positions, db, i)
        draw_skeleton_3d_visibility(template_positions, db, i)  
        draw_skeleton_3d(ax1, ax2, ax3, db, int(slider.val))
        fig.canvas.draw_idle()
        # <<// === CLAUDE === >>

def draw_skeleton_3d(ax1, ax2, ax3, db, i):

    ax3.cla() #to clear previous frame

    right_points = {12, 14, 16, 24, 26, 28}


    for point_1, point_2 in POSE_CONNECTIONS:
        if point_1 in right_points:
            color = "black"
        else:
            color = "red"

        ax3.plot(
            [db[f"landmark_x{point_1}"][i], db[f"landmark_x{point_2}"][i]],
            [db[f"landmark_z{point_1}"][i], db[f"landmark_z{point_2}"][i]],
            [-db[f"landmark_y{point_1}"][i], -db[f"landmark_y{point_2}"][i]],
            color, lw=2,
        )
            
    ax3.set_title("3D-Skelett")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Z")
    ax3.set_zlabel("Y")
    ax3.set_xlim(-1, 1)
    ax3.set_ylim(-1, 1)
    ax3.set_zlim(-1, 1)
    ax3.set_box_aspect([1, 1, 1])
    # ax3.view_init(elev=-60, azim=270)
    ax3.view_init(elev=10, azim=-50)
    

    if current_graph_type == "Kniewinkel":
        ax3.text2D(0.05, 0.95, f"Kniewinkel 3D: {db['knee_angle_right_side'][i]:.1f}°", transform=ax3.transAxes)
        ax3.text2D(0.05, 0.90, f"Kniewinkel 2D: {db['knee_angle_right_side_2d'][i]:.1f}°", transform=ax3.transAxes)
        
    elif current_graph_type == "Torsowinkel":
        ax3.text2D(0.05, 0.95, f"Torsowinkel 3D: {db['torso_angle_right_side'][i]:.1f}°", transform=ax3.transAxes)
        ax3.text2D(0.05, 0.90, f"Torsowinkel 2D: {db['torso_angle_right_side_2d'][i]:.1f}°", transform=ax3.transAxes)

    elif current_graph_type == "Segmentlängen":
        

        ax3.text2D(0.05, 0.95, f"Oberschenkel R 3D [m]: {db['thigh_right_3d'][i]:.3f} m", transform=ax3.transAxes)
        ax3.text2D(0.05, 0.90, f"Unterschenkel R 3D [m]: {db['shin_right_3d'][i]:.3f} m", transform=ax3.transAxes)

        ax3.text2D(0.05, 0.85, f"Oberschenkel R 2D [px]: {db['thigh_right_2d'][i]:.3f} m", transform=ax3.transAxes)
        ax3.text2D(0.05, 0.80, f"Unterschenkel R 2D [px]: {db['shin_right_2d'][i]:.3f} m", transform=ax3.transAxes)
        

    elif current_graph_type == "Alle_Daten":

        ax1.cla()
        ax1.axis("off")
        ax2.cla()
        ax2.axis("off")
        ax2.set_visible(True)

        ax1.text(0.1, 0.95, "3D DATEN", transform=ax1.transAxes, fontsize=12, fontweight="bold")
        ax1.text(0.1, 0.88, f"Frame: {db['frame'][i]}  |  Zeit: {db['time_ms'][i]/1000:.2f} s", transform=ax1.transAxes)
        ax1.text(0.1, 0.81, f"Kniewinkel R: {db['knee_angle_right_side'][i]:.1f}°", transform=ax1.transAxes)
        ax1.text(0.1, 0.74, f"Kniewinkel L: {db['knee_angle_left_side'][i]:.1f}°", transform=ax1.transAxes)
        ax1.text(0.1, 0.67, f"Torsowinkel R: {db['torso_angle_right_side'][i]:.1f}°", transform=ax1.transAxes)
        ax1.text(0.1, 0.60, f"Torsowinkel L: {db['torso_angle_left_side'][i]:.1f}°", transform=ax1.transAxes)
        ax1.text(0.1, 0.53, f"Oberschenkel R: {db['thigh_right_3d'][i]:.3f} m", transform=ax1.transAxes)
        ax1.text(0.1, 0.46, f"Oberschenkel L: {db['thigh_left_3d'][i]:.3f} m", transform=ax1.transAxes)
        ax1.text(0.1, 0.39, f"Unterschenkel R: {db['shin_right_3d'][i]:.3f} m", transform=ax1.transAxes)
        ax1.text(0.1, 0.32, f"Unterschenkel L: {db['shin_left_3d'][i]:.3f} m", transform=ax1.transAxes)
        ax1.text(0.1, 0.25, f"Schulterbreite: {db['shoulder_width_3d'][i]:.3f} m", transform=ax1.transAxes)
        ax1.text(0.1, 0.18, f"Hüftbreite: {db['hip_width_3d'][i]:.3f} m", transform=ax1.transAxes)
        ax1.text(0.1, 0.11, f"Height L: {db['height_left_3d'][i]:.3f} m", transform=ax1.transAxes)
        ax1.text(0.1, 0.04, f"Height R: {db['height_right_3d'][i]:.3f} m", transform=ax1.transAxes)

        

        ax2.text(0.1, 0.95, "2D DATEN", transform=ax2.transAxes, fontsize=12, fontweight="bold")
        ax2.text(0.1, 0.88, f"Frame: {db['frame'][i]}  |  Zeit: {db['time_ms'][i]/1000:.2f} s", transform=ax2.transAxes)
        ax2.text(0.1, 0.81, f"Kniewinkel R: {db['knee_angle_right_side_2d'][i]:.1f}°", transform=ax2.transAxes)
        ax2.text(0.1, 0.74, f"Kniewinkel L: {db['knee_angle_left_side_2d'][i]:.1f}°", transform=ax2.transAxes)
        ax2.text(0.1, 0.67, f"Torsowinkel R: {db['torso_angle_right_side_2d'][i]:.1f}°", transform=ax2.transAxes)
        ax2.text(0.1, 0.60, f"Torsowinkel L: {db['torso_angle_left_side_2d'][i]:.1f}°", transform=ax2.transAxes)
        ax2.text(0.1, 0.53, f"Oberschenkel R: {db['thigh_right_2d'][i]:.1f} px", transform=ax2.transAxes)
        ax2.text(0.1, 0.46, f"Oberschenkel L: {db['thigh_left_2d'][i]:.1f} px", transform=ax2.transAxes)
        ax2.text(0.1, 0.39, f"Unterschenkel R: {db['shin_right_2d'][i]:.1f} px", transform=ax2.transAxes)
        ax2.text(0.1, 0.32, f"Unterschenkel L: {db['shin_left_2d'][i]:.1f} px", transform=ax2.transAxes)
        ax2.text(0.1, 0.25, f"Schulterbreite: {db['shoulder_width_2d'][i]:.1f} px", transform=ax2.transAxes)
        ax2.text(0.1, 0.18, f"Hüftbreite: {db['hip_width_2d'][i]:.1f} px", transform=ax2.transAxes)
        ax2.text(0.1, 0.11, f"Height L: {db['height_left_2d'][i]:.1f} px", transform=ax2.transAxes)
        ax2.text(0.1, 0.04, f"Height R: {db['height_right_2d'][i]:.1f} px", transform=ax2.transAxes)

      
        
    plt.pause(0.001)  


db = ps.read_csv("daten/mediapipe_csv/kamerawinkel/vid_90_degree_test.csv")





time_sec = db["time_ms"] / 1000 

fig = plt.figure(figsize=(19, 9))# ax1 це один графік зверху, а ах2 це графік знизу, fig це полотно на якому вони створюються

ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3, projection = "3d")

ax1.set_xlabel("Zeit (s)")
ax1.set_ylabel("Kniewinkel (°)")
ax1.set_title("Rechter Kniewinkel")
ax1.grid(True)

plt.subplots_adjust(bottom=0.25)

# ax.plot(x,y)
plt.ion()
ax1.plot(time_sec,db["knee_angle_right_side"])
ax_radio = plt.axes([0.02, 0.4, 0.12, 0.2])
radio = RadioButtons(ax_radio, ["Kniewinkel", "Torsowinkel", "Segmentlängen", "Alle_Daten", "2D VS 3D Visibility"])
point, = ax1.plot(time_sec[0],db["knee_angle_right_side"][0], "ro")
ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
slider = Slider(ax_slider, "Frame", 0, len(db)-1, valinit=0, valstep=1)


slider.on_changed(slider_upt) #when slider is moving - call a function slider_upt
radio.on_clicked(switch_graph)

switch_graph("Kniewinkel")

plt.ioff()
plt.show()


