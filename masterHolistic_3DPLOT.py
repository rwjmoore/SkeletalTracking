#this version of the code enables the saving of the mediaPipe coordinates to an excel file. This does not include the overlay and depth data of the depth camera. 

#integration between realsense and mediapipe 
import pyrealsense2 as rs
import mediapipe as mp
import cv2
import numpy as np
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt




def plot_world_landmarks(
    plt,
    ax,
    landmarks,
    visibility_th=0.5,
):
    landmark_point = []

    for index, landmark in enumerate(landmarks.landmark):
        landmark_point.append(
            [landmark.visibility, (landmark.x, landmark.y, landmark.z)])

    face_index_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    right_arm_index_list = [11, 13, 15, 17, 19, 21]
    left_arm_index_list = [12, 14, 16, 18, 20, 22]
    right_body_side_index_list = [11, 23, 25, 27, 29, 31]
    left_body_side_index_list = [12, 24, 26, 28, 30, 32]
    shoulder_index_list = [11, 12]
    waist_index_list = [23, 24]

    # 顔
    face_x, face_y, face_z = [], [], []
    for index in face_index_list:
        point = landmark_point[index][1]
        face_x.append(point[0])
        face_y.append(point[2])
        face_z.append(point[1] * (-1))

    # 右腕
    right_arm_x, right_arm_y, right_arm_z = [], [], []
    for index in right_arm_index_list:
        point = landmark_point[index][1]
        right_arm_x.append(point[0])
        right_arm_y.append(point[2])
        right_arm_z.append(point[1] * (-1))

    # 左腕
    left_arm_x, left_arm_y, left_arm_z = [], [], []
    for index in left_arm_index_list:
        point = landmark_point[index][1]
        left_arm_x.append(point[0])
        left_arm_y.append(point[2])
        left_arm_z.append(point[1] * (-1))

    # 右半身
    right_body_side_x, right_body_side_y, right_body_side_z = [], [], []
    for index in right_body_side_index_list:
        point = landmark_point[index][1]
        right_body_side_x.append(point[0])
        right_body_side_y.append(point[2])
        right_body_side_z.append(point[1] * (-1))

    # 左半身
    left_body_side_x, left_body_side_y, left_body_side_z = [], [], []
    for index in left_body_side_index_list:
        point = landmark_point[index][1]
        left_body_side_x.append(point[0])
        left_body_side_y.append(point[2])
        left_body_side_z.append(point[1] * (-1))

    # 肩
    shoulder_x, shoulder_y, shoulder_z = [], [], []
    for index in shoulder_index_list:
        point = landmark_point[index][1]
        shoulder_x.append(point[0])
        shoulder_y.append(point[2])
        shoulder_z.append(point[1] * (-1))

    # 腰
    waist_x, waist_y, waist_z = [], [], []
    for index in waist_index_list:
        point = landmark_point[index][1]
        waist_x.append(point[0])
        waist_y.append(point[2])
        waist_z.append(point[1] * (-1))
            
    ax.cla()
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)

    ax.scatter(face_x, face_y, face_z)
    ax.plot(right_arm_x, right_arm_y, right_arm_z)
    ax.plot(left_arm_x, left_arm_y, left_arm_z)
    ax.plot(right_body_side_x, right_body_side_y, right_body_side_z)
    ax.plot(left_body_side_x, left_body_side_y, left_body_side_z)
    ax.plot(shoulder_x, shoulder_y, shoulder_z)
    ax.plot(waist_x, waist_y, waist_z)
    
    plt.pause(.001)

    return







#main variables
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
fig.subplots_adjust(left=0.0, right=1, bottom=0, top=1)
count = 0
alldata = []

pose_tubuh = ['NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER', 'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT',
              'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY', 'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB',
              'RIGHT_THUMB', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']

pose_tangan = ['WRIST', 'THUMB_CPC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP', 'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP', 'MIDDLE_FINGER_MCP',
               'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
               'RING_FINGER_MCP', 'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP']
 
 
pose_tangan_2 = ['WRIST2', 'THUMB_CPC2', 'THUMB_MCP2', 'THUMB_IP2', 'THUMB_TIP2', 'INDEX_FINGER_MCP2', 'INDEX_FINGER_PIP2', 'INDEX_FINGER_DIP2', 'INDEX_FINGER_TIP2', 'MIDDLE_FINGER_MCP2',
               'MIDDLE_FINGER_PIP2', 'MIDDLE_FINGER_DIP2', 'MIDDLE_FINGER_TIP2', 'RING_FINGER_PIP2', 'RING_FINGER_DIP2', 'RING_FINGER_TIP2',
               'RING_FINGER_MCP2', 'PINKY_MCP2', 'PINKY_PIP2', 'PINKY_DIP2', 'PINKY_TIP2']

#openCV window formatting
font = cv2.FONT_HERSHEY_SIMPLEX
org = (20, 100)
fontScale = .5
color = (0,150,255)
thickness = 1 

# ====== Configure Realsense Environment ======
realsense_ctx = rs.context()
connected_devices = [] # List of serial numbers for present cameras
for i in range(len(realsense_ctx.devices)):
  detected_camera = realsense_ctx.devices[i].get_info(rs.camera_info.serial_number)
  print(f"{detected_camera}")
  connected_devices.append(detected_camera)
device = connected_devices[0] # In this example we are only using one camera
pipeline = rs.pipeline()
config = rs.config()
background_removed_color = 153 # Grey

# ====== Define Variables for Mediapipe ======

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity = 2)


# ====== Enable Streams ======
config.enable_device(device)
# # For worse FPS, but better resolution:
# stream_res_x = 1280
# stream_res_y = 720
# # For better FPS. but worse resolution:
# NOTE: Resolution of LIDAR L515 depth camera is 1024 x 768 
stream_res_x = 1024
stream_res_y = 768
stream_fps = 30
# config.enable_stream(rs.stream.depth, stream_index = -1)
# config.enable_stream(rs.stream.color, stream_index = -1)

config.enable_stream(rs.stream.depth, -1, 0, 0, rs.format.z16, stream_fps)
config.enable_stream(rs.stream.color, -1, 0, 0, rs.format.bgr8, stream_fps)

#config.enable_all_streams()
profile = pipeline.start(config)
align_to = rs.stream.color
align = rs.align(align_to)

# ====== Get depth Scale ======
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print(f"\tDepth Scale for Camera SN {device} is: {depth_scale}")
# ====== Set clipping distance ======
clipping_distance_in_meters = 2
clipping_distance = clipping_distance_in_meters / depth_scale
print(f"\tConfiguration Successful for SN {device}")


while True:
    start_time = dt.datetime.today().timestamp() # Necessary for FPS calculations
    # Get and align frames
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    if not aligned_depth_frame or not color_frame:
        print('not aligned')
        continue
    

    # Process images
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    depth_image_flipped = cv2.flip(depth_image,1)
    color_image = np.asanyarray(color_frame.get_data())
    #since depth image is single channel and colour image is three we stack the depth image on itself three times
    depth_image_3d = np.dstack((depth_image,depth_image,depth_image))
    #background_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), background_removed_color, color_image)
    background_removed = color_image
    #format images for Mediapipe
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    images = cv2.flip(background_removed,1)
    color_image = cv2.flip(color_image,1)
    color_images_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)


    #we now have the depth and RGB images correctly formtted and aligned. Now processing with Mediapipe


            
        
    results = holistic.process(cv2.cvtColor(color_images_rgb, cv2.COLOR_BGR2RGB))
    if True:
        #draw landmarks
        mp_drawing.draw_landmarks(images, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(images, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(images, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        plot_world_landmarks(plt,ax, results.pose_world_landmarks)
        
        #  if(results.pose_landmarks is not None and results.left_hand_landmarks is not None and results.right_hand_landmarks is not None):

        if results.pose_landmarks: 
            data_tubuh = {}
            for i in range(len(pose_tubuh)):
                results.pose_landmarks.landmark[i].x = results.pose_landmarks.landmark[i].x * images.shape[0]
                results.pose_landmarks.landmark[i].y = results.pose_landmarks.landmark[i].y * images.shape[1]
                data_tubuh.update(
                {pose_tubuh[i] : results.pose_landmarks.landmark[i]}
                )
            alldata.append(data_tubuh)

        if results.right_hand_landmarks:
            data_tangan_kanan = {}
            for i in range(len(pose_tangan)):
                results.right_hand_landmarks.landmark[i].x = results.right_hand_landmarks.landmark[i].x * images.shape[0]
                results.right_hand_landmarks.landmark[i].y = results.right_hand_landmarks.landmark[i].y * images.shape[1]
                data_tubuh.update(
                {pose_tangan[i] : results.right_hand_landmarks.landmark[i]}
                )
            alldata.append(data_tubuh)

        if results.left_hand_landmarks:
            data_tangan_kiri  = {}
            for i in range(len(pose_tangan)):
                results.left_hand_landmarks.landmark[i].x = results.left_hand_landmarks.landmark[i].x * images.shape[0]
                results.left_hand_landmarks.landmark[i].y = results.left_hand_landmarks.landmark[i].y * images.shape[1]
                data_tubuh.update(
                {pose_tangan_2[i] : results.left_hand_landmarks.landmark[i]}
                )
            alldata.append(data_tubuh)




    #     i=0
        
    #     org2 = (20, org[1]+(20*(i+1)))
    #     hand_side_classification_list = results.multi_handedness[i]
    #     hand_side = hand_side_classification_list.classification[0].label
    #     middle_finger_knuckle = results.multi_hand_landmarks[i].landmark[9]
    #     x = int(middle_finger_knuckle.x*len(depth_image_flipped[0]))
    #     y = int(middle_finger_knuckle.y*len(depth_image_flipped))
    #     if x >= len(depth_image_flipped[0]):
    #         x = len(depth_image_flipped[0]) - 1
    #     if y >= len(depth_image_flipped):
    #         y = len(depth_image_flipped) - 1
    #     mfk_distance = depth_image_flipped[y,x] * depth_scale # meters
    #     mfk_distance_feet = mfk_distance * 3.281 # feet

        

    #     images = cv2.putText(images, f"{hand_side} Hand Distance: {mfk_distance_feet:0.3} feet ({mfk_distance:0.3} m) away", org2, font, fontScale, color, thickness, cv2.LINE_AA)
    #     i+=1
    #     images = cv2.putText(images, f"Hands: {number_of_hands}", org, font, fontScale, color, thickness, cv2.LINE_AA)
    # else:
    #     images = cv2.putText(images,"No Hands", org, font, fontScale, color, thickness, cv2.LINE_AA)
    




    # Display FPS
    time_diff = dt.datetime.today().timestamp() - start_time
    fps = int(1 / time_diff)
    org3 = (20, org[1] + 60)
    images = cv2.putText(images, f"FPS: {fps}", org3, font, fontScale, color, thickness, cv2.LINE_AA)
    name_of_window = 'SN: ' + str(device)

    # Display images
    cv2.namedWindow(name_of_window, cv2.WINDOW_AUTOSIZE)
    #cv2.imshow(name_of_window, images)
    key = cv2.waitKey(1)
    # Press esc or 'q' to close the image window
    if key & 0xFF == ord('q') or key == 27:
        print(f"User pressed break key for SN: {device}")
        df = pd.DataFrame(alldata)
        df.to_excel("SkeletalCoordinates.xlsx")
        break

print(f"Exiting loop for SN: {device}")
print(f"Application Closing.")
pipeline.stop()
print(f"Application Closed.")





