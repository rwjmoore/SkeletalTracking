#integration between realsense and mediapipe 
import pyrealsense2 as rs
import mediapipe as mp
import cv2
import numpy as np
import datetime as dt

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
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5,model_complexity = 1, smooth_landmarks = True )
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
    background_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), background_removed_color, color_image)

    #format images for Mediapipe
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    images = cv2.flip(background_removed,1)
    color_image = cv2.flip(color_image,1)
    color_images_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)


    #we now have the depth and RGB images correctly formtted and aligned. Now processing with Mediapipe
    results = hands.process(color_images_rgb)
    if results.multi_hand_landmarks:
        number_of_hands = len(results.multi_hand_landmarks)
        i=0
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(images, handLms, mpHands.HAND_CONNECTIONS)
            org2 = (20, org[1]+(20*(i+1)))
            hand_side_classification_list = results.multi_handedness[i]
            hand_side = hand_side_classification_list.classification[0].label
            middle_finger_knuckle = results.multi_hand_landmarks[i].landmark[9]
            x = int(middle_finger_knuckle.x*len(depth_image_flipped[0]))
            y = int(middle_finger_knuckle.y*len(depth_image_flipped))
            if x >= len(depth_image_flipped[0]):
                x = len(depth_image_flipped[0]) - 1
            if y >= len(depth_image_flipped):
                y = len(depth_image_flipped) - 1
            mfk_distance = depth_image_flipped[y,x] * depth_scale # meters
            mfk_distance_feet = mfk_distance * 3.281 # feet

            



            images = cv2.putText(images, f"{hand_side} Hand Distance: {mfk_distance_feet:0.3} feet ({mfk_distance:0.3} m) away", org2, font, fontScale, color, thickness, cv2.LINE_AA)
            i+=1
            images = cv2.putText(images, f"Hands: {number_of_hands}", org, font, fontScale, color, thickness, cv2.LINE_AA)
    else:
        images = cv2.putText(images,"No Hands", org, font, fontScale, color, thickness, cv2.LINE_AA)
    

  


    # Display FPS
    time_diff = dt.datetime.today().timestamp() - start_time
    fps = int(1 / time_diff)
    org3 = (20, org[1] + 60)
    images = cv2.putText(images, f"FPS: {fps}", org3, font, fontScale, color, thickness, cv2.LINE_AA)
    name_of_window = 'SN: ' + str(device)

    # Display images
    cv2.namedWindow(name_of_window, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(name_of_window, images)
    key = cv2.waitKey(1)
    # Press esc or 'q' to close the image window
    if key & 0xFF == ord('q') or key == 27:
        print(f"User pressed break key for SN: {device}")
        break

print(f"Exiting loop for SN: {device}")
print(f"Application Closing.")
pipeline.stop()
print(f"Application Closed.")