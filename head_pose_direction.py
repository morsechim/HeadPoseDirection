import cv2
import numpy as np
import mediapipe as mp
import time

# Initialize the face mesh model
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize the drawing utility
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Open the camera
camera_device = 0
cap = cv2.VideoCapture(camera_device)

# Define window name
win_name = "frame"
cv2.namedWindow(winname=win_name)

prev_frame_time = 0
new_frame_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    start = time.time()
    
    # Convert to RGB colorspace
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Flip camera for selfie-view display
    frame = cv2.flip(frame, 1)
    
    # To improve performance
    frame.flags.writeable = False
    
    # Get the result
    results = face_mesh.process(frame)
    
    # To improve performance
    frame.flags.writeable = True
    
    # Convert back to BGR colorspace for OpenCV display
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    frame_h, frame_w, frame_c = frame.shape
    
    face_3d = []
    face_2d = []
    direction_text = ""
    x = y = z = 0
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, landmark in enumerate(face_landmarks.landmark):
                if idx in [33, 263, 1, 61, 291, 199]:
                    # nose tip
                    if idx == 1:
                        nose_2d = (landmark.x * frame_w, landmark.y * frame_h)
                        nose_3d = (landmark.x * frame_w, landmark.y * frame_h, landmark.z * 3000)
                        
                    x_coord, y_coord = int(landmark.x * frame_w), int(landmark.y * frame_h)
                    
                    # get the 2D coordinates
                    face_2d.append([x_coord, y_coord])
                    # get the 3D coordinates
                    face_3d.append([x_coord, y_coord, landmark.z])
                    
            # Convert to numpy array 
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)
            
            # Ensure there are at least 4 points
            if len(face_2d) >= 4:
                # The camera matrix
                focal_length = 1 * frame_w
                
                camera_matrix = np.array([
                    [focal_length, 0, frame_w / 2],
                    [0, focal_length, frame_h / 2],
                    [0, 0, 1]
                ])
                
                # The distance matrix
                dist_matrix = np.zeros((4, 1), dtype=np.float64)
                
                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, camera_matrix, dist_matrix)
                
                if success:
                    # Get rotational matrix
                    rot_matrix, jac = cv2.Rodrigues(rot_vec)
                    
                    # Get angles 
                    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rot_matrix)
                    
                    # Get the y rotation degree
                    y = angles[0] * 360
                    x = angles[1] * 360
                    z = angles[2] * 360
                    
                    # Determine where the user's head is tilting
                    if x < -5:
                        direction_text = "Left"
                    elif x > 5:
                        direction_text = "Right"
                    elif y < -5: 
                        direction_text = "Down"
                    elif y > 5: 
                        direction_text = "Up"
                    else:
                        direction_text = "Forward"
                        
                    # Display the nose direction
                    nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, camera_matrix, dist_matrix)
                    
                    p1 = (int(nose_2d[0]), int(nose_2d[1]))
                    p2 = (int(nose_2d[0] + x * 10), int(nose_2d[1] - y * 10))
                    
                    cv2.line(frame, p1, p2, (255, 0, 255), 3)
            
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )
    
    # Calculate FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    
    # Convert FPS to string
    fps_text = f'FPS: {int(fps)}'
    
    # Put FPS text on frame
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Put tilt, x, y, z text on frame
    cv2.putText(frame, f'Direction: {direction_text}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'X: {x:.2f}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Y: {y:.2f}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Z: {z:.2f}', (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv2.LINE_AA)
    
    cv2.imshow(win_name, frame)
    
    if cv2.waitKey(1) == ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()