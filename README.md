# Real-time Head Pose Estimation with Face Mesh

This Python script demonstrates real-time head pose estimation using the MediaPipe Face Mesh and OpenCV. It calculates the rotational angles of the head (yaw, pitch, roll) and determines the direction the user's head is tilting.

## Requirements
- Python 3.x
- OpenCV (`pip install opencv-python`)
- Mediapipe (`pip install mediapipe`)

## Usage
1. Clone the repository.
2. Install the required dependencies.
3. Run the script `head_pose_estimation.py`.
4. The script will open a window showing the webcam feed with real-time head pose estimation.

## Features
- Uses the MediaPipe Face Mesh to detect and track facial landmarks.
- Estimates head pose (yaw, pitch, roll) using solvePnP and Rodrigues transformation.
- Displays direction of head tilt (Forward, Left, Right, Up, Down) based on pose angles.
- Draws landmarks and connects them in a mesh for visualization.
- Calculates and displays FPS (frames per second) in real-time.

## Description
The script initializes the MediaPipe Face Mesh model and captures frames from the webcam. It processes each frame to detect facial landmarks and estimate the head pose in terms of rotation angles. These angles are then used to determine the direction of head tilt and are displayed on the frame along with FPS information. The nose tip direction is visualized by drawing a line from the nose tip based on the estimated pose.

## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

## Notes
- Ensure proper lighting conditions for accurate face landmark detection.
- The script assumes a single camera setup (webcam).
- Adjust `min_detection_confidence` and `min_tracking_confidence` for better performance as needed.
- Press 'q' to quit the application.

Feel free to modify and integrate this script into your projects for real-time head pose estimation applications.