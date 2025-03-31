Voxigo - Object Detection for the Blind
Voxigo is an assistive technology project aimed at helping visually impaired individuals by providing real-time object detection and auditory feedback. The application uses YOLO (You Only Look Once) model for object detection and integrates text-to-speech feedback to describe the objects detected around the user.

Features
Real-Time Object Detection: Uses YOLOv8 model to detect various objects in the environment.

Distance Estimation: Estimates the distance of detected objects based on their size and focal length.

Speech Feedback: Provides auditory feedback on detected objects, their distance, and navigation instructions (e.g., "Move left" or "Move right").

Motion Detection: Uses background subtraction to detect motion and provide feedback when no significant motion is detected.

Interactive GUI: Built with Tkinter for easy interaction. Allows input for the IP address of an IP camera for video feed.

Installation Requirements
Python 3.x

Required Python libraries:
opencv-python
pyttsx3
numpy
ultralytics (YOLOv8)
tkinter

Install dependencies using pip:
pip install opencv-python pyttsx3 numpy ultralytics tkinter

Setup Instructions
Download YOLOv8 Model: Download the YOLOv8 model (yolov8n.pt) and place it in the same directory as the script.

Run the Application:

Launch the script (main.py) in your Python environment.

In the Tkinter GUI, input the IP address of your camera for the video feed (default is 192.168.43.9).

Click "Start Detection" to begin the object detection process.

The system will detect objects in real-time and provide speech feedback on detected objects and distances.

Usage
Start Detection:

After entering the IP camera URL in the input field, click the "Start Detection" button.

The system will begin detecting objects and providing feedback via speech.

Stop Detection:

Click the "Stop Detection" button to halt the detection process.

Exit:

The "Exit" button allows you to safely exit the application.

Object Distance Calculation
The application calculates the distance of detected objects based on their real-world height and the camera's focal length. The default height of objects is defined in a dictionary (OBJECT_HEIGHTS), but it can be adjusted as needed.

Known Limitations
Accuracy of distance estimation may vary depending on the camera’s focal length and positioning.

Low-resolution video feeds or unstable camera connections may impact detection speed.

License
This project is licensed under the MIT License.
