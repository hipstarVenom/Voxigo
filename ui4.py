import cv2
import pyttsx3
import threading
import time
import numpy as np
from queue import Queue
from ultralytics import YOLO
import tkinter as tk
from tkinter import messagebox
from tkinter import font

# -----------------------------
# Global Variables and Initialization
# -----------------------------

# For dynamic instructions from detection
latest_instruction = "Waiting for detection..."
previous_label = None  # Track the previously detected object's label

# Initialize Text-to-Speech Engine
engine = pyttsx3.init()
engine.setProperty('rate', 180)  # Adjust speech speed
last_speech_time = {}  # Track last speech time for each object
speech_queue = Queue()

def speech_worker():
    """Process speech requests sequentially to avoid overlapping."""
    while True:
        text, label = speech_queue.get()
        engine.say(text)
        engine.runAndWait()
        speech_queue.task_done()

threading.Thread(target=speech_worker, daemon=True).start()

def speak(text, label):
    """Add speech request to the queue with a cooldown."""
    current_time = time.time()
    if label not in last_speech_time or (current_time - last_speech_time[label]) > 3:
        last_speech_time[label] = current_time
        speech_queue.put((text, label))

def flush_speech_queue():
    """Flush the speech queue when needed."""
    while not speech_queue.empty():
        speech_queue.get()
        speech_queue.task_done()

# Load YOLO Model
model = YOLO("yolov8n.pt")

# Object Height Mapping for Distance Estimation
OBJECT_HEIGHTS = {
    "person": 1.7, "bicycle": 1.0, "car": 1.5, "motorcycle": 1.2, "bus": 3.0,
    "train": 4.0, "truck": 3.5, "boat": 2.5, "traffic light": 2.5, "fire hydrant": 0.75,
    "stop sign": 1.8, "parking meter": 1.2, "bench": 0.5, "bird": 0.3, "cat": 0.3,
    "dog": 0.5, "horse": 1.6, "cow": 1.4, "elephant": 3.0, "bear": 1.5,
    "chair": 0.8, "couch": 1.0, "potted plant": 0.6, "bed": 0.7, "dining table": 0.75,
    "toilet": 0.5, "tv": 0.7, "laptop": 0.3, "cell phone": 0.15, "microwave": 0.4,
    "oven": 0.9, "sink": 0.9, "refrigerator": 1.8, "book": 0.25, "clock": 0.4
}
DEFAULT_HEIGHT = 1.7
FOCAL_LENGTH = 700  # Estimated focal length

# Global variables for frame processing and control
frame = None
frame_lock = threading.Lock()
detection_active = False  # Flag to control detection loop
detection_thread = None   # Will hold our detection thread
ip_camera_url = ""        # Will be updated from the GUI input

# Create a background subtractor for improved motion detection
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=16, detectShadows=True)

# -----------------------------
# Detection and Frame Update Functions
# -----------------------------

def update_frame(cap):
    """Continuously capture frames in a separate thread to reduce lag."""
    global frame
    while detection_active:
        ret, latest_frame = cap.read()
        if not ret:
            # Restart the stream if connection fails
            cap.release()
            cap = cv2.VideoCapture(ip_camera_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue
        # Resize frame to lower resolution for speed
        resized_frame = cv2.resize(latest_frame, (640, 360))
        with frame_lock:
            frame = resized_frame
    cap.release()

def run_detection():
    global frame, latest_instruction, previous_label
    cap = cv2.VideoCapture(ip_camera_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Process only the latest frame

    # Start the frame update thread
    threading.Thread(target=update_frame, args=(cap,), daemon=True).start()

    frame_counter = 0
    previous_frame = None

    # Detection loop
    while detection_active:
        if frame is None:
            continue  # Wait until a frame is available

        frame_counter += 1
        # Skip every 2nd frame to improve speed
        if frame_counter % 2 != 0:
            continue

        with frame_lock:
            current_frame = frame.copy()

        # Apply background subtraction to detect motion more robustly
        fg_mask = bg_subtractor.apply(current_frame)
        motion_level = np.sum(fg_mask) / 255

        # If motion is low, overlay message and flush speech queue
        if motion_level < 500:  # Threshold value; adjust as needed
            latest_instruction = "⚠️ No significant motion detected."
            cv2.putText(current_frame, latest_instruction, (10, 350),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            flush_speech_queue()
            cv2.imshow("Live Object Detection", current_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Reduce memory usage by converting scale
        current_frame = cv2.convertScaleAbs(current_frame, alpha=0.5, beta=0)

        # Run YOLOv8 Object Detection
        results = model(current_frame, conf=0.35, verbose=False)

        nearest_object = None
        min_distance = float("inf")
        left_objects, right_objects, center_objects = 0, 0, 0
        detected_objects = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                label = result.names[cls_id]
                confidence = box.conf[0].item()
                detected_objects.append(label)

                # Estimate Distance
                pixel_height = y2 - y1
                real_height = OBJECT_HEIGHTS.get(label, DEFAULT_HEIGHT)
                if pixel_height > 0:
                    distance_m = (real_height * FOCAL_LENGTH) / pixel_height

                    # Find nearest object
                    if distance_m < min_distance:
                        min_distance = distance_m
                        nearest_object = (x1, y1, x2, y2, label, confidence, distance_m)

                    # Determine Object Position for Navigation Instructions
                    center_x = (x1 + x2) // 2
                    if center_x < 213:
                        left_objects += 1
                    elif center_x > 427:
                        right_objects += 1
                    else:
                        center_objects += 1

        # Update dynamic instruction based on detection
        if nearest_object:
            x1, y1, x2, y2, label, confidence, distance_m = nearest_object
            # If a new object is detected, flush the speech queue
            if previous_label is None or label != previous_label:
                flush_speech_queue()
            previous_label = label

            # Set instruction based on object's distance and position with arrow emojis
            if distance_m < 2.5:  # Object is close
                if center_objects > 0:
                    latest_instruction = f"⚠️ {label} at {distance_m:.2f} m ahead! move right or left "
                elif left_objects > right_objects:
                    latest_instruction = f"⚠️ {label} at {distance_m:.2f} m - move right "
                elif right_objects > left_objects:
                    latest_instruction = f"⚠️ {label} at {distance_m:.2f} m - move left "
            else:
                latest_instruction = f"{label} detected at {distance_m:.2f} m."
            speak(latest_instruction, label)
            # Draw bounding box and overlay detection information
            cv2.rectangle(current_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(current_frame, f"{label} ({confidence:.2f})", (x1, y1 - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(current_frame, f"Distance: {distance_m:.2f}m", (x1, y1 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        else:
            latest_instruction = "No objects detected."

        # Overlay additional information on the frame
        cv2.putText(current_frame, "Detection Active", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(current_frame, f"Objects Detected: {len(detected_objects)}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Show Live Feed
        cv2.imshow("Live Object Detection", current_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# -----------------------------
# GUI Functions
# -----------------------------

def start_detection():
    global detection_active, detection_thread, ip_camera_url
    ip = ip_entry.get().strip()
    if not ip:
        messagebox.showerror("Input Error", "Please enter the IP address of the camera.")
        return
    ip_camera_url = f"http://{ip}:8080/video?nocache=1"
    if detection_active:
        messagebox.showinfo("Detection", "Detection is already running.")
        return
    detection_active = True
    detection_thread = threading.Thread(target=run_detection, daemon=True)
    detection_thread.start()
    messagebox.showwarning("Detection Started", "Detection has been started. Stay alert for warnings!")
    start_btn.config(state=tk.DISABLED)
    stop_btn.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

def stop_detection():
    global detection_active
    if not detection_active:
        messagebox.showinfo("Detection", "Detection is not running.")
        return
    detection_active = False
    messagebox.showinfo("Detection Stopped", "Detection has been stopped.")
    stop_btn.grid_forget()

def exit_program():
    global detection_active
    if detection_active:
        detection_active = False
    root.destroy()

def update_gui_instructions():
    """Periodically update the instruction label with dynamic instructions from detection."""
    instruction_label.config(text="Instructions:\n" + latest_instruction)
    root.after(500, update_gui_instructions)

# -----------------------------
# GUI Setup using Tkinter
# -----------------------------

root = tk.Tk()
root.title("Object Detection GUI")

# Set up a custom font for the instructions
instruction_font = font.Font(family="Helvetica", size=12, weight="bold")

# IP Address Input
ip_label = tk.Label(root, text="IP Address:")
ip_label.grid(row=0, column=0, padx=10, pady=10)
ip_entry = tk.Entry(root, width=25)
ip_entry.grid(row=0, column=1, padx=10, pady=10)
ip_entry.insert(0, "192.168.43.9")

# Start Detection Button (visible initially)
start_btn = tk.Button(root, text="Start Detection", width=20, command=start_detection)
start_btn.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

# Stop Detection Button (will be created after Start Detection is clicked)
stop_btn = tk.Button(root, text="Stop Detection", width=20, command=stop_detection)

# Exit Button (always visible)
exit_btn = tk.Button(root, text="Exit", width=20, command=exit_program)
exit_btn.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

# Instruction Label for dynamic interactivity
instruction_label = tk.Label(root, text="Instructions:\nWaiting for detection...", 
                             justify="left", fg="blue", font=instruction_font)
instruction_label.grid(row=4, column=0, columnspan=2, padx=10, pady=10)

root.after(500, update_gui_instructions)
root.mainloop()
