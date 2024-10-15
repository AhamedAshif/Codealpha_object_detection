import cv2
import torch
from ultralytics import YOLO
from tkinter import *
from PIL import Image, ImageTk

# Load the pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  # You can use 'yolov8l.pt' for a larger model if needed

# Function to process the frame and update the Tkinter window
def update_frame():
    global cap, lbl_video

    # Capture frame-by-frame from the webcam
    ret, frame = cap.read()

    if ret:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame (draw bounding boxes and labels)
        annotated_frame = results[0].plot()

        # Convert the frame to RGB (OpenCV uses BGR by default)
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Convert the frame to ImageTk format for displaying in Tkinter
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        # Update the label with the new image
        lbl_video.imgtk = imgtk
        lbl_video.configure(image=imgtk)

    # Repeat the process after 10ms to continuously update the frame
    lbl_video.after(10, update_frame)

# Initialize the main Tkinter window
root = Tk()
root.title("YOLOv8 Object Detection Interface")
root.geometry("800x600")  # Set the window size

# Create a label widget to display the video feed
lbl_video = Label(root)
lbl_video.pack()

# Open the default camera (usually webcam)
cap = cv2.VideoCapture(0)

# Start the video capture and update the frame function
update_frame()

# Start the Tkinter event loop
root.mainloop()

# Release the camera and close any OpenCV windows when done
cap.release()
cv2.destroyAllWindows()
