import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation

# Load your video file
video_path = 'downloaded_video.mp4'
cap = cv2.VideoCapture(video_path)

# Load your anomaly scores for each frame (replace sra_list with your actual list)
sra_list = np.load('Arrest_001_sr_.npy')

# Initialize Matplotlib figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Video progress plot
ax1.set_title('Video Frame Progress')
ax1.set_xlabel('Frame Number')
ax1.set_ylabel('Gray Level')
video_plot, = ax1.plot([], [], color='blue')

# Anomaly score progress plot
ax2.set_title('Anomaly Score Progress')
ax2.set_xlabel('Frame Number')
ax2.set_ylabel('Anomaly Score')
sra_plot, = ax2.plot([], [], color='red')

def update_plot(frame_number):
    ret, frame = cap.read()
    if not ret:
        return

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Update video progress plot
    video_plot.set_data(range(frame_number), gray_frame[:frame_number])

    # Update anomaly score progress plot
    sra_plot.set_data(range(frame_number), sra_list[:frame_number])

    return video_plot, sra_plot


# Set the total number of frames
total_frames = 240

# Create animation
ani = FuncAnimation(fig, update_plot, frames=total_frames, interval=100, blit=True)

# Display the animation
plt.show()

# Release the video capture object
cap.release()
