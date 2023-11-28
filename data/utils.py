import cv2
def frame_details(video_path):
    
    # capturing the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('Error: Video not found')
        return -1
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frame
