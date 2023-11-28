import torch
import torch.nn as nn 
import cv2
import numpy as np 

class VideoDataGenerator(torch.utils.data.Dataset):
    
    def __init__(self,path_df,batch_size,sequence_length):
        
        self.path_df=path_df
        self.batch_size=batch_size
        self.sequence_length=sequence_length
        
    def frame_extraction(self,video_path):
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print('Error: Video Path check failed')
            return -1

        frame_list = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize the frame
            img = cv2.resize(frame, (64, 64))
            # Normalize the frame
            img_norm = np.array(img, dtype=np.float32) / 256.0
            frame_list.append(img_norm)
        cap.release()
        return frame_list
    
    def generate_batches(self):
        
        for vid_path in self.path_df:
            
            #extraction frames using frame_extraction
            frames = self.frame_extraction(vid_path)

            # Determine the number of extra frames
            extra_frames = len(frames) % self.sequence_length

            # Check if there are extra frames and adjust the number of frames
            if extra_frames > 0:
                # Calculate the number of frames to add to the next sequence
                extra_to_next_sequence = self.sequence_length - extra_frames
                frames += frames[:extra_to_next_sequence]

            # Organize frames into sequences
            sequence_frames = [frames[i:i + self.sequence_length] for i in range(0, len(frames), self.sequence_length)]

            # Determine the number of extra sequences
            extra_sequences = len(sequence_frames) % self.batch_size

            # Check if there are extra sequences and adjust the number of sequences
            if extra_sequences > 0:
                # Calculate the number of sequences to add to the next batch
                extra_to_next_batch = self.batch_size - extra_sequences
                sequence_frames += sequence_frames[:extra_to_next_batch]
            # Create batches of sequences
            for x in range(0, len(sequence_frames), self.batch_size):
                batch = sequence_frames[x:x + self.batch_size]
                yield batch