import torch
from torch.utils.data import Dataset, DataLoader
from Data_generators import VideoDataGenerator
from utils import w_pathlist_1
class VideoDataset(Dataset):
    def __init__(self, video_paths,batch_size,sequence_length=10):
        self.video_paths = video_paths
        self.sequence_length = sequence_length
        self.batch_size=batch_size
        self.data_generator = VideoDataGenerator(video_paths,batch_size,sequence_length=sequence_length)

    def __len__(self):
        # Define the length of the dataset (number of batches)
        return len(self.data_generator.path_df) // self.data_generator.batch_size

    def __getitem__(self, idx):
        # Use the VideoDataGenerator to generate the next batch
        batch = next(self.data_generator.generate_batches())
        return torch.from_numpy(batch)  # Assuming frames are numpy arrays

# Example usage:
video_paths =  w_pathlist_1 # Your list of video paths
sequence_length = 10
batch_size = 32

video_dataset = VideoDataset(video_paths,batch_size, sequence_length=sequence_length)
data_loader = DataLoader(video_dataset, batch_size=batch_size, shuffle=True)

# Iterate over batches in the training loop
for batch in data_loader:
    # Process the batch and train your model
    print(batch.shape)  # Example: torch.Size([32, 10, 256, 256, 3])
