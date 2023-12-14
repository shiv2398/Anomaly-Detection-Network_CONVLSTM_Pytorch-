import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel

# Assuming model is an instance of your EncoderDecoderModel
model = EncoderDecoderModel(timesteps, in_channels)



# Move the model to GPU
model = model.to(device)

# Wrap the model with DataParallel
model = DataParallel(model)

# Define your loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs=10
# Training loop
for epoch in range(epochs):
    running_loss=0.0
    print('Epoch : ',epoch+1)
    for i, (batch, num_frames_read) in enumerate(dataset_generator._generator_batches_()):
        input_ = torch.tensor(batch, dtype=torch.float32).unsqueeze(2).to(device)

        optimizer.zero_grad()
        output = model(input_)
        loss = criterion(output, input_)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
        if i % 100 == 0:
            print(f"Iteration : {i+1} | Loss: {running_loss:.4f}")
    print(f"Epoch {epoch+1} Loss : {running_loss:.4f}")
    running_loss=0.0

