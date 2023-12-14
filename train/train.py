from tqdm import tqdm
epochs = 10

for epoch in range(epochs):
    print('Epoch : ', epoch + 1)
    running_loss = 0.0
    
    for i, (batch, num_frames_read) in enumerate(dataset_generator._generator_batches_()):
        input_ = torch.tensor(batch, dtype=torch.float32).unsqueeze(2).to(device)
        
        optimizer.zero_grad()
        output = model(input_)
        loss = criterion(output, input_)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if i % 10== 0:
            print(f"Loss : {running_loss / min(i + 1, 100)}")  # Averaging over the last 100 batches
            running_loss = 0.0

