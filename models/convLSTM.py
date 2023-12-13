import torch
import torch.nn as nn 
class TimeDistribution(nn.Module):
    def __init__(self, timesteps, in_channels, out_channels, kernel_size, stride, padding):
        super(TimeDistribution, self).__init__()
        self.layer = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding) for _ in range(timesteps)])
        #self.layer_norm = nn.LayerNorm([out_channels, H, W])  # You need to define H and W

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        output = torch.tensor([])

        for i in range(timesteps):
            output_t = self.layer[i](x[:, i, :, :, :])
            output_t = output_t.unsqueeze(1)
            output = torch.cat((output, output_t), 1)

        #output = self.layer_norm(output)
        return output

class TimeTransposeDistribution(nn.Module):
    def __init__(self, timesteps, in_channels, out_channels, kernel_size, stride, padding):
        super(TimeTransposeDistribution, self).__init__()
        self.layer = nn.ModuleList([
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding) for _ in range(timesteps)
        ])

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        output = []

        for i in range(timesteps):
            output_t = self.layer[i](x[:, i, :, :, :])
            output.append(output_t.unsqueeze(1))

        output = torch.cat(output, dim=1)

        return output

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias, padding):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.bias = bias
        self.padding = padding

        self.layer = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                               out_channels=4 * self.hidden_dim,
                               kernel_size=self.kernel_size,
                               padding=self.padding)

    def forward(self, x, cur_state):
        h_state, c_state = cur_state

        input_ = torch.cat([x, h_state], 1)
        layer_output = self.layer(input_)

        cc_i, cc_f, cc_o, cc_g = torch.split(layer_output, self.hidden_dim, 1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_state + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.layer.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.layer.weight.device))


class ConvLSTM(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,bias,padding):
        super(ConvLSTM,self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.bias=bias
        self.padding=padding

        self.convlstm_cell=ConvLSTMCell(in_channels,out_channels,kernel_size,bias,padding)

    def forward(self,x):
        batch_size,sequence_len,_,width,height=x.size()
        image_size=width,height
        output=torch.zeros(batch_size,sequence_len,self.out_channels,width,height)
        hidden_state,cell_state=self.convlstm_cell.init_hidden(batch_size,image_size)

        for t in range(sequence_len):
            hidden_state,cell_state=self.convlstm_cell(x[:,t,:,:,:],(hidden_state,cell_state))
            output[:,t,:,:,:]=hidden_state
        return output

class ActivationTimeDistribution(nn.Module):
    def __init__(self, timesteps, in_channels, out_channels, kernel_size, activation):
        super(ActivationTimeDistribution, self).__init__()

        # Calculate "same" padding
        padding = (kernel_size - 1) // 2

        # Use nn.ModuleList for convolutional layers
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
            for _ in range(timesteps)
        ])

        # Use nn.ModuleList for activation functions
        self.activations = nn.ModuleList([
            nn.Sigmoid() if activation == "sigmoid" else nn.ReLU()  # Add more activations as needed
            for _ in range(timesteps)
        ])

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        output = []

        for i in range(timesteps):
            layer_output = self.layers[i](x[:, i, :, :, :])
            layer_output = self.activations[i](layer_output)
            output.append(layer_output.unsqueeze(1))

        # Concatenate along the channel dimension
        output = torch.cat(output, dim=1)

        return output

    import torch
import torch.nn as nn
from collections import OrderedDict

class EncoderDecoderModel(nn.Module):
    def __init__(self, timesteps, in_channels):
        super().__init__()
        self.time_steps = timesteps
        self.in_channels = in_channels

        self.encoder = nn.Sequential(OrderedDict([
            ('TD_Layer_1', TimeDistribution(self.time_steps, self.in_channels, out_channels=128, kernel_size=(11, 11), stride=4, padding=5)),
            ('TD_Layer_2', TimeDistribution(self.time_steps, in_channels=128, out_channels=64, kernel_size=(5, 5), stride=2, padding=2)),
            ('convlstm_layer1', ConvLSTM(in_channels=64, out_channels=64, kernel_size=(3, 3), bias=True, padding=1)),
            ('convlstm_layer2', ConvLSTM(in_channels=64, out_channels=32, kernel_size=(3, 3), bias=True, padding=1)),
            ('convlstm_layer3', ConvLSTM(in_channels=32, out_channels=64, kernel_size=(3, 3), bias=True, padding=1))
        ]))

        self.decoder = nn.Sequential(OrderedDict([
            ('TTD_Layer_1', TimeTransposeDistribution(self.time_steps, in_channels=64, out_channels=64, kernel_size=(4, 4), stride=2, padding=1)),
            ('TTD_Layer_2', TimeTransposeDistribution(self.time_steps, in_channels=64, out_channels=128, kernel_size=(4, 4), stride=4, padding=0)),
            ('TDA_Layer_1', ActivationTimeDistribution(self.time_steps, in_channels=128, out_channels=1, kernel_size=3, activation="sigmoid"))
        ]))

    def forward(self, x):
        print(f"Input : {x.shape}")
        batch_size, channels, timesteps, H, W = x.size()
        
        # Encoder
        encoder_output = self.encoder(x)
        
        # Decoder
        decoder_output = self.decoder(encoder_output)
        
        print(f"Output: {decoder_output.shape}")
        return decoder_output

# Instantiate the EncoderDecoderModel
timesteps = 10  # Adjust based on your specific requirements
in_channels = 1 # Adjust based on your input size
model = EncoderDecoderModel(timesteps, in_channels)
print(model)
# Example input
input_data = torch.randn(2,  timesteps,in_channels, 256, 256)

# Forward pass for inference
output_data = model(input_data)
