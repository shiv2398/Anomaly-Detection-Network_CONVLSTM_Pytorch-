import torch
import torch.nn as nn 
class TimeDistributionTranspose(nn.Module):
    def __init__(self,timesteps,in_channels,out_channels,kernel_size,stride,padding):
        super(TimeDistribution,self).__init__()
        self.layer=nn.ModuleList([nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride,paddding) for _ in range(timesteps)])
        self.layer_norm=nn.LayerNorm([out_channels,H,W])
        
    def forward(self,x):
        batch_size,timesteps,C,H,W=x.size()
        output=torch.tensor([])
        for i in range(timesteps):
            output_t=self.layer(x)[i][:,i,:,:,:]
            output_t=output_t.unsqueeze(1)
            output=torch.cat((output,output_t),1)
        output=self.layer_norm(output)
        return output
class TimeDistribution(nn.Module):
    def __init__(self,timesteps,in_channels,out_channels,kernel_size,stride,padding):
        super(TimeDistribution,self).__init__()
        self.layer=nn.ModuleList([nn.Conv2d(in_channels,out_channels,kernel_size,stride,paddding) for _ in range(timesteps)])
        self.layer_norm=nn.LayerNorm([out_channels,H,W])
        
    def forward(self,x):
        batch_size,timesteps,C,H,W=x.size()
        output=torch.tensor([])
        for i in range(timesteps):
            output_t=self.layer(x)[i][:,i,:,:,:]
            output_t=output_t.unsqueeze(1)
            output=torch.cat((output,output_t),1)
        output=self.layer_norm(output)
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
        self.layer_norm = nn.LayerNorm(normalized_shape=[4 * self.hidden_dim, height, width])

    def forward(self, x, cur_state):
        h_state, c_state = cur_state

        input_ = torch.cat([x, h_state], 1)
        
        layer_output=self.layer_norm(self.layer(input_))
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
class EncoderDecoderModel(nn.Module):
    
    def __init__(self,timesteps,in_channel,hidden_dim,bias,padding):
        super(EncoderDecoderModel).__init__()
        self.in_channel=in_channel
        self.hidden_dim=hidden_dim
        self.timesteps=timesteps
        # Encoder Structure
         #TimeDistributed(Conv2D) with 128 filters, (11, 11) kernel, strides=4, padding="same"
        self.layer1 = TimeDistribution(timesteps=self.timesteps,
                                       in_channels=self.in_channel,
                                       out_channels=128,kernel_size=(11,11),stride=4,padding=0)
        #TimeDistributed(Conv2D) with 64 filters, (5, 5) kernel, strides=2, padding="same"
        self.layer2=TimeDistribution(timesteps=self.timesteps,
                                    in_channels=128,out_channels=64,kernel_size=(5,5),stride=2,padding=0)
        #ConvLSTM2D with 64 filters, (3, 3) kernel, padding="same", return_sequences=True
        self.layer3=ConvLSTMCell(input_dim=self.input_dim,
                                 hidden_dim=64, kernel_size=(3,3), bias, padding)
        #
        self.layer4=ConvLSTMCell(input_dim=self.input_dim,
                                 hidden_dim=32, kernel_size=(3,3), bias, padding)
        self.layer5=ConvLSTMCell(input_dim=self.input_dim,
                                 hidden_dim=64, kernel_size=(3,3), bias, padding)
        
        # Decoder Model
        
        #TimeDistributed(Conv2D) with 128 filters, (11, 11) kernel, strides=4, padding="same"
        self.layer1 = TimeDistributionTranspose(timesteps=self.timesteps,
                                       in_channels=self.in_channel,
                                       out_channels=64,kernel_size=(11,11),stride=4,padding=0)
        #TimeDistributed(Conv2D) with 64 filters, (5, 5) kernel, strides=2, padding="same"
        self.layer2=TimeDistributionTranspose(timesteps=self.timesteps,
                                    in_channels=128,out_channels=64,kernel_size=(5,5),stride=2,padding=0)
         #TimeDistributed(Conv2D) with 64 filters, (5, 5) kernel, strides=2, padding="same"
        self.layer2=TimeDistribution(timesteps=self.timesteps,
                                    in_channels=128,out_channel=1,kernel_size=(11,11),stride=2,padding=0)
        
