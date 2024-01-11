import numpy as np

def batch_sra(inp, out):
    inp_array = inp.cpu().detach().numpy()
    out_array = out.cpu().detach().numpy()
    recons_cost = np.linalg.norm(np.subtract(inp_array, out_array))
    sa = (recons_cost - np.min(recons_cost)) / np.max(recons_cost)
    sr = 1.0 - sa
    return sr

def sequence_sra(input_, output_, sequence_length):
    sra_list = []
    for sequence_index in range(sequence_length):
        inp_array = input_.cpu().detach().numpy()
        out_array = output_.cpu().detach().numpy()
        recons_cost = np.linalg.norm(np.subtract(inp_array[:, sequence_index, :, :], out_array[:, sequence_index, :, :]))
        sa = (recons_cost - np.min(recons_cost)) / np.max(recons_cost)
        sra_list.append(sa)
    return sra_list

def frame_sra(input_, output_, sequence_length):
    sra_list = []
    for j, (in_batch, out_batch) in enumerate(zip(input_, output_)):
        for sequence_index in range(sequence_length):
            inp_array = in_batch.cpu().detach().numpy()
            out_array = out_batch.cpu().detach().numpy()
            recons_cost = np.linalg.norm(np.subtract(inp_array[j][:, sequence_index, :, :], out_array[j][:, sequence_index, :, :]))
            sa = (recons_cost - np.min(recons_cost)) / np.max(recons_cost)
            sra_list.append(sa)
    return sra_list

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Assuming you have a list of 240 points (replace data_list with your actual list)
data_list = np.load('Arrest_001_sr_.npy')

# Initialize the plot
fig, ax = plt.subplots()
line, = ax.plot(data_list, label='Data')

ax.set_xlim(0, len(data_list))
ax.set_ylim(0, 2)
ax.set_xlabel('frame(t)')
ax.set_ylabel('SignularityScore(sr)')
ax.set_yticks(np.arange(0.0,2, 0.2).tolist())
ax.legend(loc='lower right')

# Function to update the plot in each iteration
def update_plot(i):
    # Update the data in the plot
    line.set_data(range(i), data_list[:i])

    return line,

# Create animation
ani = FuncAnimation(fig, update_plot, frames=len(data_list), interval=100, blit=True)
plt.savefig('arrest_001')
# Display the plot
plt.show()
