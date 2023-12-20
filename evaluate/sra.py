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

