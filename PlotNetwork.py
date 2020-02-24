import matplotlib.pylab as plt
import math
import itertools

# Define the function for plotting the channels. The rows are inputs and the columns are outputs
# Define the function to plot out the kernel parameters of each channel 
# Plots a grid from x inputs to y outputs.
def plot_channels(model, layer_num, number_per_row=8):
    name="Parameters of layer "+str(layer_num)

    W  = model.state_dict()['module_list.'+str(layer_num*3)+'.weight']
    n_out = W.shape[0]
    n_in = W.shape[1]
    w_min = W.min().item()
    w_max = W.max().item()
    if number_per_row is None:
        number_per_row = n_in
    n_rows = math.ceil(n_out*n_in/number_per_row)
    fig, axes = plt.subplots(n_rows, number_per_row, figsize=(15, 2*n_rows), dpi= 300)

    out_index = 0
    in_index = 0

    #plot outputs as rows inputs as columns 
    for ax in axes.flat:
        if out_index*n_in + in_index >= number_per_row*n_rows:
            break
        if in_index > n_in-1:
            out_index = out_index + 1
            in_index = 0
        ax.imshow(W[out_index, in_index, :, :].cpu().numpy(), vmin=w_min, vmax=w_max, cmap='seismic')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xlabel("o_" +str(out_index) + "-in_" + str(in_index))
        in_index = in_index + 1

    plt.suptitle(name, fontsize=10)   
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
#     plt.savefig(savedModelName+"/"+name+"_channels.png")

    
# Define the function for plotting the parameters (kernels)
# Define the function to plot out the kernel parameters of each channel with Multiple outputs . 
# The inpt defines which input you are plotting parameters for.
def plot_parameters(model, layer_num, inpt=0, number_per_row=8):
    W  = model.state_dict()['module_list.'+str(layer_num*3)+'.weight']
    name="Kernels of layer "+str(layer_num)+" for input " + str(inpt) 

    W = W.data[:, inpt, :, :]
    n_filters = W.shape[0]
    w_min = W.min().item()
    w_max = W.max().item()
    n_rows = math.ceil(n_filters/number_per_row)
    fig, axes = plt.subplots(n_rows, number_per_row, figsize=(15, 2*n_rows), dpi= 300)

    for i, ax in enumerate(axes.flat):
        if i < n_filters:
            # Set the label for the sub-plot.
            ax.set_xlabel("kernel:{0}".format(i + 1))

            # Plot the image.
            ax.imshow(W[i, :].cpu().numpy(), vmin=w_min, vmax=w_max, cmap='seismic')
            ax.set_xticks([])
            ax.set_yticks([])
    plt.suptitle(name, fontsize=10)    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
#     plt.savefig(savedModelName+"/"+name+"_parameters.png")
    
# Define the function for plotting the activations
def plot_activations(A, number_rows=4, name=""):
    A = A[0, :].cpu().detach().numpy()
    n_activations = A.shape[0]
    A_min = A.min().item()
    A_max = A.max().item()

    if (A.ndim > 2):
        fig, axes = plt.subplots(number_rows, math.ceil(n_activations / number_rows), figsize=(15, 2*number_rows), dpi= 300)
        for i, ax in enumerate(axes.flat):
            if i < n_activations:
                # Set the label for the sub-plot.
                ax.set_xlabel("activation:{0}".format(i + 1))

                # Plot the image.
                ax.imshow(A[i], vmin=A_min, vmax=A_max, cmap='seismic')
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                break
    else:
        thresh = A.max() / 1.5
        fig, axes = plt.subplots(1, 1, figsize=(15, 2), dpi= 300)
        axes.imshow(A.reshape(1, n_activations), vmin=A_min, vmax=A_max, cmap='Blues')
        for i in range(n_activations):
            plt.text(i, 0, "{:0.2f}".format(A[i]),
                     horizontalalignment="center",
                     color="white" if A[i] > thresh else "black")


    plt.suptitle(name, fontsize=10)   
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
#     plt.savefig(savedModelName+"/"+name+"_activations.png")