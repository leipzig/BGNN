import matplotlib.pylab as plt
import math
import itertools
import os
import torch
from CNN import CNN
import progressbar


def get_CNN_premitive_layers(model):
    result = []
    children = list(model.children())
    if len(children) == 0:
        result = [model]
    else:
        for child in children:
            child_results = get_CNN_premitive_layers(child)
            result = result + child_results
    return result

def get_premitive_layers(model, usePreTrained=False):
    if usePreTrained:
        result = [model.conv1,
          model.bn1,
          model.relu,
          model.maxpool,
          model.layer1,
          model.layer2,
          model.layer3,
          model.layer4,
          model.avgpool]
    else:
        result = get_CNN_premitive_layers(model)[:-1]
    return result

class model_activations(torch.nn.Module):
    def __init__(self, original_model, layer_num):
        super(model_activations, self).__init__()
        if layer_num == -1:
            self.features = original_model
        else:    
            usePreTrained = not isinstance(original_model, CNN)
            layers = get_premitive_layers(original_model, usePreTrained)[:layer_num+1]
            self.features = torch.nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.features(x)
        return x

# Define the function for plotting the channels. The rows are inputs and the columns are outputs
# Define the function to plot out the kernel parameters of each channel 
# Plots a grid from x inputs to y outputs.
def plot_channels(model, layer_num, experimentName, number_per_row=8):
    layers = list(map(lambda x: x[1], model.named_parameters()))

#     W  = model.state_dict()[layer_name]
#     sub_layers = list(layers[layer_num].named_parameters())
    W = layers[layer_num]
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
        ax.imshow(W[out_index, in_index, :, :].detach().cpu().numpy(), vmin=w_min, vmax=w_max, cmap='seismic')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xlabel("o_" +str(out_index) + "-in_" + str(in_index))
        in_index = in_index + 1

    name="Parameters of layer "+str(layer_num)
    plt.suptitle(name, fontsize=10)   
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(experimentName, name+"_channels.pdf"))
    plt.show()
    
# Define the function for plotting the activations
def plot_activations(model, layer_num, input_img, experimentName, title="", number_per_row=4):
    sub_model = model_activations(model, layer_num)
    A = sub_model(input_img)
    
    A = A[0, :].detach().cpu().detach().numpy()
    n_activations = A.shape[0]
    A_min = A.min().item()
    A_max = A.max().item()

    if (A.ndim > 2 and A.shape[2]>1):
        number_rows = math.ceil(n_activations / number_per_row)
        fig, axes = plt.subplots(number_rows,number_per_row, figsize=(15, 2*number_rows), dpi= 300)
        with progressbar.ProgressBar(maxval=number_rows*number_per_row, redirect_stdout=True) as bar:
            for i, ax in enumerate(axes.flat):
                bar.update(i)
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
        A = A.squeeze()
        thresh = A.max() / 1.5
        fig, axes = plt.subplots(1, 1, figsize=(15, 2), dpi= 300)
        axes.imshow(A.reshape(1, n_activations), vmin=A_min, vmax=A_max, cmap='Blues')
        with progressbar.ProgressBar(maxval=n_activations, redirect_stdout=True) as bar:
            for i in range(n_activations):
                bar.update(i)
                plt.text(i, 0, "{:0.2f}".format(A[i]),
                         horizontalalignment="center",
                         color="white" if A[i] > thresh else "black")


    plt.suptitle(title, fontsize=10)   
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(experimentName, title+"_activations.pdf"))
    plt.show()