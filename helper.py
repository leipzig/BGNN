def getModelName(hyperp):
    imgH = hyperp["imgH"]
    imgW = hyperp["imgW"]
    n_epochs = hyperp["n_epochs"]
    training_count = hyperp["training_count"]
    validation_count = hyperp["validation_count"]
    kernels = hyperp["kernels"]
    kernelSize = hyperp["kernelSize"]
    return ('BGNN_H%s_W%s_n%s_train%s_val%s_k%s_ksize%s'
     ) % (imgH, imgW, n_epochs, training_count, validation_count, len(kernels), kernelSize)