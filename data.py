import random
import numpy as np
import matplotlib.pyplot as plt

image_size = 28 # width and length
no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size
data_path = "data/mnist/"
train_data = np.loadtxt(data_path + "mnist_train.csv", 
                        delimiter=",")
test_data = np.loadtxt(data_path + "mnist_test.csv", 
                       delimiter=",")

train = []
for d in train_data:
    out = np.array([0,0,0,0,0,0,0,0,0,0])
    out[int(d[0])]=1
    out = out.reshape(10,1)
    inp = d[1::]
    inp = inp.reshape(inp.size,1)/256
    train.append([inp,out])
train = np.array(train)
np.save('data/mnist/train',train)
test = []
for d in test_data:
    out = np.array([0,0,0,0,0,0,0,0,0,0])
    out[int(d[0])]=1
    out = out.reshape(10,1)
    inp = d[1::]
    inp = inp.reshape(inp.size,1)/256
    test.append([inp,out])
test = np.array(test)
np.save('data/mnist/test',test)