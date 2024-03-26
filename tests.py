from network import Network
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import freeze_support
import math
from digits import bmptoarray,show,nberreurs

n = Network.load("untitled")

def train(a,b,n):
    for i in range(a):
        for j in range(b):
            n.learn('data/mnist/train.npy',rate = 0.05)
            n.save()

train(1,10,n)
#show(t[0][0])

nberreurs("data/mnist/train.npy",n)

