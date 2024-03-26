import numpy as np
from multiprocessing import Pool,freeze_support


def sigmoid(a):
    np.clip(a,-300,300,a)
    return 1.0/(1.0+np.exp(-a))

def dsigmoid(a):
    return sigmoid(a)*(1-sigmoid(a))

def ReLU(a):
    return np.array([[max(0,a[i][j]) for j in range(a.shape[1])] for i in range(a.shape[0])])

class Layer:

    def __init__(self,weights,biases):
        self.weights = weights
        self.biases = biases

    def clear(self):
        self.inputs = None
        
    def run(self,inputs):
        freeze_support()
        #with Pool(15) as pool:
        self.inputs = inputs
        self.zlist = np.dot(self.weights, inputs) + self.biases
        #res = pool.map(sigmoid,self.zlist)
        res = sigmoid(self.zlist)
        #pool.close()
        #pool.join()
        return np.array(res)

    def __repr__(self):
        str = ""
        for w,b in zip(self.weights,self.biases):
            str += np.array_str(w) +"  " + np.array_str(b) +"\n"
        return str

    def backprop(self,delta,rate):
        freeze_support()
        #with Pool(15) as pool:
        #self.around()
        #delta = delta*pool.map(dsigmoid,self.zlist)
        delta = delta*dsigmoid(self.zlist)
        #pool.close()
        #pool.join()
        self.biases += rate*delta
        self.weights += rate*self.inputs.reshape(1,self.inputs.size)*delta
        a = delta.reshape(1,delta.size)
        deltaback = np.dot(a,self.weights)
        #np.around(deltaback,100,deltaback)
        return deltaback.reshape(deltaback.size,1)

    def around(self):
        np.around(self.inputs,100,self.inputs)
        np.around(self.weights,100,self.weights)
        np.around(self.biases,100,self.biases)
        np.around(self.zlist,100,self.zlist)

class Network:

    def __init__(self,weights,biases,name='untitled'):
        self.name = name
        self.layers = np.array([ Layer(w,b) for w,b in zip(weights,biases) ])

    @classmethod
    def random(cls,size,name = 'untitled'):
        network = cls([0],[0],name)
        templayers = []
        for i,j in zip(size[1:],size[:-1]):
            weights = np.array(np.random.rand(i,j))*2 - np.ones((i,j))
            biases = np.array(np.random.rand(i,1))*2 - np.ones((i,1))
            templayers.append(Layer(weights,biases))
        network.layers = np.array(templayers)
        return network
   
    @classmethod
    def load(cls,name):
        arr = np.load("savedNetworks/"+name+".npy",allow_pickle=True)
        net = cls([0],[0],name)
        net.layers = arr
        return net

    def __repr__(self):
        str=""
        for l in self.layers:
            str += l.__repr__() + "\n"
        return str

    def clear(self):
        for l in self.layers:
            l.clear()

    def run(self,inputs):
        inputs = np.array(inputs).reshape(len(inputs),1)
        for layer in self.layers:
            inputs = layer.run(inputs)
        return inputs

    def save(self):
        self.clear()
        np.save("savedNetworks/"+self.name,self.layers)

    def saveAs(self,name):
        self.clear()
        np.save("savedNetworks/"+name,self.layers)
  
    def backprop(self,delta,rate):  
        #np.around(delta,100,delta)
        for l in self.layers[::-1]:
            delta = l.backprop(delta,rate)
    
    def learnStep(self,ex,rate):
        expected = ex[1]
        out = self.run(ex[0])
        self.backprop(expected - out,rate)
    
    def learn(self,dataPath,n = None,rate = 1):
        data = np.load(dataPath,allow_pickle=True)
        for d in data[0:n]:
            self.learnStep(d,rate)

    def accuracy(self,dataPath,n=None):
        data = np.load(dataPath,allow_pickle=True)[0:n]
        res = 0
        for d in data:
            res += self.cost(d)
        return res/data.size

    def cost(self,ex):
        out = self.run(ex[0])
        expected = ex[1]
        return np.sum((out-expected)**2)

