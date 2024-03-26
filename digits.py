from network import Network
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import freeze_support
import math
np.seterr('raise')

def bmptoarray(bmppath):
    bmp=open(bmppath,'rb')
    byts=list(bmp.read())
    bmp.close()
    shape=(byts[18]+256*byts[19]+256*256*byts[20]+256*256*256*byts[21], byts[22]+ 256*byts[23]+256*256*byts[24]+256*256*256*byts[25])
    pixels = byts[54::]
    img = []
    start = 0
    for i in range(shape[1]):
        line=[]
        for j in range(shape[0]):
            b=3*j+start
            line.append(255-pixels[b])
            1+1
        img.append(line)       
        a = shape[1]*3
        start += math.ceil(a/4.)*4
    img = np.array(img)[-1::-1]
    return img

#n=Network.load('digitrecognition')

def show(e,n):
    img = e.reshape((28,28))
    plt.imshow(img, cmap="Greys")        
    plt.show()
    m=0
    a=n.run(e.reshape((28*28,1)))
    for i in range(10):
        r=a[i]
        if a[i]>m:
            m=r
            u=i
    print(u,"  ",int(m*100),"%")
        
def nberreurs(path,n):
    ex = np.load(path,allow_pickle=True)
    cnt = 0
    erreurs = []
    for e in range(10000):
        m=0
        a=n.run(ex[e][0])
        for i in range(10):
            r=a[i]
            if r>m:
                m=r
                u=i
        at=ex[e][1]
        mt = 0
        for it in range(10):
            rt=at[it]
            if rt>mt:
                mt=rt
                ut=it
        if not(ut==u):
            erreurs.append([ex[e][0],u,int(100*m)])
            cnt += 1
    for err in erreurs:
        '''
        img = err[0].reshape((28,28))
        plt.imshow(img, cmap="Greys")
        plt.show()
        '''
        print(err[1],"  ",err[2],"%")
    print(cnt)

def train(a,b):
    for i in range(a):
        for j in range(b):
            n.learn('data/mnist/train.npy',rate = 0.002)
            n.save()


#show(bmptoarray('data/2.bmp'))