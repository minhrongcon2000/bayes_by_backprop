import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(0)

def sigmoid(x):
  return 1/(1+np.exp(-x))

def softplus(x):
  return 1+np.log(np.exp(x))

def crossentropy(pred,label):
  return np.sum(-label*np.log(pred)-(1-label)*np.log(1-pred))


data = np.array([[0,0,0],
        [0,1,0],
        [1,0,0],
        [1,1,1]])

# initialize weights
w = np.random.random((data[:,2].reshape(1,-1).shape[0],data[:,:2].shape[1]))
b = np.random.random((data[:,2].reshape(1,-1).shape[0],1))

# hyperparameters
lr = .3

# logs
errors = []
acc = []
ws = []
bs = []

for i in range(30000):
  np.random.shuffle(data)
  x = data[:,:2]
  y = data[:,2].reshape(1,-1)

  pred = sigmoid(w @ x.T + b)
  
  errors.append(crossentropy(pred,y))

  error = pred - y

  # calculate derivative of crossentropy loss wrt w,b
  error = pred - y
  dw_crossentropy = error.dot(x)
  db_crossentropy = np.sum(error,axis=1,keepdims=True)

  # update weights
  w = w - lr*dw_crossentropy
  b = b - lr*db_crossentropy

errors = np.array(errors)

np.savetxt("result/error/errors_normal_nn.txt",errors)


