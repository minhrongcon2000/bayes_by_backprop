import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(0)

def gaussian_pdf(x,mu,sigma):
	'''
	Args:
	x     -  float: input
	mu    -  float: Gaussian mean
	sigma -  float: Gaussian variance
	'''
	return 1/(np.sqrt(2*np.pi*sigma))*np.exp(-0.5*(x-mu)*(x-mu)/sigma)

def sigmoid(x):
	return 1/(1+np.exp(-x))

def softplus(x):
	return 1+np.log(np.exp(x))

def crossentropy(pred,label):
	return np.sum(label*np.log(pred)+(1-label)*np.log(1-pred))

def prior_loss(w,b,cov_w,cov_b):
	prior_w = - 0.5*np.log(2*np.pi*cov_w*cov_w) - 0.5*(w*w)/(cov_w*cov_w)
	prior_b = - 0.5*np.log(2*np.pi*cov_b*cov_b) - 0.5*(b*b)/(cov_b*cov_b)

	return np.sum(prior_w) + np.sum(prior_b)

def scale_mixture_prior(w,b,cov_1=1,cov_2=1e-6,p=0.25):
	return np.sum(np.log(p*gaussian_pdf(w,0,cov_1*cov_1)+(1-p)*gaussian_pdf(w,0,cov_2*cov_2))) + np.sum(np.log(p*gaussian_pdf(b,0,cov_1*cov_1)+(1-p)*gaussian_pdf(b,0,cov_2*cov_2)))

def variantion_loss(w,b,mean_w,mean_b,cov_w,cov_b):
	variation_w = - 0.5 * np.log(2*np.pi*cov_w*cov_w) - 0.5 * ((w - mean_w)*(w - mean_w))/(cov_w*cov_w)
	variation_b = - 0.5 * np.log(2*np.pi*cov_b*cov_b) - 0.5 * ((b - mean_b)*(b - mean_b))/(cov_b*cov_b)
	return np.sum(variation_w) + np.sum(variation_b)

def total_loss(pred,label,w,b,mean_w,mean_b,cov_w,cov_b):
	return variantion_loss(w,b,mean_w,mean_b,cov_w,cov_b) - crossentropy(pred,label) - prior_loss(w,b,cov_w,cov_b)

def total_loss_with_scale(pred,label,w,b,mean_w,mean_b,cov_w,cov_b,p=0.25,cov_1=1,cov_2=1e-6):
	return variantion_loss(w,b,mean_w,mean_b,cov_w,cov_b) - crossentropy(pred,label) - scale_mixture_prior(w,b,cov_1,cov_2,p)


data = np.array([[0,0,0],
			  [0,1,0],
			  [1,0,0],
			  [1,1,1]])

# initialize weights
mean_w = 2*np.random.random((data[:,2].reshape(1,-1).shape[0],data[:,:2].shape[1]))-1
mean_b = 2*np.random.random((data[:,2].reshape(1,-1).shape[0],1))-1
p_w = 2*np.random.random((data[:,2].reshape(1,-1).shape[0],data[:,:2].shape[1]))-1
p_b = 2*np.random.random((data[:,2].reshape(1,-1).shape[0],1))-1

# hyperparameters
lr = 1e-5
p_scale = .25
cov_1 = 1
cov_2 = 1e-6

# logs
errors = []
acc = []
ws = []
bs = []

for i in range(10000):
	np.random.shuffle(data)
	x = data[:,:2]
	y = data[:,2].reshape(1,-1)
	eps_w = np.random.multivariate_normal(np.zeros(mean_w.flatten().shape),
								np.eye(mean_w.flatten().shape[0]),size=mean_w.shape[0])
	eps_b = np.random.multivariate_normal(np.zeros(mean_b.flatten().shape),
								np.eye(mean_b.flatten().shape[0]),size=mean_b.shape[0])

	cov_w = softplus(p_w)
	cov_b = softplus(p_b)

	w = mean_w + cov_w * eps_w
	b = mean_b + cov_b * eps_b

	pred = sigmoid(w @ x.T + b)

	# if i%100==0:
	# 	print(pred)
	
	true_pred = np.sum(((pred >= 0.5).astype(np.int) == y).astype(int))
	errors.append(total_loss_with_scale(pred,y,w,b,mean_w,mean_b,cov_w,cov_b,p_scale,cov_1,cov_2))
	acc.append(true_pred/4)
	ws.append(w)
	bs.append(b)

	# calculate derivative of crossentropy loss wrt w,b
	error = y - pred
	dw_crossentropy = error.dot(x)
	db_crossentropy = np.sum(error,axis=1,keepdims=True)

	# calculate derivative of variational loss wrt w,b
	dw_variation = - (w - mean_w)/(cov_w*cov_w)
	db_variation = - (b - mean_b)/(cov_b*cov_b)

	# calculate derivative of variational loss wrt mean_w, mean_b
	dmean_w_variation = np.sum((w - mean_w)/(cov_w*cov_w))
	dmean_b_variation = np.sum((b - mean_b)/(cov_b*cov_b))

	# calculate derivative of variational loss wrt cov_w, cov_b
	dcov_w_variation = np.sum(((w - mean_w)*(w - mean_w) - cov_w*cov_w)/(cov_w*cov_w*cov_w))
	dcov_b_variation = np.sum(((b - mean_b)*(b - mean_b) - cov_b*cov_b)/(cov_b*cov_b*cov_b))

	# calculate the scale mixture prior wrt w
	dpdw = p_scale*gaussian_pdf(w,0,cov_1*cov_1)*(-w/(cov_1*cov_1)) + (1-p_scale)*gaussian_pdf(w,0,cov_2*cov_2)*(-w/(cov_2*cov_2))
	dw_scale = dpdw*1/np.exp(scale_mixture_prior(w,b,cov_1,cov_2,p_scale))

	dpdb = p_scale*gaussian_pdf(b,0,cov_1*cov_1)*(-b/(cov_1*cov_1)) + (1-p_scale)*gaussian_pdf(b,0,cov_2*cov_2)*(-b/(cov_2*cov_2))
	db_scale = dpdb*1/np.exp(scale_mixture_prior(w,b,cov_1,cov_2,p_scale))	

	# calculate the derivative of final loss wrt w,b
	dw = dw_variation - dw_scale - dw_crossentropy
	db = db_variation - db_scale - db_crossentropy

	# calculate the derivative of final loss wrt mean_w, mean_b
	dmean_w = dw + dmean_w_variation
	dmean_b = db + dmean_b_variation

	# calculate the derivative of final loss wrt p_w, p_b
	dp_w = dw*eps_w*sigmoid(p_w) + dcov_w_variation*sigmoid(p_w)
	dp_b = db*eps_b*sigmoid(p_b) + dcov_b_variation*sigmoid(p_b)

	# update the mean_w, mean_b, p_w, p_b
	mean_w = mean_w - lr*dmean_w
	mean_b = mean_b - lr*dmean_b
	p_w = p_w - lr*dp_w
	p_b = p_b - lr*dp_b


ws = np.array(ws)
bs = np.array(bs)
errors = np.array(errors)

np.savetxt("result/weights/weight1_bbb_nn.txt",ws[:,:,0].flatten())
np.savetxt("result/weights/weight2_bbb_nn.txt",ws[:,:,0].flatten())
np.savetxt("result/weights/bias_bbb_nn.txt",bs[:,:,0].flatten())
np.savetxt("result/weights/errors_bbb_nn.txt",errors)

















