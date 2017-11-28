def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

import numpy as np
cat_data = np.zeros((10000,3073))
train_data = unpickle('data_batch_1')
for i in range(0,10000):
	 if train_data['labels'][i] == 3:
		 cat_data[i,0:3072] = train_data['data'][i]
		 cat_data[i,3072] = 1
	 else:
		 cat_data[i,0:3072] = train_data['data'][i]
		 cat_data[i,3072] = 0
cat_data_train = np.zeros((600,3073))
cat_data_dev = np.zeros((200,3073))
cat_data_test = np.zeros((200,3073))
cat_data_train = cat_data[0:600,:]

cat_data_dev = cat_data[600:800,:]
cat_data_test = cat_data[800:1000,:]
cat_data_train[:,0:3072] = cat_data_train[:,0:3072]  - np.mean(cat_data_train[:,0:3072],axis = 0)
cat_data_dev[:,0:3072] = cat_data_dev[:,0:3072]  - np.mean(cat_data_dev[:,0:3072],axis = 0)
cat_data_train[:,0:3072] = cat_data_train[:,0:3072]  / np.std(cat_data_train[:,0:3072],axis = 0)
cat_data_dev[:,0:3072] = cat_data_dev[:,0:3072]  / np.std(cat_data_dev[:,0:3072],axis = 0)


cat_data_train = np.transpose(cat_data_train) 
cat_data_dev = np.transpose(cat_data_dev) 
cat_data_test = np.transpose(cat_data_test) 

def activation(z):
    output = np.tanh(z)
    return output
def sigmoid(z):
	
    output = 1/(1+np.exp(-z))
    return output

def initialize(n_x,n_h,n_y):

    w1 = np.random.rand(n_x,n_h) * 0.01
    b1 = np.random.rand(n_h,1) * 0.01
    w2 = np.random.rand(n_h,n_y) * 0.01
    b2 = np.random.rand(n_y,1) * 0.01
    return w1,w2,b1,b2 
	
def forward_prop(w1,b1,w2,b2,x_input):

	z1 = np.dot(np.transpose(w1),x_input) + b1
	a1 = activation(z1)
	

	z2 = np.dot(np.transpose(w2),a1) + b2
	#print w2
	#print w1
	
	result = sigmoid(z2)
	a2 = result
	return a2,z1,a1,z2
def cost(y_output, result):
    loss = np.sum(np.multiply(y_output, np.log(result)) + np.multiply(1-y_output, np.log(1-result)))
    return loss

def backprop(a1,a2,x,w1,w2,y):
	dz2 = a2 - y
	db2 = dz2
	dw2 = np.dot(dz2,np.transpose(a1))
	print a1
	dz1 = np.dot(w2 , dz2) *(1-a1**2)
	print 'hello'
	print dz1
	dw1 = np.dot(x,np.transpose(dz1))
	print dw1
	db1 = dz1
	return dw1,dw2,db1,db2,dz1,dz2
def parameter_update(dw1,dw2,db1,db2,w1,w2,b1,b2,learning_rate):
    w1 = w1 - learning_rate*dw1
    w2 = w2-learning_rate*dw2
    b1 = b1 - learning_rate*db1
    b2 = b2 - learning_rate*db2
    
    return b1,b2,w1,w2

def model(x_input, y_output, learning_rate,num_iteration, hidden_layer):
	w1,w2,b1,b2 = initialize(x_input.shape[0], hidden_layer,1)
	for i in range(num_iteration) :
		a2,z1,a1,z2 = forward_prop(w1,b1,w2,b2,x_input)
		model_cost =[]
		model_cost.append(cost(y_output,a2))
		t1,t2,t3,t4,t5,t6 = backprop(a1,a2,x_input,w1,w2,y_output)
		b1,b2,w1,w2 = parameter_update(t1,t2,t3,t4,w1,w2,b1,b2,learning_rate)
	return model_cost

c = model(cat_data_train[0:3072,:],cat_data_train[3072,:],0.05,1000,1000)
#print activation (10)
# import matplotlib.pyplot as plt
# plt.plot(c)
# plt.show() 
    