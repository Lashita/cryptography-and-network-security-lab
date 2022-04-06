# Importing libraries and dataset
from __future__ import print_function
import tensorflow.compat.v1 as tf
from tensorflow.keras import layers
from keras.datasets import mnist
# from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt
import numpy as np
tf.disable_v2_behavior()
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape([-1,28,28,1])
x_train = x_train.astype('float32')
x_test = x_test.reshape([-1,28,28,1])
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Considering a test image for the attack

test_id = 10
Test_image = x_train[test_id].reshape([1,28,28,1])
True_label = np.argmax(y_train[test_id,:])
print (True_label)

def Load_Model():
  model = tf.keras.models.load_model('my_MNIST.h5',custom_objects =None,compile = True)
  return model
class Construct_Graph_Objfn():
  def __init__(self):
    self.const = 0 
    self.session = tf.compat.v1.Session()
    self.Confidence = 0
    self.orig_x = tf.Variable(tf.ones(dtype = tf.float32, shape=(1,28,28,1), name='orig_x')) #img for finding its corresponding adversary
    self.target_class = tf.Variable(tf.ones(dtype = tf.float32, shape=(10))) #target the adversary wants to attain
    # self.const = np.array(dtype = np.float32(5)) #Since const value keeps changing based on binary search (Hence it can neither be a tf.constant nor tf.variable )
    
    #Variable transformation based Box constraint
    self.w = tf.Variable(tf.zeros([1,28,28,1], dtype = tf.float32))
    self.winit = self.w.initializer
    self.new_x = 0.5*(tf.tanh(self.w) + tf.ones_like(self.w)) 
    
    #construting ops for the objective function
    
    self.L2_dist = tf.norm(tf.reshape(self.new_x - self.orig_x, [-1]), ord = 'euclidean')
    self.F_of_x = model(self.new_x)
    before_conf = tf.reduce_max(self.F_of_x*(1-self.target_class)) - tf.reduce_sum(self.F_of_x*self.target_class)
    target_loss = tf.maximum(before_conf, -self.Confidence)
    self.loss = tf.square(self.L2_dist) + target_loss
    self.opt = tf.train.AdamOptimizer(0.1)
    self.optim = self.opt.minimize(self.loss, var_list = [self.w])
   
  #initializes optimizer variables and function variables
  def Initialize_vars(self):
    self.session.run(self.winit)
    self.session.run(tf.variables_initializer(self.opt.variables()))   
    
  #(i)For each const, gradient descent runs for 10000 iteration
  #(ii)Then the const is updated based on whether or not adversary has achieved target label (Using binary search)
  def TwoD_Optim(self,iterations,initial_const,upper_bound,lower_bound,Test_image,t_label,bs_steps):    
    const_var = initial_const
    for bin_search in range(bs_steps):
    
      for iter in range(iterations):
        _,NEW_X, L2_DIST, PREDICTION, LOSS = self.session.run(([self.optim, self.new_x, self.L2_dist, self.F_of_x, self.loss ],
                                                            {self.const:const_var,self.orig_x: Test_image ,self.target_class: t_label}))
        if ((iter % 500)==0):
          print (iter,"   ", L2_DIST)
          
      #update the constant with binary search
      if (np.argmax(PREDICTION) == np.argmax(t_label)):
        #it is a success, decrease the c value to bring adversary closer to the original image
        print ("<------ Const:",const_var)
        upper_bound = np.minimum(upper_bound, const_var)
        const_var = (lower_bound + const_var)/2
        
      else: #failure, prioritize attaining the target label
        print ("------->Const:",const_var)
        lower_bound = np.maximum(lower_bound, const_var)
        const_var = (upper_bound + const_var)/2   
    
    
# sess = tf.Session() 
Confidence = 0
model = Load_Model() #trained Deep Net
print(model.summary())
# print(sess)
AG = Construct_Graph_Objfn() #AG: Attack_Graph

#initialize uninitialized variables  
AG.Initialize_vars()

#some other variables (non-tensors)
iterations = 10
initial_const = tf.constant(100)
upper_bound = tf.constant(50)
lower_bound = tf.constant(0)  
t_label = [1,0,0,0,0,0,0,0,0,0]
bs_steps = 20
  
  #perform gradient descent with Adam optimizer
AG.TwoD_Optim(iterations,initial_const,upper_bound,lower_bound,Test_image,t_label,bs_steps)
  
  