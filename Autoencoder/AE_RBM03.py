'''
Author: Rhuan Caetano
E-mail: rhuancaetano@gmail.com
Here is a implementation of an autoencoder with 3 encoder and decoder layers, based on the Restricted Boltzmann Machine (RBM) in TensorFlow according to Hinton's: 
[1] G. E. Hinton* and R. R. Salakhutdinov. "Reducing the Dimensionality of Data with Neural Networks" 
Science  28 Jul 2006: Vol. 313, Issue 5786, pp. 504-507.
'''

from RBM import RBM
import numpy as np
import tensorflow as tf
from numpy import genfromtxt

class RBM_Weights:
    def __init__(self, weights, visBias, hidBias):
        self.weights = weights
        self.visBias = visBias
        self.hidBias = hidBias
    def getWeights(self):
        return self.weights
    def getVisBias(self):
        return self.visBias
    def getHidBias(self):
        return self.hidBias    

class AE_RBM:
    nEncoderLayers = 3 # number of encoder layers of the autoencoder
    sEncoderLayers = None # size of the encoder layers 

    def __init__ (self, sEncoderLayers):
        if(self.nEncoderLayers != sEncoderLayers.shape[0]):
            print('Invalid number of size layers')
            raise Exception('Autoencoder constructor ERROR !!!')
        self.sEncoderLayers = sEncoderLayers
    
    def train(self, trainingData):
        rbmList = [] # list RBM's weights
        tempData = trainingData
        # start RBM's training and get the respective weights
        for n in range(self.nEncoderLayers):
            if(n==0 or n==(self.nEncoderLayers-1)):
                rbm = RBM(tempData, self.sEncoderLayers[n], rbmType='GBRBM')
            else:
                rbm = RBM(tempData, self.sEncoderLayers[n], rbmType='BBRBM')
            
            print('Start %d RBM training' % (n+1) )
            rbm.train(batchSize=100)

            [weights, visBias, hidBias] = rbm.getWeights()
            rbmList.append(RBM_Weights(weights, visBias, hidBias))

            data = tf.convert_to_tensor( tempData, dtype=tf.float32, name='data')
            probHid = tf.sigmoid( tf.matmul( data, weights) + hidBias)
            hid = tf.cast( tf.greater( probHid, tf.random_uniform( tf.shape(probHid), minval=0, maxval=1, dtype=tf.float32)), dtype=tf.float32)

            with tf.Session() as sess:
                if((self.nEncoderLayers-1) == n):
                    tempData = sess.run(probHid)
                else:
                    tempData = sess.run(hid)

        # start the fine tuning process
        return self.fineTuning( rbmList, trainingData)


    def fineTuning(self, rbmList, trainingData):

            # create the weight variables
            layer_01_Weights = tf.Variable(rbmList[0].getWeights(), dtype=tf.float32, name='layer_01_Weights')
            layer_01_VisBias = tf.Variable(rbmList[0].getVisBias(), dtype=tf.float32, name='layer_01_VisBias')
            layer_01_HidBias = tf.Variable(rbmList[0].getHidBias(), dtype=tf.float32, name='layer_01_HidBias')
            layer_02_Weights = tf.Variable(rbmList[1].getWeights(), dtype=tf.float32, name='layer_02_Weights')
            layer_02_VisBias = tf.Variable(rbmList[1].getVisBias(), dtype=tf.float32, name='layer_02_VisBias')
            layer_02_HidBias = tf.Variable(rbmList[1].getHidBias(), dtype=tf.float32, name='layer_02_HidBias')
            layer_03_Weights = tf.Variable(rbmList[2].getWeights(), dtype=tf.float32, name='layer_03_Weights')
            layer_03_VisBias = tf.Variable(rbmList[2].getVisBias(), dtype=tf.float32, name='layer_03_VisBias')
            layer_03_HidBias = tf.Variable(rbmList[2].getHidBias(), dtype=tf.float32, name='layer_03_HidBias')

            # create some placeholders for the model
            probHid_01 = tf.placeholder(dtype=tf.float32, name='probHid_01')
            hid_01 = tf.placeholder(dtype=tf.float32, name='hid_01')
            probHid_02 = tf.placeholder(dtype=tf.float32, name='probHid_02')
            probHid_03 = tf.placeholder(dtype=tf.float32, name='probHid_03')
            hid_03 = tf.placeholder(dtype=tf.float32, name='hid_03')
            recons_03 = tf.placeholder(dtype=tf.float32, name='recons_03')
            recons_02 = tf.placeholder(dtype=tf.float32, name='recons_02')
            recons_01 = tf.placeholder(dtype=tf.float32, name='recons_01')

            data = tf.convert_to_tensor( trainingData, dtype=tf.float32, name='visRecs_01')

            # W1_Encoder
            probHid_01 = tf.sigmoid( tf.matmul( data, layer_01_Weights) + layer_01_HidBias)
            hid_01 = tf.cast( tf.greater( probHid_01, tf.random_uniform( tf.shape(probHid_01), minval=0, maxval=1, dtype=tf.float32)), dtype=tf.float32)
            
            # W2_Encoder
            probHid_02 = tf.sigmoid( tf.matmul( hid_01, layer_02_Weights) + layer_02_HidBias)
            
            # W3_Encoder
            probHid_03 = tf.sigmoid( tf.matmul( probHid_02, layer_03_Weights) + layer_03_HidBias)
            hid_03 = tf.cast( tf.greater( probHid_03, tf.random_uniform( tf.shape(probHid_03), minval=0, maxval=1, dtype=tf.float32)), dtype=tf.float32)
            
            # W3_Decoder
            recons_03 = tf.sigmoid( tf.matmul(hid_03, layer_03_Weights, False, True) + layer_03_VisBias)

            # W2_Decoder
            recons_02 = tf.sigmoid( tf.matmul(recons_03, layer_02_Weights, False, True) + layer_02_VisBias)

            # W1_Decoder
            recons_01 = tf.matmul( recons_02, layer_01_Weights, False, True) + layer_01_VisBias

            # cost function
            error = tf.losses.mean_squared_error(trainingData, recons_01)
            
            # some tensorflow optimizers
            #train_op = tf.train.AdagradOptimizer(0.1).minimize(error)
            train_op = tf.train.AdadeltaOptimizer(1).minimize(error)
            #train_op = tf.train.GradientDescentOptimizer(0.1).minimize(error)

            errorArray = np.array([])
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                for i in range(1000):
                    _, auxError = sess.run([train_op, error])
                    errorArray = np.append(errorArray, auxError)
                    print('IT: %lf    Error: %lf'%(i,auxError))
                    if(i > 200 and auxError < 0.1):
                        break
                self.layer_01_Weights, self.layer_01_VisBias, self.layer_01_HidBias = sess.run( [layer_01_Weights, layer_01_VisBias, layer_01_HidBias])
                self.layer_02_Weights, self.layer_02_VisBias, self.layer_02_HidBias = sess.run( [layer_02_Weights, layer_02_VisBias, layer_02_HidBias])
                self.layer_03_Weights, self.layer_03_VisBias, self.layer_03_HidBias = sess.run( [layer_03_Weights, layer_03_VisBias, layer_03_HidBias])

            self.saveTrainedWeights()
            return errorArray

    def recon(self, trainingData):
        
        # create some placeholders for the model
        probHid_01 = tf.placeholder(dtype=tf.float32, name='probHid_01')
        hid_01 = tf.placeholder(dtype=tf.float32, name='hid_01')
        probHid_02 = tf.placeholder(dtype=tf.float32, name='probHid_02')
        probHid_03 = tf.placeholder(dtype=tf.float32, name='probHid_03')
        hid_03 = tf.placeholder(dtype=tf.float32, name='hid_03')

        # create the weight variables
        layer_01_Weights = tf.Variable(self.layer_01_Weights, dtype=tf.float32, name='layer_01_Weights')
        layer_01_VisBias = tf.Variable(self.layer_01_VisBias, dtype=tf.float32, name='layer_01_VisBias')
        layer_01_HidBias = tf.Variable(self.layer_01_HidBias, dtype=tf.float32, name='layer_01_HidBias')
        layer_02_Weights = tf.Variable(self.layer_02_Weights, dtype=tf.float32, name='layer_02_Weights')
        layer_02_VisBias = tf.Variable(self.layer_02_VisBias, dtype=tf.float32, name='layer_02_VisBias')
        layer_02_HidBias = tf.Variable(self.layer_02_HidBias, dtype=tf.float32, name='layer_02_HidBias')
        layer_03_Weights = tf.Variable(self.layer_03_Weights, dtype=tf.float32, name='layer_03_Weights')
        layer_03_VisBias = tf.Variable(self.layer_03_VisBias, dtype=tf.float32, name='layer_03_VisBias')
        layer_03_HidBias = tf.Variable(self.layer_03_HidBias, dtype=tf.float32, name='layer_03_HidBias')

        data = tf.convert_to_tensor( trainingData, dtype=tf.float32, name='visRecs_01')

        # W1_Encoder
        probHid_01 = tf.sigmoid( tf.matmul( data, layer_01_Weights) + layer_01_HidBias)
        hid_01 = tf.cast( tf.greater( probHid_01, tf.random_uniform( tf.shape(probHid_01), minval=0, maxval=1, dtype=tf.float32)), dtype=tf.float32)
        
        # W2_Encoder
        probHid_02 = tf.sigmoid( tf.matmul( hid_01, layer_02_Weights) + layer_02_HidBias)
        
        # W3_Encoder
        probHid_03 = tf.sigmoid( tf.matmul( probHid_02, layer_03_Weights) + layer_03_HidBias)
        hid_03 = tf.cast( tf.greater( probHid_03, tf.random_uniform( tf.shape(probHid_03), minval=0, maxval=1, dtype=tf.float32)), dtype=tf.float32)
        
        # W3_Decoder
        recons_03 = tf.sigmoid( tf.matmul(hid_03, layer_03_Weights, False, True) + layer_03_VisBias)

        # W2_Decoder
        recons_02 = tf.sigmoid( tf.matmul(recons_03, layer_02_Weights, False, True) + layer_02_VisBias)

        # W1_Decoder
        recons_01 = tf.matmul( recons_02, layer_01_Weights, False, True) + layer_01_VisBias

        with tf.Session() as sess:
            # run model and return the reconstruction matrix
            sess.run(tf.global_variables_initializer())
            recons = sess.run(recons_01)
        return recons

    def loadTrainedWeights(self):
        self.layer_01_Weights = genfromtxt('layer_01_Weights.csv', delimiter=' ')
        self.layer_01_VisBias = genfromtxt('layer_01_VisBias.csv', delimiter=' ')
        self.layer_01_HidBias = genfromtxt('layer_01_HidBias.csv', delimiter=' ') 
        self.layer_02_Weights = genfromtxt('layer_02_Weights.csv', delimiter=' ')
        self.layer_02_VisBias = genfromtxt('layer_02_VisBias.csv', delimiter=' ')
        self.layer_02_HidBias = genfromtxt('layer_02_HidBias.csv', delimiter=' ')
        self.layer_03_Weights = genfromtxt('layer_03_Weights.csv', delimiter=' ')
        self.layer_03_VisBias = genfromtxt('layer_03_VisBias.csv', delimiter=' ')
        self.layer_03_HidBias = genfromtxt('layer_03_HidBias.csv', delimiter=' ') 
    
    def saveTrainedWeights(self):
        np.savetxt( 'layer_01_Weights.csv', self.layer_01_Weights)
        np.savetxt( 'layer_01_VisBias.csv', self.layer_01_VisBias)
        np.savetxt( 'layer_01_HidBias.csv', self.layer_01_HidBias)
        np.savetxt( 'layer_02_Weights.csv', self.layer_02_Weights)
        np.savetxt( 'layer_02_VisBias.csv', self.layer_02_VisBias)
        np.savetxt( 'layer_02_HidBias.csv', self.layer_02_HidBias)
        np.savetxt( 'layer_03_Weights.csv', self.layer_03_Weights)
        np.savetxt( 'layer_03_VisBias.csv', self.layer_03_VisBias)
        np.savetxt( 'layer_03_HidBias.csv', self.layer_03_HidBias)

    def filter(self, trainingData):
        # create some placeholders for the model
        probHid_01 = tf.placeholder(dtype=tf.float32, name='probHid_01')
        hid_01 = tf.placeholder(dtype=tf.float32, name='hid_01')
        probHid_02 = tf.placeholder(dtype=tf.float32, name='probHid_02')
        probHid_03 = tf.placeholder(dtype=tf.float32, name='probHid_03')

        # create the weight variables
        layer_01_Weights = tf.Variable(self.layer_01_Weights, dtype=tf.float32, name='layer_01_Weights')
        layer_01_HidBias = tf.Variable(self.layer_01_HidBias, dtype=tf.float32, name='layer_01_HidBias')
        layer_02_Weights = tf.Variable(self.layer_02_Weights, dtype=tf.float32, name='layer_02_Weights')
        layer_02_HidBias = tf.Variable(self.layer_02_HidBias, dtype=tf.float32, name='layer_02_HidBias')
        layer_03_Weights = tf.Variable(self.layer_03_Weights, dtype=tf.float32, name='layer_03_Weights')
        layer_03_HidBias = tf.Variable(self.layer_03_HidBias, dtype=tf.float32, name='layer_03_HidBias')

        data = tf.convert_to_tensor( trainingData, dtype=tf.float32, name='trainingData')

        # W1_Encoder
        probHid_01 = tf.sigmoid( tf.matmul( data, layer_01_Weights) + layer_01_HidBias)
        hid_01 = tf.cast( tf.greater( probHid_01, tf.random_uniform( tf.shape(probHid_01), minval=0, maxval=1, dtype=tf.float32)), dtype=tf.float32)
        
        # W2_Encoder
        probHid_02 = tf.sigmoid( tf.matmul( hid_01, layer_02_Weights) + layer_02_HidBias)
        
        # W3_Encoder
        probHid_03 = tf.sigmoid( tf.matmul( probHid_02, layer_03_Weights) + layer_03_HidBias)

        with tf.Session() as sess:
            # run model and return the reconstruction matrix
            sess.run(tf.global_variables_initializer())
            output = sess.run(probHid_03)
        return output