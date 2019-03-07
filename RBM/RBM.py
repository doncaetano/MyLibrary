'''
Author: Rhuan Caetano
E-mail: rhuancaetano@gmail.com
Here is a implementation of the Restricted Boltzmann Machine (RBM) in TensorFlow according to
Hinton's guideline: 
[1] Hinton, Geoffrey. "A practical guide to training restricted Boltzmann machines." 
Momentum, v. 9, n. 1, p. 926, 2010.
'''

import numpy as np
import tensorflow as tf

class RBM:
    trainingData = None # numpy array for training the RBM
    nVisible = None # visible neurons number
    nHidden = None # hidden neurons number
    rbmType = None # Bernoulli-Bernoulli RBM (BBRBM) or Gaussian-Bernoulli (GBRBM)
    
    def __init__ ( self, trainingData, nHidden, rbmType='GBRBM'):

        # setting some RBM basic attributes 
        self.trainingData = trainingData 
        self.nSamples, self.nVisible = trainingData.shape
        self.nHidden = nHidden

        # setting the RBM type
        if (rbmType == 'GBRBM') or (rbmType == 'BBRBM'): 
            self.rbmType = rbmType
        else:
            print('select a valid RBM type!')
            raise Exception('RBM Type Error!!!')
    
    def train( self, maxIter=200, lr=0.001, wc=0.0002, iMom=0.5, fMom=0.9, cdIter=1, batchSize=100, verbose=True):

        # verifying the validity of the batch size parameter
        # if it's 0, the whole dataset will be used
        if batchSize < 0:
            print('invalid batch size')
            raise Exception('Error!!!')
        elif batchSize == 0:
            batchSize = self.nSamples

        # verifying the validity of the cdIter parameter
        if cdIter < 1:
            print('invalid cdIter size')
            raise Exception('Error!!!')
        
        # initializing some needed variables
        weights = tf.Variable( tf.random_normal( shape=[self.nVisible,self.nHidden], mean=0.0, stddev=0.1, dtype=tf.float32), name='weights')
        visBias = tf.Variable( tf.zeros( shape=[1, self.nVisible], dtype=tf.float32), name='visBias')
        hidBias = tf.Variable( tf.zeros (shape=[1, self.nHidden], dtype=tf.float32), name='hidBias')

        # setting some placeholder layers 
        visTensor = tf.placeholder( dtype=tf.float32, name='visibleTensor')
        hidTensor = tf.placeholder( dtype=tf.float32, name='hidTensor')  
        probHid = tf.placeholder( dtype=tf.float32, name='probHid')
        probHidTensor = tf.placeholder( dtype=tf.float32, name='probHidTensor')

        vis0Tensor = tf.placeholder( dtype=tf.float32, name='visible0Tensor')
        hid0Tensor = tf.placeholder( dtype=tf.float32, name='hid0Tensor')
        visNTensor = tf.placeholder( dtype=tf.float32, name='visibleNTensor')
        hidNTensor = tf.placeholder( dtype=tf.float32, name='hidNTensor')

        momTensor = tf.placeholder(dtype=tf.float32, name='momentum')
        attDeltaHidBias = tf.placeholder(dtype=tf.float32, name='attDeltaHidBias')
        attDeltaVisBias = tf.placeholder(dtype=tf.float32, name='attDeltaVisBias')
        attDeltaHidBias = tf.placeholder(dtype=tf.float32, name='attDeltaHidBias')

        # Initializing the delta matrices for momentum
        deltaWeights = tf.Variable( tf.zeros_like( weights, dtype=tf.float32), name='deltaWeights')       
        deltaVisBias = tf.Variable( tf.zeros_like( visBias, dtype=tf.float32), name='deltaVisBias')
        deltaHidBias = tf.Variable( tf.zeros_like( hidBias, dtype=tf.float32), name='deltaHidBias')

        # diff weight control
        prevWeights = tf.Variable(tf.zeros_like(weights, dtype=tf.float32), name='prevWeights')
        attPrevWeights = prevWeights.assign(weights)

        ### RBM training process ###
        probHid = tf.sigmoid( tf.matmul( visTensor, weights) + hidBias)
        hid = tf.cast( tf.greater(probHidTensor, tf.random_uniform( tf.shape(probHidTensor), minval=0, maxval=1, dtype=tf.float32)), dtype=tf.float32)
        
        if self.rbmType == 'GBRBM':
            vis = tf.matmul( hidTensor, weights, False, True) + visBias
        else:
            vis = tf.sigmoid( tf.matmul( hidTensor, weights, False, True) + visBias)

        dw = tf.matmul(vis0Tensor,hid0Tensor,True) - tf.matmul(visNTensor,hidNTensor,True)
        dv = tf.reduce_sum(vis0Tensor, axis=0) - tf.reduce_sum(visNTensor, axis=0)
        dh = tf.reduce_sum(hid0Tensor, axis=0) - tf.reduce_sum(hidNTensor, axis=0)

        attDeltaWeights = deltaWeights.assign((momTensor*deltaWeights) + (lr*dw/batchSize) - (wc*weights))
        attDeltaVisBias = deltaVisBias.assign((momTensor*deltaVisBias) + (lr*dv/batchSize))
        attDeltaHidBias = deltaHidBias.assign((momTensor*deltaHidBias) + (lr*dh/batchSize))

        attWeights = weights.assign_add(attDeltaWeights)
        attVisBias = visBias.assign_add(attDeltaVisBias)
        attHidBias = hidBias.assign_add(attDeltaHidBias)

        ############################
        
        with tf.Session() as sess:
            sess.run( tf.global_variables_initializer())

            for it in range(maxIter):

                # Setting the momentum
                if it < 5:
                    mom = iMom
                else:
                    mom = fMom

                permutedData = np.random.permutation(self.trainingData)
                for batch in range (0, self.nSamples, batchSize):                
                    if batch + batchSize > self.nSamples:
                        break

                    # selecting a batch
                    dataBatch = permutedData[batch : batch+batchSize] 

                    # execute the training model
                    prob0Hid = sess.run( probHid, feed_dict={visTensor:dataBatch})
                    hid0_ = hidN_ = sess.run( hid, feed_dict={probHidTensor:prob0Hid})
                    for _ in range(cdIter):
                        visN_ = sess.run( vis, feed_dict={hidTensor:hidN_})
                        probNHid = sess.run( probHid, feed_dict={visTensor:visN_})
                        hidN_ = sess.run( hid, feed_dict={probHidTensor:probNHid})

                    # updating the weights and bias
                    sess.run([attWeights, attVisBias, attHidBias], feed_dict={ vis0Tensor:dataBatch, hid0Tensor:prob0Hid, visNTensor:visN_, hidNTensor:probNHid,  momTensor:mom})  

                # Reconstruction error:
                error = self.rmse( dataBatch, visN_)
                diff = self.rmse( sess.run(weights),sess.run(prevWeights))
                sess.run( attPrevWeights)
                if verbose and (it % (maxIter//10)) == 0:
                    print ('Reconstruction error: ', error, 'Iter: ', it, '  Diff: ', diff)   

                if it > 300 and diff <= 10e-5:
                    break
            
            # save the trained weights    
            self.weights = sess.run(weights)
            self.visBias = sess.run(visBias)
            self.hidBias = sess.run(hidBias)
        
    def rmse ( self, x, y):
	    return np.sum( np.sqrt( np.sum( np.power(x-y,2),axis=1)/x.shape[1]))/x.shape[0]

    def getReconstruction ( self, data):
        vis = tf.convert_to_tensor( data, dtype=tf.float32, name='visRecs')
        probHid = tf.sigmoid( tf.matmul( vis, self.weights) + self.hidBias)
        hid = tf.cast( tf.greater( probHid, tf.random_uniform( tf.shape(probHid), minval=0, maxval=1, dtype=tf.float32)), dtype=tf.float32)
        
        if self.rbmType == 'GBRBM':            
            recons = tf.matmul( hid,self.weights, False, True)+self.visBias
        else:
            recons = tf.sigmoid( tf.matmul(hid, self.weights, False, True) + self.visBias)
            
        with tf.Session() as sess:
            rec = sess.run(recons)
        return rec
   
    def saveWeights( self, fileName='RBM'):
        np.savetxt( fileName+'Weights.csv', self.weights)
        np.savetxt( fileName+'VisBias.csv', self.visBias)
        np.savetxt( fileName+'HidBias.csv', self.hidBias)   
        

    


