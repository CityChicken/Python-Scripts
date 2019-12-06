# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 22:54:19 2018

@author: Jacob
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
import simplegeneric, gzip, os, timeit, sys
import theano
import theano.tensor as T
from PIL import Image

masterpath = "F:/MachineLearningData/Bees_Dataset/"
imagepath = "bee_imgs/"
beemasterkey = "bee_data.csv"
beeimg = "001_043.png"


image = mpimg.imread(masterpath + imagepath + beeimg)
imgplot = plt.imshow(image)
plt.show()

beemasterkey_df = pd.read_csv(masterpath + beemasterkey)
print(beemasterkey_df.head(5))

#print(beemasterkey_df['file'])
k = len(beemasterkey_df.index)
dimensions = np.zeros((k,3))
desired_size = 90
for i in range (0,k):
    image = Image.open(masterpath + imagepath 
                         + beemasterkey_df.iloc[i]['file'])
    #print(image.shape)
    dimensions[i,0:2] = image.size
    old_size = ((dimensions[i,0],dimensions[i,1]))
    #print(old_size)
    if i == 15:
        imgplot = plt.imshow(image)
        plt.show()
    
        ratio = float(desired_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])
        print(new_size)
        image = image.resize(new_size, Image.ANTIALIAS)
        new_im = Image.new("RGB", (desired_size, desired_size), (255,255,255))
        new_im.paste(image, ((desired_size-new_size[0])//2,
                        (desired_size-new_size[1])//2))
        
        imgplot = plt.imshow(new_im)
        plt.show()

print( max(dimensions[:,0]) , max(dimensions[:,1]), min(dimensions[:,0]) , min(dimensions[:,1]))

#**************************************
#f = gzip.open('mnist.pkl.gz', 'rb') 
#train_set, valid_set, test_set = simplegeneric.pickle.load(f) 
#f.close()




def load_data(dataset):
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        new_path = os.path.join(
                os.path.split(__file__)[0],
                "..",
                "data",
                dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz': 
            dataset = new_path
    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz': 
        import urllib 
        origin = ( 
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz' 
        ) 
        print('Downloading data from %s'  % origin)
        urllib.urlretrieve(origin, dataset)
    print('... loading data')
    f = gzip.open(dataset, 'rb') 
    train_set, valid_set, test_set = simplegeneric.pickle.load(f) 
    f.close() 
    
    def shared_dataset(data_xy): 
        """ Function that loads the dataset into shared variables
        The reason we store our dataset in shared variables is to allow Theano to 
        copy it into the GPU memory (when code is run on GPU). Since copying data 
        into the GPU is slow, copying a minibatch everytime is needed (the default 
        behaviour if the data is not in a shared variable) would lead to a large 
        decrease in performance. """ 
        data_x, data_y = data_xy 
        shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX)) 
        shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX)) # When storing data on the GPU it has to be stored as floats # therefore we will store the labels as ‘‘floatX‘‘ as well # (‘‘shared_y‘‘ does exactly that). But during our computations # we need them as ints (we use labels as index, and if they are # floats it doesn’t make sense) therefore instead of returning # ‘‘shared_y‘‘ we will have to cast it to int. This little hack # lets us get around this issue
        return shared_x, T.cast(shared_y, 'int32')
    test_set_x, test_set_y = shared_dataset(test_set) 
    valid_set_x, valid_set_y = shared_dataset(valid_set) 
    train_set_x, train_set_y = shared_dataset(train_set)
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), 
            (test_set_x, test_set_y)] 
    return rval


class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
        self.W = theano.shared(
            value=np.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        self.b = theano.shared(
            value=np.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]
        self.input = input
    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arrange(y.shape[0]), y])
    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.ypred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            return NotImplementedError()



class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation = T.tanh):
        self.input = input
        if W is None:
            W_values = np.asarray(
            rng.uniform(
            low= np.sqrt(6. / (n_in + n_out)),
            high=np.sqrt(6. / (n_in + n_out)),
            size=(n_in, n_out)
            ),
            dtype=theano.config.floatX
            )
            if activation == T.nnet.sigmoid:
                W_values *= 4
              
            W = theano.shared(value=W_values, name='W', borrow=True)
        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
        
        self.W = W
        self.b = b
        
        lin_output = T.dot(input, self.W) + self.b
        self.output = (
                lin_output if activation is None
                else activation(lin_output)
        )
        
        self.params [self.W, self.b]
            

class MLP(object):
    def __init__(self, rng, input, n_in, n_hidden, n_out):
        self.hiddenLayer = HiddenLayer(
                rng=rng,
                input=input,
                n_in=n_in,
                n_out=n_hidden,
                activation=T.tanh
        )
        self.logRegressionLayer = LogisticRegression(
                input=self.hiddenLayer.output,
                n_in=n_hidden,
                n_out=n_out
        )
        self.L1 = (
                abs(self.hiddenLayer.W).sum()
                + abs(self.logRegressionLayer.W).sum()
        )
        self.L2_sqr = ( 
                (self.hiddenLayer.W ** 2).sum() 
                + (self.logRegressionLayer.W ** 2).sum() 
        )
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood 
        ) 
        self.errors = self.logRegressionLayer.errors
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params
        self.input = input
                      
def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000, 
             dataset='mnist.pkl.gz', batch_size=20, n_hidden=500):
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0] 
    valid_set_x, valid_set_y = datasets[1] 
    test_set_x, test_set_y = datasets[2]
    # compute number of minibatches for training, validation and testing 
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size 
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size 
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size
    ###################### 
    # BUILD ACTUAL MODEL # 
    ######################
    print('... building the model')
    # allocate symbolic variables for the data 
    index = T.lscalar() # index to a [mini]batch 
    x = T.matrix('x') # the data is presented as rasterized images 
    y = T.ivector('y') # the labels are presented as 1D vector of 
                        # [int] labels
    rng = np.random.RandomState(1234)
    # construct the MLP class 
    classifier = MLP( rng=rng,
        input=x,
        n_in=28 * 28,
        n_hidden=n_hidden,
        n_out=10 )
    cost = ( classifier.negative_log_likelihood(y) + 
            L1_reg * classifier.L1 + L2_reg * classifier.L2_sqr )
    test_model = theano.function( 
            inputs=[index],
            outputs=classifier.errors(y),
            givens={ 
                x: test_set_x[index * batch_size:(index + 1) * batch_size], 
                y: test_set_y[index * batch_size:(index + 1) * batch_size] }
    )
    validate_model = theano.function(
            inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                y: valid_set_y[index * batch_size:(index + 1) * batch_size] 
                }
    )

    gparams = [T.grad(cost, param) for param in classifier.params]
    updates = [ (param, param - learning_rate * gparam) 
    for param, gparam in zip(classifier.params, gparams) ]
    train_model = theano.function( 
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
                }
            )
    print('... training')
    patience = 10000
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience / 2)
    best_validation_loss = np.inf 
    best_iter = 0 
    test_score = 0. 
    start_time = timeit.default_timer()
    epoch = 0
    done_looping = False
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1 
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index
            
            if (iter + 1) % validation_frequency == 0:
                validation_losses = [validate_model(i) for i 
                                     in range(n_valid_batches)] 
                this_validation_loss = np.mean(validation_losses)
                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' % 
                     (
                         epoch,
                         minibatch_index + 1,
                         n_train_batches,
                         this_validation_loss * 100. 
                     ) 
                )
            if this_validation_loss < best_validation_loss:
                if (
                    this_validation_loss < best_validation_loss * 
                    improvement_threshold
                    ):
                    patience = max(patience, iter * patience_increase)
                best_validation_loss = this_validation_loss 
                best_iter = iter
                test_losses = [test_model(i) for i in range(n_test_batches)] 
                test_score = np.mean(test_losses)
                print((' epoch %i, minibatch %i/%i, test error of ' 
                       'best model %f %%') % 
                        (epoch, minibatch_index + 1, n_train_batches, 
                        test_score * 100.))
            if patience <= iter: 
                done_looping = True 
                break
    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% ' 
           'obtained at iteration %i, with test performance %f %%') % 
            (best_validation_loss * 100., best_iter + 1, test_score * 100.)) 
    print('The code for file ' 
                          + os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.) , 
                          file=sys.stderr)

#if __name__ == '__main__': 
 #   test_mlp()




















