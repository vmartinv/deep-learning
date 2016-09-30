import numpy as np
import theano
import theano.tensor as T
rng = np.random

N = 400                                   # training sample size
feats = 784                               # number of input variables

# generate a dataset: D = (input_values, target_class)
D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))
training_steps = 10

# Declare Theano symbolic variables
x = T.dmatrix("x")
y = T.dvector("y")

# initialize the weight vector w randomly
#
# this and the following bias variable b
# are shared so they keep their values
# between training iterations (updates)
capa_interna = 10
w1 = theano.shared(rng.randn(feats, capa_interna) * sqrt(2.0/feats), name="w1")
w = theano.shared(rng.randn(capa_interna), name="w")

# initialize the bias term
b1 = theano.shared(np.zeros(capa_interna), name="b1")
b = theano.shared(0., name="b")

# Construct Theano expression graph
layer_output = T.nnet.relu(T.dot(x, w1) - b1)   
p_1 = 1 / (1 + T.exp(-T.dot(layer_output, w) - b))   # Probability that target = 1
prediction = p_1 > 0.5                    # The prediction thresholded
xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1) # Cross-entropy loss function
cost = xent.mean() + 0.01 * (w ** 2).sum() + 0.01 * (w1 ** 2).sum()# The cost to minimize
gw, gw1, gb, gb1 = T.grad(cost, [w, w1, b, b1])        # Compute the gradient of the cost
                                              # w.r.t weight vector w and
                                              # bias term b
                                              # (we shall return to this in a
                                              # following section of this tutorial)

# Compile
train = theano.function(
          inputs=[x,y],
          outputs=[prediction, xent],
          updates=((w, w - 0.1 * gw), (w1, w1 - 0.1 * gw1), (b, b - 0.1 * gb), (b1, b1 - 0.1 * gb1)))
predict = theano.function(inputs=[x], outputs=prediction)

# Train
for i in range(training_steps):
    pred, err = train(D[0], D[1])

match = ((y - prediction)**2).mean()
count_matches = theano.function(inputs=[x, y], outputs=match)
print("error: %f" % (count_matches(D[0], D[1])))
