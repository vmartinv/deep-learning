import numpy as np
import theano
import theano.tensor as T
import funcstp1
rng = np.random

def make_vec(img):
    img = funcstp1.normalize(img)
    img = funcstp1.resize_image(img,28)
    img = funcstp1.to_gray_scale(img)
    return img.flatten()

# generate a dataset: D = (input_values, target_class)
print("Cargando imagenes...")
np.random.seed(2342345)
dataset = []
for img in funcstp1.get_images("airplanes", 60):
    dataset.append((make_vec(img), 0))
for img in funcstp1.get_images("motorbikes", 60):
    dataset.append((make_vec(img), 1))
    
np.random.shuffle(dataset)
D = list(zip(*dataset))
N = len(D[0])                         # training sample size
print("%d imagenes cargadas." % (N))
feats = len(D[0][0])                  # number of input variables
       
training_steps = 100

print("Creando grafo de computación...")
# Declare Theano symbolic variables
x = T.dmatrix("x")
y = T.dvector("y")

# initialize the weight vector w randomly
#
# this and the following bias variable b
# are shared so they keep their values
# between training iterations (updates)
w = theano.shared(rng.randn(feats), name="w")

# initialize the bias term
b = theano.shared(0., name="b")

# Construct Theano expression graph
p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))   # Probability that target = 1
prediction = p_1 > 0.5                    # The prediction thresholded
xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1) # Cross-entropy loss function
cost = xent.mean() + 0.01 * (w ** 2).sum()# The cost to minimize
gw, gb = T.grad(cost, [w, b])             # Compute the gradient of the cost
                                          # w.r.t weight vector w and
                                          # bias term b
                                          # (we shall return to this in a
                                          # following section of this tutorial)

N_valid = 20
validacion = (D[0][:N_valid], D[1][:N_valid])
D = (D[0][N_valid:], D[1][N_valid:])
# Compile
print("Entrenando red neuronal con %d imagenes..." % (len(D[0])))
train = theano.function(
          inputs=[x,y],
          outputs=[prediction, xent],
          updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))
predict = theano.function(inputs=[x], outputs=prediction)

# Train
for i in range(training_steps):
    pred, err = train(D[0], D[1])

match = ((y - prediction)**2).mean()
count_matches = theano.function(inputs=[x, y], outputs=match)
print("error: %f" % (count_matches(D[0], D[1])))
print("error en validacion: %f" % (count_matches(validacion[0], validacion[1])))

images = funcstp1.get_images("airplanes") + funcstp1.get_images("motorbikes")
np.random.shuffle(images)
for img in images:
    if predict([make_vec(img)]) == [0]:
        title = "Es un avión!"
    else:
        title = "Es una moto!"
    funcstp1.show(img, title)
    
