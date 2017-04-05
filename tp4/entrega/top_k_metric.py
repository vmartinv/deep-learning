import theano
from theano import tensor as T
import keras.backend as K

def in_top_k(predictions, targets, k):
    '''Says whether the `targets` are in the top `k` `predictions`
    # Arguments
        predictions: A tensor of shape batch_size x classess and type float32.
        targets: A tensor of shape batch_size and type int32 or int64.
        k: An int, number of top elements to consider.
    # Returns
        A tensor of shape batch_size and type int. output_i is 1 if
        targets_i is within top-k values of predictions_i
    '''
    predictions_top_k = T.argsort(predictions)[:, -k:]
    result, _ = theano.map(lambda prediction, target: K.any(K.equal(prediction, target)), sequences=[predictions_top_k, targets])
    return result

def top_k_categorical_accuracy(y_true, y_pred, k=5):
    '''Calculates the top-k categorical accuracy rate, i.e. success when the
    target class is within the top-k predictions provided
    '''
    return K.mean(in_top_k(y_pred, K.argmax(y_true, axis=-1), k))

def top3(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred,3)

#~ Ejemplo de como usar esto:
#~ from top_k_metric import top3
#~ model.compile(loss='categorical_crossentropy',
              #~ optimizer='adadelta',
              #~ metrics=['accuracy', top3 ])

def top3(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred,3)
