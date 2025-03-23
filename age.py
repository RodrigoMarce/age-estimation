import numpy as np
import matplotlib.pyplot as plt
import os

IM_WIDTH = 48
NUM_INPUT = IM_WIDTH**2
NUM_HIDDEN = 30
NUM_OUTPUT = 1

def relu (z):
    return np.maximum(0, z)

def forward_prop (x, y, W1, b1, W2, b2):
    z = W1 @ x + b1[:, np.newaxis]
    h = relu(z)
    yhat = W2 @ h + b2
    num = yhat.shape[1]
    loss = 0.5 * np.sum((y - yhat)**2) / num
    return loss, x, z, h, yhat
   
def back_prop (X, y, W1, b1, W2, b2):
    loss, x, z, h, yhat = forward_prop(X, y, W1, b1, W2, b2)
    yDiff = (yhat - y)
    gradW2 = (yDiff) @ h.T
    gradb2 = np.sum(yDiff)
    g = (yDiff.T @ W2) * ((z.T > 0).astype(float))
    g = g.T
    gradW1 = g @ x.T
    gradb1 = np.sum(g, axis=1)/ X.shape[1]
    return loss, gradW1, gradb1, gradW2, gradb2

def train (trainX, trainY, W1, b1, W2, b2, testX, testY, epsilon = 1e-5, batchSize = 32, numEpochs = 50):
    n = batchSize
    numSamples = trainX.shape[1]
    trainloss = []
    testloss = []
    trainY = trainY.reshape(1, -1)
    for epoch in range(numEpochs):
        indices = np.random.permutation(numSamples)
        trainX, trainY = trainX[:, indices], trainY[:, indices]
        if epoch < 10:
            epsilon = 1e-4
        elif epoch < 110:
            epsilon =  1e-5 
        else:
            epsilon =  1e-6
        for i in range(0, numSamples, n):
            X = trainX[:, i:i+n]
            Y = trainY[:, i:i+n]
            loss, gradW1, gradb1, gradW2, gradb2 = back_prop(X, Y, W1, b1, W2, b2)
            W1 -= epsilon * gradW1
            b1 -= epsilon * gradb1
            W2 -= epsilon * gradW2
            b2 -= epsilon * gradb2
            trainloss.append(loss)
            if(epoch > numEpochs-2):
                testloss.append(forward_prop(testX, testY, W1, b1, W2, b2)[0])
    
    if len(trainloss) >= 20:
        last20train = trainloss[-20:]
        last20test = testloss[-20:]

    if not os.path.exists("weight_images"):
        os.makedirs("weight_images")
        
    #Save W1 row by row as 48x48 images into multiple png
    for i in range(W1.shape[0]):
        image = np.reshape(W1[i,:], [ IM_WIDTH, IM_WIDTH ])
        plt.imshow(image, cmap='gray')
        plt.savefig("weight_images/weight_image_{}.png".format(i))
    
    print("Last twenty train losses: ", last20train)
    print("Last twenty test losses: ", last20test)
    return W1, b1, W2, b2

def show_weight_vectors (W1):
    # Show weight vectors in groups of 5.
    for i in range(NUM_HIDDEN//5):
        plt.imshow(np.hstack([ np.pad(np.reshape(W1[idx,:], [ IM_WIDTH, IM_WIDTH ]), 2, mode='constant') for idx in range(i*5, (i+1)*5) ]), cmap='gray'), plt.show()
    plt.show()

def loadData (which, mu = None):
    images = np.load("age_regression_X{}.npy".format(which)).reshape(-1, 48**2).T
    labels = np.load("age_regression_y{}.npy".format(which))

    if which == "tr":
        mu = np.mean(images)

    return images - mu, labels, mu

if __name__ == "__main__":
    # Load data
    if "trainX" not in globals():
        trainX, trainY, mu = loadData("tr")
        testX, testY, _ = loadData("te", mu)

    # Initialize weights to reasonable random values
    W1 = 2*(np.random.random(size=(NUM_HIDDEN, NUM_INPUT))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
    b1 = 0.01 * np.ones(NUM_HIDDEN)
    W2 = 2*(np.random.random(size=(NUM_OUTPUT, NUM_HIDDEN))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
    b2 = np.mean(trainY)

    # Train NN
    W1, b1, W2, b2 = train(trainX, trainY, W1, b1, W2, b2, testX, testY)
