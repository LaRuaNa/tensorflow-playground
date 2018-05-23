import matplotlib.pyplot as plt
import numpy as np
import random

def generate_data_or():
    X = [[0, 0], [1, 1], [1, 0], [0, 1]]
    y = [0, 1, 1, 1]
    return X, y

# Perceptron class for OR
class Perceptron():
    def __init__(self, X, y):
        self.w1 = -0.5
        self.w2 = -0.3
        self.learning_rate = 1.0
        self.epochs = 20
        self.X = X
        self.y = y

    def activation(self, Xp):
        y = Xp[0] * self.w1 + Xp[1] * self.w2
        if y > 0:
            return 1
        else:
            return 0

    def update_weights(self, Xp, yp, y_hat):
        delta_w1 = self.learning_rate * (yp - y_hat) * Xp[0]
        delta_w2 = self.learning_rate * (yp - y_hat) * Xp[1] 

        self.w1 = self.w1 + delta_w1
        self.w2 = self.w2 + delta_w2

    def training(self):
        error = 0.0
        for epoch in range(self.epochs):
            choice = random.randint(0, len(self.X)-1)
            Xp = self.X[choice]
            yp = self.y[choice]
            y_hat = self.activation(Xp)
            if y_hat != yp: 
                error += abs(yp - y_hat)
                self.update_weights(Xp, yp, y_hat)
            if epoch == self.epochs - 1:
                for indx in range(len(self.X)):
                    if y[indx] == 1:
                        plt.scatter(self.X[indx][0], self.X[indx][1], color ='red')
                    else:
                        plt.scatter(self.X[indx][0], self.X[indx][1], color ='blue')
                plt.plot([self.w2, -self.w2], [-self.w1, self.w1], '--k')
                print('New weights: ', self.w1, " ", self.w2)
                plt.show()
        print('Sum of Error: ', error)

    def testing(self):
        false_results = 0
        for indx in range(len(self.X)):
            Xp = self.X[indx]
            yp = self.y[indx]
            y_hat = self.activation(Xp)
            print('Predicted: ', y_hat, ' Label: ', yp)
            print('on:', Xp[0], Xp[1])
            if y_hat != yp:
                false_results += 1
        print('Accuracy: ', 100 * (len(self.X) - false_results) / len(self.X), '%')
            
 
X, y = generate_data_or() 
p = Perceptron(X, y) 
p.training()
p.testing()