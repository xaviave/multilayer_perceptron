class NeuralNetwork(object):
    def __init__(self, architecture):
        #architecture - numpy array with ith element representing the number of neurons in the ith layer.
        
        #Initialize the network architecture
        self.L = architecture.size - 1 #L corresponds to the last layer of the network.
        self.n = architecture #n stores the number of neurons in each layer
        #input_size is the number of neurons in the first layer i.e. n[0]
        #output_size is the number of neurons in the last layer i.e. n[L]
        
        #Parameters will store the network parameters, i.e. the weights and biases
        self.parameters = {}
        
        #Initialize the network weights and biases:
        for i in range (1, self.L + 1): 
            #Initialize weights to small random values
            self.parameters['W' + str(i)] = np.random.randn(self.n[i], self.n[i - 1]) * 0.01
            
            #Initialize rest of the parameters to 1
            self.parameters['b' + str(i)] = np.ones((self.n[i], 1))
            self.parameters['z' + str(i)] = np.ones((self.n[i], 1))
            self.parameters['a' + str(i)] = np.ones((self.n[i], 1))
        
        #As we started the loop from 1, we haven't initialized a[0]:
        self.parameters['a0'] = np.ones((self.n[i], 1))
        
        #Initialize the cost:
        self.parameters['C'] = 1
        
        #Create a dictionary for storing the derivatives:
        self.derivatives = {}
                    
    def forward_propagate(self, X):
        #Note that X here, is just one training example
        self.parameters['a0'] = X
        
        #Calculate the activations for every layer l
        for l in range(1, self.L + 1):
            self.parameters['z' + str(l)] = np.add(np.dot(self.parameters['W' + str(l)], self.parameters['a' + str(l - 1)]), self.parameters['b' + str(l)])
            self.parameters['a' + str(l)] = sigmoid(self.parameters['z' + str(l)])
        
    def compute_cost(self, y):
        self.parameters['C'] = -(y*np.log(self.parameters['a' + str(self.L)]) + (1-y)*np.log( 1 - self.parameters['a' + str(self.L)]))
    
    def compute_derivatives(self, y):
        #Partial derivatives of the cost function with respect to z[L], W[L] and b[L]:        
        #dzL
        self.derivatives['dz' + str(self.L)] = self.parameters['a' + str(self.L)] - y
        #dWL
        self.derivatives['dW' + str(self.L)] = np.dot(self.derivatives
        ['dz' + str(self.L)], np.transpose(self.parameters['a' + str(self.L - 1)]))
        #dbL
        self.derivatives['db' + str(self.L)] = self.derivatives['dz' + str(self.L)]

        #Partial derivatives of the cost function with respect to z[l], W[l] and b[l]
        for l in range(self.L-1, 0, -1):
            self.derivatives['dz' + str(l)] = np.dot(np.transpose(self.parameters['W' + str(l + 1)]), self.derivatives['dz' + str(l + 1)])*sigmoid_prime(self.parameters['z' + str(l)])
            self.derivatives['dW' + str(l)] = np.dot(self.derivatives['dz' + str(l)], np.transpose(self.parameters['a' + str(l - 1)]))
            self.derivatives['db' + str(l)] = self.derivatives['dz' + str(l)]
            
    def update_parameters(self, alpha):
        for l in range(1, self.L+1):
            self.parameters['W' + str(l)] -= alpha*self.derivatives['dW' + str(l)]
            self.parameters['b' + str(l)] -= alpha*self.derivatives['db' + str(l)]
        
    def predict(self, x):
        self.forward_propagate(x)
        return self.parameters['a' + str(self.L)]
        
    def fit(self, X, Y, num_iter, alpha = 0.01):
        for iter in range(0, num_iter):
            c = 0 #Stores the cost
            n_c = 0 #Stores the number of correct predictions
            
            for i in range(0, X.shape[0]):
              x = X[i].reshape((X[i].size, 1))
              y = Y[i]

              self.forward_propagate(x)
              self.compute_cost(y)
              self.compute_derivatives(y)
              self.update_parameters(alpha)

              c += self.parameters['C'] 

              y_pred = self.predict(x)
              #y_pred is the probability, so to convert it into a class value:
              y_pred = (y_pred > 0.5) 

              if y_pred == y:
                  n_c += 1
            
            c = c/X.shape[0]
            print('Iteration: ', iter)
            print("Cost: ", c)
            print("Accuracy:", (n_c/X.shape[0])*100)

def calcul_der(self, y):
        dz = [self.get_output_delta(self.activations[-1], self._to_one_hot(int(y), 2))]
        dz[-1] = np.reshape(dz[-1], (-1, 1))
        self.activations[-2] = np.reshape(self.activations[-2], (-1, 1))
        dw = [self.get_weight_gradient2(dz[-1], self.activations[-2].T)]
        db = [dz[-1]]

        for i in range(len(self.layers) - 2, -1, -1):
            layer = self.layers[i]
            next_layer = self.layers[i + 1]
            derivate = layer.activation_prime(self.weighted_sums[i])
            dz.append(np.dot(next_layer.weights.T, dz[-1]) * derivate)
            dw.append(np.dot(dz[-1], self.activations[i].))
            #print(dz[-1].shape, self.activations[i].shape) ; quit()
            db.append(dz[-1])
            quit()

        quit()