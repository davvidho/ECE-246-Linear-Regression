import numpy as np

class Regression(object):
    def __init__(self, m=1, reg_param=0):
        """"
        Inputs:
          - m Polynomial degree
          - regularization parameter reg_param
        Goal:
         - Initialize the weight vector self.w
         - Initialize the polynomial degree self.m
         - Initialize the  regularization parameter self.reg
        """
        self.m = m
        self.reg  = reg_param
        self.dim = [m+1 , 1]
        self.w = np.zeros(self.dim)
    def gen_poly_features(self, X):
        """
        Inputs:
         - X: A numpy array of shape (N,1) containing the data.
        Returns:
         - X_out an augmented training data to an mth degree feature vector e.g. [1, X, X^2, ..., X^m].
        """
        N,d = X.shape
        m = self.m
        X_out= np.zeros((N,m+1))
        if m==1:
            # IMPLEMENT THE MATRIX X_out=[1, X]
            temp = np.ones((N,1))
            X_out = np.hstack((temp,X))
        else:
            temp = np.ones((N,1))
            X_out = np.hstack((temp,X))
            X_poly=np.zeros((N,1))
            for i in range(m-1):
                for j in range(N):
                    X_poly[j]=i+2
                poly=np.power(X,X_poly)
                X_out = np.hstack((X_out,poly))
            # IMPLEMENT THE MATRIX X_out=[1, X, x^2,....,X^m]
        return X_out  
    
    def loss_and_grad(self, X, y):
        """
        Inputs:
        - X: N x d array of training data.
        - y: N x 1 targets 
        Returns:
        - loss: a real number represents the loss 
        - grad: a vector of the same dimensions as self.w containing the gradient of the loss with respect to self.w 
        """
        loss = 0.0
        grad = np.zeros_like(self.w) 
        m = self.m
        N,d = X.shape 
        if m==1:
            # Calculate the loss function of the polynomial regression with order 1
            # Use Mean Squared error function as Loss
            X_bias=self.gen_poly_features(X)
            X_transpose = np.transpose(X_bias)
            y=np.reshape(y,(N,1))
            wTx = np.dot( X_bias, self.w) 
            loss = (1/N)*np.sum(np.square(y-wTx)) + (0.5*(self.reg)*np.sum(np.square((self.w))))
            grad = (2/N) * np.dot(X_transpose, wTx-y) + (self.reg* self.w)
        else:
            # Calculate the loss function of the polynomial regression with order m
            X_bias=self.gen_poly_features(X)
            X_transpose = np.transpose(X_bias)
            y=np.reshape(y,(N,1))
            wTx = np.dot( X_bias, self.w) 
            loss = (1/N)*np.sum(np.square(y-wTx)) + (0.5*(self.reg)*np.sum(np.square((self.w))))
            grad = (2/N) * np.dot(X_transpose, wTx-y) + (self.reg* self.w)
        return loss, grad
    
    def train_LR(self, X, y, eta=1e-3, batch_size=1, num_iters=1000) :
        """
        Finds the coefficients of a {d-1}^th degree polynomial
        that fits the data using least squares batch gradient descent.

        Inputs:
         - X         -- numpy array of shape (N,1), features
         - y         -- numpy array of shape (N,), targets
         - eta       -- float, learning rate
         - num_iters -- integer, maximum number of iterations
         
        Returns:
         - loss_history: vector containing the loss at each training iteration.
         - self.w: optimal weights 
        """
        loss_history = []
        N,d = X.shape
        for t in np.arange(num_iters):
                X_batch = None
                y_batch = None
                # Sample batch_size elements from the training data for use in gradient descent.  
                # After sampling, X_batch should have shape: (batch_size,1), y_batch should have shape: (batch_size,)
                # The indices should be randomly generated to reduce correlations in the dataset.  
                # Use np.random.choice.  It is better to user WITHOUT replacement.
                indexArr=np.array(range(0,N))
                X_batch=list()
                y_batch=list()
                index=np.random.choice(indexArr, batch_size,replace = False)
                X_batch = X[index] 
                y_batch = y[index]
               
                loss = 0.0
                grad = np.zeros_like(self.w) 
                # evaluate loss and gradient for batch data
                # save loss as loss and gradient as grad
                # update the weights self.w
                loss, grad = self.loss_and_grad(X_batch, y_batch)
                self.w=self.w-(eta*grad)
                loss_history.append(loss)
        return loss_history, self.w
    def closed_form(self, X, y):
        """
        Inputs:
        - X: N x 1 array of training data.
        - y: N x 1 array of targets
        Returns:
        - self.w: optimal weights 
        """
        m = self.m
        N,d = X.shape
        if m==1:
            # obtain the optimal weights from the closed form solution 
            # this is done by taking the gradient of the loss function with respect to w and solving for w
            X_bias=self.gen_poly_features(X)
            X_transpose=np.transpose(X_bias)
            y=np.reshape(y,(N,1))
            xTx_inverse = np.linalg.inv(np.dot(X_transpose, X_bias)+(self.reg*np.identity(m+1)))
            xTy = np.dot(X_transpose, y)
            self.w=np.dot(xTx_inverse, xTy)
            wTx = np.dot( X_bias, self.w)
            loss = (1/N)*np.sum(np.square(y-wTx)) + (0.5*(self.reg)*np.sum(np.square(self.w)))
        else:
            # IMPLEMENT THE MATRIX X_out=[1, X, x^2,....,X^m]
            X_bias=self.gen_poly_features(X)
            X_transpose=np.transpose(X_bias)
            y=np.reshape(y,(N,1))
            xTx_inverse = np.linalg.inv(np.dot(X_transpose, X_bias)+(self.reg*np.identity(m+1)))
            xTy = np.dot(X_transpose, y)
            self.w=np.dot(xTx_inverse, xTy)
            wTx = np.dot( X_bias, self.w)
            loss = (1/N)*np.sum(np.square(y-wTx)) + (0.5*(self.reg)*np.sum(np.square(self.w)))
        return loss, self.w
    
    
    def predict(self, X):
        """
        Inputs:
        - X: N x 1 array of training data.
        Returns:
        - y_pred: Predicted targets for the data in X. y_pred is a 1-dimensional
          array of length N.
        """
        y_pred = np.zeros(X.shape[0])
        m = self.m
        if m==1:
            # PREDICT THE TARGETS OF X 
            # ================================================================ #
            X_bias=self.gen_poly_features(X)
            y_pred = np.dot(X_bias,self.w)
        else:
            # IMPLEMENT THE MATRIX X_out=[1, X, x^2,....,X^m]
            # ================================================================ #
            X_bias=self.gen_poly_features(X)
            y_pred = np.dot(X_bias,self.w)
        return y_pred