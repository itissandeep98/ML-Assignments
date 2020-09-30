import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class MyPreProcessor():
    """
    My steps for pre-processing for the three datasets.
    """

    def __init__(self):
        pass

    def pre_process(self, dataset):
        """
        Reading the file and preprocessing the input and output.
        Note that you will encode any string value and/or remove empty entries in this function only.
        Further any pre processing steps have to be performed in this function too. 

        Parameters
        ----------

        dataset : integer with acceptable values 0, 1, or 2
        0 -> Abalone Dataset
        1 -> VideoGame Dataset
        2 -> BankNote Authentication Dataset

        Returns
        -------
        X : 2-dimensional numpy array of shape (n_samples, n_features)
        y : 1-dimensional numpy array of shape (n_samples,)
        """

        if dataset == 0:
            # data read from file
            df = pd.read_csv('/content/Dataset.data',
                             delim_whitespace=True, header=None)
            df.sample(frac=1)  # data shuffled

            # changed gender values to integers
            df[0].replace('M', 1, inplace=True)
            df[0].replace('F', 2, inplace=True)
            df[0].replace('I', 3, inplace=True)

            data = df.to_numpy()  # converted dataframe into numpy array
            X = data[:, :-1]
            y = data[:, -1]

        elif dataset == 1:
            # data read from file
            df = pd.read_csv('/content/VideoGameDataset.csv')
            # required colums extracted
            df = df[['Critic_Score', 'User_Score', 'Global_Sales']]
            df = df.sample(frac=1)  # data shuffled

            # replaced NaN values with median of the column
            df['Critic_Score'].fillna(
                df['Critic_Score'].median(), inplace=True)
            # replaced the cell with 'tbd' value to NaN value in the colum
            df['User_Score'].replace(
                to_replace='tbd', value=np.nan, inplace=True)
            # converted column from strings to float values
            df['User_Score'] = df['User_Score'].astype(np.float)
            # replaced NaN values with median of the column
            df['User_Score'].fillna(df['User_Score'].median(), inplace=True)

            data = df.to_numpy()  # converted dataframe into numpy array
            X = data[:, :-1]
            y = data[:, -1]

        elif dataset == 2:
            # Implement for the banknote authentication dataset
            df = pd.read_csv('/content/data_banknote_authentication.txt',
                             header=None)  # data read from file
            df = df.sample(frac=1)  # data shuffled
            X = df[[0, 1, 2, 3]].to_numpy()
            y = df[[4]].to_numpy()
            y = y.squeeze()

        elif dataset == 3:
            df = pd.read_csv('/content/Q4_Dataset.txt',
                             delim_whitespace=True, header=None)
            X = df[[1, 2]].to_numpy()
            y = df[[0]].to_numpy()

        X = (X-X.mean(axis=0))/X.std(axis=0)  # normalized the data

        return X, y


class MyLinearRegression():
    """
          My implementation of Linear Regression.
          """

    def __init__(self):
        pass

    def cross_validation(self, X, y, epoch=1000, alpha=0.01, k=10, lossfunc=1):
        """
        performs k fold cross validation on the given dataset

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) 

        y : 1-dimensional numpy array of shape (n_samples,)

        k : Number of folds the data needs to be splitted into

        epoch : Number of times gradient descent has to run

        alpha : Learning rate of gradient descent

        lossfunc: determines which loss function to call

        Returns
        -------
        """
        m = X.shape[0]  # number of samples

        split_start = 0  # initial split's first index
        split_end = m//k  # initial split's last index

        theta_list = [0]*k  # initialized theta
        # initialized list to store all the training loss from every fold
        training_loss_list = [0]*k
        # initialized list to store all the validation loss from every fold
        validation_loss_list = [0]*k

        error_min = float("inf")
        idx = 0

        for i in range(k):

            # Extracting X and y for train and test set
            X_train = np.concatenate((X[:split_start], X[split_end:]), axis=0)
            y_train = np.concatenate((y[:split_start], y[split_end:]), axis=0)

            X_test = X[split_start:split_end]
            y_test = y[split_start:split_end]

            # calculating model parameters by running the gradient descent
            self.fit(X_train, y_train, X_test, y_test, epoch, alpha, lossfunc)

            # storing the results of current fold in the array
            theta_list[i] = self.theta
            training_loss_list[i] = self.training_loss
            validation_loss_list[i] = self.validation_loss

            split_start = split_end  # updating slice parameters
            split_end += m//k

            error = training_loss_list[i][-1]

            # if the error in this fold is minimum of all errors seen upto now then update it and store the fold number
            if(error < error_min):
                idx = i
                error_min = error

        # final storing the values associated with minimum error
        self.theta = theta_list[idx]
        self.training_loss = training_loss_list[idx]
        self.validation_loss = validation_loss_list[idx]

    def MSE(self, X, y, theta):
        """
        finding Mean Squared Error based on current model parameters

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features)

        y : 1-dimensional numpy array of shape (n_samples,)

        theta : Value of theta at which derivative of cost has to be found

        Returns
        -------
        derv : derivative of cost at the value theta

        err  : Error/cost in prediction at the value theta
        """
        m = len(y)

        # Transpose of vector X
        X_trans = np.transpose(X)
        err = X.dot(theta)-y
        # Calculates X` * ( X*theta - y )
        derv = (1/m)*(X_trans.dot(err))

        return derv, sum(err**2)/m

    def MAE(self, X, y, theta):
        """
        finding Mean Absolute Error based on current model parameters

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features)

        y : 1-dimensional numpy array of shape (n_samples,)

        theta : Value of theta at which derivative of cost has to be found

        Returns
        -------
        derv : derivative of cost at the value theta
        err  : Error/cost in prediction at the value theta
        """
        m = len(y)
        # Calculates (1/m) *( X*theta - y )
        err = (1/m)*(X.dot(theta)-y)
        X_trans = X.T                         # Transpose of vector X
        # epsilon used only for datset-2 where gradient overshoots value and gives nan
        epsilon = 10**-7
        derv = (1/m)*(X_trans.dot(abs(err)/(err)))

        # returns gradient and sum((1/m) *( | X*theta - y |))
        return derv, np.sum(abs(err))

    def RMSE(self, X, y, theta):
        """
        finding Root Mean Squared Error based on current model parameters

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features)

        y : 1-dimensional numpy array of shape (n_samples,) 

        theta : Value of theta at which derivative of cost has to be found

        Returns
        -------
        derv : derivative of cost at the value theta

        err  : Error/cost in prediction at the value theta
        """
        m = len(y)
        X_trans = X.T                                          # Transpose of vector X
        # Calculates ( X*theta - y )
        diff = X.dot(theta)-y

        # Calculates (1/m) *sqrt(sum((X*theta - y)^2))
        err = ((1/m)*np.sum((diff)**2))**0.5
        derv = (1/m)*(X_trans.dot(diff))/err

        return derv, err

    def gradient_descent(self, X, y, X_test, y_test, epochs, alpha, lossfunc):
        """
        Finding theta using the gradient descent method

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as training data.

        y : 1-dimensional numpy array of shape (n_samples,) which acts as training labels.

        X_test : 2-dimensional numpy array of shape (n_samples, n_features) which acts as Testing data.

        y_test : 1-dimensional numpy array of shape (n_samples,) which acts as Testing labels.

        epochs : Number of times gradient descent has to run

        alpha : Learning rate of gradient descent

        lossfunc: determines which loss function to call

        Returns
        -------
        theta : Calculated value of theta on given test set (X,y) with learning rate alpha 

        training_loss: Calculated training loss at every theta

        validation_loss: Calculated validation loss at every theta
        """

        # created a column vector theta of length equal to number of features in X with all the initial values 0
        theta = np.zeros((X.shape[1],))

        # initializing array to store training loss at every value of theta
        training_loss = np.array([])
        # initializing array to store validation loss at every value of theta
        validation_loss = np.array([])

        for i in range(epochs):
            if(lossfunc == 1):     # Using RMSE Loss Function
                derv, train_loss = self.RMSE(X, y, theta)
            elif(lossfunc == 2):   # Using MAE Loss Function
                derv, train_loss = self.MAE(X, y, theta)
            else:                # Using MSE Loss Function
                derv, train_loss = self.MSE(X, y, theta)
            training_loss = np.append(training_loss, train_loss)

            if(X_test is not None):  # calculate validation loss only if test set is provided
                if(lossfunc == 1):    # Using RMSE Loss Function
                    derv_val, val_loss = self.RMSE(X_test, y_test, theta)
                elif(lossfunc == 2):  # Using MAE Loss Function
                    derv_val, val_loss = self.MAE(X_test, y_test, theta)
                else:               # Using MSE Loss Function
                    derv_val, val_loss = self.MSE(X_test, y_test, theta)

                validation_loss = np.append(validation_loss, val_loss)

            theta = theta-alpha*derv

        return theta, training_loss, validation_loss

    def fit(self, X, y, X_test=None, y_test=None, epoch=400, alpha=0.01, lossfunc=1):
        """
        Fitting (training) the linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as training data.

        y : 1-dimensional numpy array of shape (n_samples,) which acts as training labels.

        X_test : 2-dimensional numpy array of shape (n_samples, n_features) which acts as Testing data.

        y_test : 1-dimensional numpy array of shape (n_samples,) which acts as Testing labels.

        epoch : Number of times gradient descent has to run

        alpha : Learning rate of gradient descent

        lossfunc: determines which loss function to call

        Returns
        -------
        self : an instance of self
        """

        # Adding a bias variable i.e columns of 1 to data
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

        if(X_test is not None):  # if validation set is provided then add a bias variable i.e columns of 1 to data
            X_test = np.concatenate(
                (np.ones((X_test.shape[0], 1)), X_test), axis=1)

        X_trans = np.transpose(X)
        if(lossfunc == 4):
            try:
                self.theta = np.linalg.inv(X_trans.dot(X)).dot(X_trans).dot(
                    y)  # using the normal eqn, theta = inv(X`*X)*X`*y
            except:
                # using the gradient descent method with RMSE loss function(default) if the given data is non invertible
                self.theta, self.training_loss, self.validation_loss = self.gradient_descent(
                    X, y, X_test, y_test, epoch, alpha, lossfunc=1)
        else:
            # using the gradient descent method with given number of epochs and learning rate
            self.theta, self.training_loss, self.validation_loss = self.gradient_descent(
                X, y, X_test, y_test, epoch, alpha, lossfunc)

        # fit function has to return an instance of itself or else it won't work with test.py
        return self

    def predict(self, X):
        """
        Predicting values using the trained linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        Returns
        -------
        y : 1-dimensional numpy array of shape (n_samples,) which contains the predicted values.
        """

        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

        y = np.dot(X, self.theta)

        # return the numpy array y which contains the predicted values
        return y

    def plot_loss(self):  # prints and plots all the class variables and regression results
        print("Thetas:", self.theta)
        print("Training Loss:", self.training_loss[-1])
        print("Validation Loss:", self.validation_loss[-1])

        plt.plot(self.training_loss, color="g", label="Training Loss")
        plt.plot(self.validation_loss, color="r", label="Validation Loss")
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()


class MyLogisticRegression():
    """
    My implementation of Logistic Regression.
    """

    def __init__(self):
        pass

    def sigmoid(self, z):
        """
        Find the sigmoid value of z

        Parameters
        ----------
        z : 1-dimensional numpy array of shape (n_samples,)

        Returns
        -------
        value of z in sigmoid function
        """
        return 1/(1+np.exp(-z))

    def accuracy(self, y_hat, y):
        """
        Find the accuracy in predicted data

        Parameters
        ----------
        y_hat : 1-dimensional numpy array of shape (n_samples,), predicted values using regression

        y : 1-dimensional numpy array of shape (n_samples,), Original

        Returns
        -------
        Accuracy value
        """
        m = y.shape[0]  # number of samples
        # if probability of getting 1 is greater than or equal to 0.5 then predict 1
        y_hat[y_hat >= 0.5] = 1
        # if probability of getting 1 is less than 0.5 then predict 0
        y_hat[y_hat < 0.5] = 0
        # returns the average of difference of values in both arrays
        return (1-sum(abs(y-y_hat))/m)*100

    def cost_diff(self, X, y, theta):
        """
        Find Log Loss error in current model parameters

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as training data.

        y : 1-dimensional numpy array of shape (n_samples,) which acts as training labels.

        theta : Value of theta at which derivative of cost has to be found

        Returns
        -------
        derv : derivative of cost at the value theta

        err: error in prediction using the current value of theta

        accuracy: accuracy of prediction of data
        """
        m = y.shape[0]
        X_trans = X.T		  	# Transpose of vector X

        activ = self.sigmoid(X.dot(theta))

        # Calculates X` * ( sigmoid(X*theta) - y )
        derv = (X_trans.dot(activ-y))

        # calculates the cross entropy loss
        err = (-1/m)*(np.sum(y*np.log(activ)+(1-y)*np.log(1-activ+10**-7)))

        accuracy = self.accuracy(activ, y)  # calculates the accuracy

        return derv, err, accuracy

    def stochastic_gradient_descent(self, X, y, X_test=None, y_test=None, epoch=1000, alpha=0.01):
        """
        Finding theta using the stochastic gradient descent model

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as training data.

        y : 1-dimensional numpy array of shape (n_samples,) which acts as training labels.

        X_test : 2-dimensional numpy array of shape (n_samples, n_features) which acts as Testing data.y_test : 1-dimensional numpy array of shape (n_samples,) which acts as Testing labels.

        epoch : Number of times gradient descent has to run

        alpha : Learning rate of gradient descent

        Returns
        -------
        theta : Calculated value of theta on given test set (X,y) with learning rate alpha 

        training_loss: Calculated training loss at every theta

        validation_loss: Calculated validation loss at every theta

        training_loss_acc: Calculated training accuracy at every theta

        validation_lossacc_: Calculated validation accuracy at every theta
        """
        m = y.shape[0]

        # created a column vector theta of length equal to number of features in X with all the initial values 0
        theta = np.zeros((X.shape[1],))

        validation_loss_list = np.array([])
        training_loss_list = np.array([])

        validation_acc_list = np.array([])
        training_acc_list = np.array([])

        for i in range(epoch):

            curr_X = np.array([X[i % m]])
            curr_y = np.array([y[i % m]])

            derv, loss, accuracy = self.cost_diff(curr_X, curr_y, theta)

            derv_train, loss_train, accuracy_train = self.cost_diff(
                X, y, theta)

            training_loss_list = np.append(training_loss_list, loss_train)
            training_acc_list = np.append(training_acc_list, accuracy_train)

            if(X_test is not None):
                derv_val, loss_val, accuracy_val = self.cost_diff(
                    X_test, y_test, theta)

                validation_loss_list = np.append(
                    validation_loss_list, loss_val)
                validation_acc_list = np.append(
                    validation_acc_list, accuracy_val)

            theta = theta-(alpha)*derv

        return theta, validation_loss_list, training_loss_list, validation_acc_list, training_acc_list

    def batch_gradient_descent(self, X, y, X_test=None, y_test=None, epoch=1000, alpha=0.01):
        """
        Finding theta using the batch gradient descent model

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as training data.

        y : 1-dimensional numpy array of shape (n_samples,) which acts as training labels.

        X_test : 2-dimensional numpy array of shape (n_samples, n_features) which acts as Testing data.

        y_test : 1-dimensional numpy array of shape (n_samples,) which acts as Testing labels.

        epoch : Number of times gradient descent has to run

        alpha : Learning rate of gradient descent

        Returns
        -------
        theta : Calculated value of theta on given test set (X,y) with learning rate alpha 

        training_loss: Calculated training loss at every theta

        validation_loss: Calculated validation loss at every theta

        training_loss_acc: Calculated training accuracy at every theta

        validation_lossacc_: Calculated validation accuracy at every theta
        """
        m = y.shape[0]

        # created a column vector theta of length equal to number of features in X with all the initial values 0
        theta = np.zeros((X.shape[1],))

        validation_loss_list = np.array([])
        training_loss_list = np.array([])

        validation_acc_list = np.array([])
        training_acc_list = np.array([])

        for i in range(epoch):
            derv_train, loss_train, accuracy_train = self.cost_diff(
                X, y, theta)
            training_loss_list = np.append(training_loss_list, loss_train)
            training_acc_list = np.append(training_acc_list, accuracy_train)

            if(X_test is not None):
                derv_val, loss_val, accuracy_val = self.cost_diff(
                    X_test, y_test, theta)
                validation_loss_list = np.append(
                    validation_loss_list, loss_val)
                validation_acc_list = np.append(
                    validation_acc_list, accuracy_val)

            theta = theta-(alpha)*derv_train

        return theta, validation_loss_list, training_loss_list, validation_acc_list, training_acc_list

    def fit(self, X, y, X_test=None, y_test=None, epoch=10000, alpha=0.01, type=0):
        """
        Fitting (training) the logistic model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as training data.

        y : 1-dimensional numpy array of shape (n_samples,) which acts as training labels.

        X_test : 2-dimensional numpy array of shape (n_samples, n_features) which acts as Testing data.

        y_test : 1-dimensional numpy array of shape (n_samples,) which acts as Testing labels.

        epoch : Number of times gradient descent has to run

        alpha : Learning rate of gradient descent

        type : 0 value runs batch gradient descent, else stochastic gradient descent is run 

        Returns
        -------
        self : an instance of self
        """
        X = np.concatenate((np.ones(
            (X.shape[0], 1)), X), axis=1)  # Adding a bias variable i.e columns of 1 to data

        if(X_test is not None):  # if validation set is provided then add a bias variable i.e columns of 1 to data
            X_test = np.concatenate(
                (np.ones((X_test.shape[0], 1)), X_test), axis=1)
        if(type == 0):
            self.theta, self.validation_loss, self.training_loss, self.validation_acc, self.training_acc = self.batch_gradient_descent(
                X, y, X_test, y_test, epoch, alpha)		  # using the batch gradient descent method with given number of epochs and learning rate
        else:
            self.theta, self.validation_loss, self.training_loss, self.validation_acc, self.training_acc = self.stochastic_gradient_descent(
                X, y, X_test, y_test, epoch, alpha)  # using the stochastic gradient descent method with given number of epochs and learning rate

        # fit function has to return an instance of itself or else it won't work with test.py
        return self

    def predict(self, X):
        """
        Predicting values using the trained logistic model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        Returns
        -------
        y : 1-dimensional numpy array of shape (n_samples,) which contains the predicted values.
        """
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        y = self.sigmoid(X.dot(self.theta))

        y[y >= 0.5] = 1
        y[y < 0.5] = 0

        # return the numpy array y which contains the predicted values
        return y

    def plot_loss(self):
        print("Thetas:", self.theta)
        print("Training Loss:", self.training_loss[-1])
        print("Validation Loss:", self.validation_loss[-1])

        plt.plot(self.training_loss, color="g", label="Training Loss")
        plt.plot(self.validation_loss, color="r", label="Validation Loss")
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
