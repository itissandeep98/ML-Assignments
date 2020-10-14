def create_sample(X_train,y_train):
  """
    creates a bootstrap sample from the given dataset

    Parameters
    ----------
    X_train : 2-dimensional numpy array of shape (n_samples, n_features) 

    y_train : 1-dimensional numpy array of shape (n_samples,)

    Returns
    -------
    X : 2-dimensional numpy array of shape (n_samples, n_features) 

    y : 1-dimensional numpy array of shape (n_samples,)
    """
  y=np.array([])
  X=np.array([])
  
  for i in range(y_train.shape[0]):
    idx=np.random.randint(0,y_train.shape[0]) # randomly select a sample
    X=np.append(X,X_train[idx])
    y=np.append(y,y_train[idx])
  X=X.reshape((X.shape[0],1))
  return X,y


preprocessor= MyPreProcessor()
X,y=preprocessor.pre_process(2)
B=100
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

predictions=None
for _ in range(B):
  X_temp,y_temp=create_sample(X_train,y_train) # create Bootstrap sample

  linear=LinearRegression() 
  linear.fit(X_temp,y_temp) # Train the linear regression model using SKlearn

  temp=X_test.dot(linear.coef_)+linear.intercept_ # calculate the prediction on test dataset
  temp=temp.reshape((temp.shape[0],1))
  
  if(predictions is not None): # STore the predictions in a numpy array
    predictions=np.concatenate((predictions,temp),axis=1)
  else:
    predictions=temp
  
avg_pred=np.mean(predictions,axis=1) # Average prediction of all bootstrap samples
diff=avg_pred-y_test  

bias=abs(diff).mean() # average bias of every sample

avg=avg_pred.reshape((avg_pred.shape[0],1))
variance=(1/(B-1))*np.sum((predictions-avg)**2,axis=1)
variance=variance.mean() # average variance of every sample

y_test=y_test.reshape((y_test.shape[0],1))
MSE=(1/B)*np.sum((predictions-y_test)**2,axis=1)
MSE=MSE.mean() # average Mean Squared error of every sample

print("Bias:",bias)
print("Variance:",variance)
print("MSE:",MSE)
print(MSE-bias**2-variance)
