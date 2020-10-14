''' Implementation '''
class NaiveBayes():
  '''
  My implementatin of Naive Bayes
  '''
  def __init__(self):
    pass

  def calculateStats(self,X,size):
    """
    calculate statistics like mean and variance of column data

    Parameters
    ----------
    X : 2-dimensional numpy array of shape (n_samples, n_features) 

    size : total size of dataset

    Returns
    -------
    stats : dictionary containing all the statistics of each label and each column
    """
    stats = np.array([[np.mean(col), np.var(col),X.shape[0]/size] for col in zip(*X)])
    return stats

  def separateByClass(self,X,y):
    """
    Separates the given dataset on the basis of labels of y

    Parameters
    ----------
    X : 2-dimensional numpy array of shape (n_samples, n_features) 

    y : 1-dimensional numpy array of shape (n_samples,)

    Returns
    -------
    separated : dictionary having labels as keys and values as associated datasets
    """
    separated={}
    for i in range(X.shape[0]):
      if(y[i] in separated):
        separated[y[i]]=np.concatenate([separated[y[i]],[X[i]]])
      else:
        separated[y[i]]=[X[i]]
    
    return separated

  def fit(self,X,y):
    """
    Trains the dataset for the naive bayes model

    Parameters
    ----------
    X : 2-dimensional numpy array of shape (n_samples, n_features) 

    y : 1-dimensional numpy array of shape (n_samples,)

    Returns
    -------
    """

      separated= self.separateByClass(X,y)
      self.stats={}
      for label in separated:
        self.stats[label]=self.calculateStats(separated[label],X.shape[0])
        
      return self.stats

  def calculateGaussian(self,x, mean, var):
    """
    calculates the log of gaussian probability value for given row and mean and variance

    Parameters
    ----------
    X : 2-dimensional numpy array of shape (1, 1) 

    mean : mean of columns

    var: variance of columns

    Returns
    -------
    """
    if(var<10**-7):
      return 0
    exponent = -((x-mean)**2 / (2 * var ))
    return np.log((1 / np.sqrt(2 * pi * var))) + exponent

  
  def calculateClassProbability(self,X):
    """
    Calculates the probabilities of each label for the row

    Parameters
    ----------
    X : 2-dimensional numpy array of shape (1, n_features) 

    Returns
    -------
    probabilities : calculated probailities of each
    """
    probabilities={}
    for label in self.stats: # for each label
      probabilities[label]=0
      for col in range(X.shape[0]): # for each column
        mean,var,_=self.stats[label][col]
        probabilities[label]+=self.calculateGaussian(X[col],mean,var) # calculates the probabilities
        # assert probabilities[label]!=float("-inf"),"data:"+str(mean)+" "+str(var) +" col:"+str(col)

    return probabilities


  def getPrediction(self,X):
    """
    calculates the prediction(label) of the row

    Parameters
    ----------
    X : 2-dimensional numpy array of shape (1, n_features) 

    Returns
    -------
    bestlabel: label having max probability
    """
    bestlabel=0
    max_prob=float("-inf")
    probabilities=self.calculateClassProbability(X) # calculate probability for each label
  
    for label in probabilities: # select the label having maximum probability
      if(max_prob<probabilities[label]):
        max_prob=probabilities[label]
        bestlabel=label
    return bestlabel


  def predict(self,X):
    """
    predicts output for given dataset

    Parameters
    ----------
    X : 2-dimensional numpy array of shape (n_samples, n_features) 

    Returns
    -------
    y : 1-dimensional numpy array of shape (n_samples,)

    """
    y=np.array([])
    for i in range(X.shape[0]):
      yi=self.getPrediction(X[i])
      y=np.append(y,yi)
    y=y.astype(int)
    return y
    
def naive_bayes(data,predictor):
  preprocessor = MyPreProcessor()
  X, y = preprocessor.pre_process(data)
  X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y, test_size=0.1,random_state=40) 
  predictor.fit(X_train,y_train)
  y_pred=predictor.predict(X_test)
  return accuracy_score(y_test,y_pred)
    
    
    
''' Assessment '''

nb=NaiveBayes()
gnb=GaussianNB()
print("SKlearn's naive Bayes:",naive_bayes(0,gnb))
print("Custom naive Bayes:",naive_bayes(0,nb))
    
    
    
    
nb=NaiveBayes()
gnb=GaussianNB()
print("SKlearn's naive Bayes:",naive_bayes(1,gnb))
print("Custom naive Bayes:",naive_bayes(1,nb))
    
    
    
    
    
    
    
    
    
    
    
