''' Grid Search Implementation '''
class GridSearch():
  '''
  My Implementation of Grid search cross validation
  '''

  def __init__(self,estimator,params={"max_depth":None},cv=5):
    """
    Initialises the class with parameters

    Parameters
    ----------
    estimator : estimator that needs to be evaluated for best parameters 

    params : user provided Dictionary of parameters for which grid search is implemented

    k : Number of folds the data needs to be splitted into

    cv : Number of cross validation sets

    Returns
    -------
    """
    self.estimator=estimator
    self.k=cv
    self.depths=params['max_depth']
    self.grid=list(ParameterGrid(params)) # creates a grid of parameters

  def fit(self,X,y):
    validation_acc=[]
    training_acc=[]
    max_val=-1
    for i in self.grid:
      params=i
      self.estimator.set_params(**params) # update estimator with selected parameters
      val,train=self.k_fold(X,y) # calculates the kfold validation and training accuracy on selected set of parameters

      print(params,"Validation Accuracy:",val,"Training Accuracy:",train)

      validation_acc.append(val) # store the accuracy values
      training_acc.append(train)

      if(val>max_val): # store the best estimator using pickle library
        pickle.dump(self.estimator,open('model_dt-0','wb'))
        max_val=val
    
    #plot the Training and validation accuracy
    plt.plot(self.depths,validation_acc,color="g", label="validation Accuracy")
    plt.plot(self.depths,training_acc,color="r", label="training Accuracy")
    plt.xlabel('Depth')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

  def k_fold(self,X,y):
    """
    performs k fold cross validation on the given dataset with given  estimator 
    updated with selected parameters

    Parameters
    ----------
    X : 2-dimensional numpy array of shape (n_samples, n_features) 

    y : 1-dimensional numpy array of shape (n_samples,)

    Returns
    -------
    val_acc : average Validation accuracy

    train_acc : average Training accuracy
    """
    m=X.shape[0] # number of samples

    split_start=0 # initial split's first index
    split_end=m//4 # initial split's last index
    score_val=0
    score_train=0
    max_score_val=-1
    max_score_train=-1

    for i in range(self.k):
      X_train=np.concatenate((X[:split_start],X[split_end:]))
      y_train=np.concatenate((y[:split_start],y[split_end:]),axis=0)

      X_val=X[split_start:split_end]
      y_val=y[split_start:split_end]

      split_start+=100
      split_end+=100

      self.estimator.fit(X_train,y_train) # train the model
      score_val+=self.estimator.score(X_val,y_val) # calculate the validation accuracy
      score_train+=self.estimator.score(X_train,y_train) # calculate the training accuracy
      max_score_val=max(max_score_val,self.estimator.score(X_val,y_val))
      max_score_train=max(max_score_train,self.estimator.score(X_train,y_train))
    print("maximum Validation score:",max_score_val)
    print("maximum Training score:",max_score_train)

    return score_val/self.k, score_train/self.k
    
    
    
''' Testing '''
preprocessor = MyPreProcessor()
X, y = preprocessor.pre_process(1)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.20, random_state=40)
dt= DecisionTreeClassifier()
gnb=GaussianNB()

params = {'max_depth': list(range(2, 20))}
grid_search_cv = GridSearch(dt,params)
grid_search_cv.k_fold(X_train,y_train)

grid_search_cv.fit(X_train, y_train)

# model=pickle.load(open('model-dt-0','rb'))
# model.score(X_test,y_test)


''' D part '''

class Evaluation_Metric():
  '''
  My implementation of evaluation metrics
  '''
  def __init__(self,model,X,y):
    """
    Initialises the evaluation metrics

    Parameters
    ----------
    X : 2-dimensional numpy array of shape (n_samples, n_features) 

    y : 1-dimensional numpy array of shape (n_samples,)

    model : Model used to predict the values

    Returns
    -------
    """
    self.y=y
    self.y_pred=model.predict(X)
    self.prob=model.predict_proba(X)
    self.matrix=self.confusion_matrix(self.y,self.y_pred)
    
  
  def confusion_matrix(self,y,y_pred):
    """
    Creates the confusion matrix with given y and y_pred

    Parameters
    ----------
    y : 1-dimensional numpy array of shape (n_samples,) 

    y_pred : 1-dimensional numpy array of shape (n_samples,)

    Returns
    -------
    result : Created confusion matrix
    """
    K = len(np.unique(y)) # Number of classes 
    result = np.zeros((K, K))

    for i in range(len(y)):
      result[y[i]][y_pred[i]] += 1
    result=result.astype(int)

    return result

  def asses(self):
    """
    prints all the details of given dataset with given model
    
    """
    print("Confusion Matrix")
    print(tabulate(self.matrix))
    print("*-"*30)

    K = len(np.unique(self.y))
    if(K>2):
      data=np.zeros((K,3))
    else:
      data=np.zeros((1,3))
    data[:,0]=self.precision()
    data[:,1]=self.recall()
    data[:,2]=self.F1score()
   
    print(tabulate(data,headers=['Precision','Recall','F1-score']))
    

    plt.figure(figsize=(16,9))
    if(K>2):
      data=[["Accuracy/Micro-average:",self.accuracy()],["Macro-Average:", * self.macro_average()]]
      print(tabulate(data))
      for i in range(K):
        y_temp=self.y.copy()
        y_temp[y_temp==i]=K+1
        y_temp[y_temp!=K+1]=0
        y_temp[y_temp==K+1]=1
       
        FPR,TPR=self.plotROC(y_temp,self.prob[:,i])
        plt.plot(FPR,TPR,label=str(i))
      
    else:
      data=[["Accuracy:",self.accuracy()]]
      print(tabulate(data))
      FPR,TPR=self.plotROC(self.y,self.prob[:,1])
      plt.plot(FPR,TPR,label="ROC Curve")

    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    
    
  def macro_average(self):  
    """
    Calculates the macro average of precision, recall, F1score

    Parameters
    ----------
    y : 1-dimensional numpy array of shape (n_samples,) 

    y_pred : 1-dimensional numpy array of shape (n_samples,)

    Returns
    -------
    result : Created confusion matrix
    """  
    return [self.precision().mean(),self.recall().mean(),self.F1score().mean()]

  def accuracy(self):
    """
    Calculates the accuracy of dataset
    """  
    return self.matrix.trace()/self.matrix.sum()

  def precision(self):
    """
    Calculates the precision of dataset
    """  

    K = len(np.unique(self.y))
    labels=np.zeros(K)
    if(K==2):
      return self.matrix[1,1]/self.matrix[:,1].sum()

    for i in range(K):
      labels[i]=self.matrix[i][i]/self.matrix[:,i].sum()

    return labels

  def recall(self):
    """
    Calculates the recall value of dataset
    """  
    K = len(np.unique(self.y))
    labels=np.zeros(K)
    if(K==2):
      return self.matrix[1,1]/self.matrix[1].sum()

    for i in range(K):
      labels[i]=self.matrix[i][i]/self.matrix[i].sum()

    return labels

  def F1score(self):
    """
    Calculates the F1 score value of dataset
    """  
    prec=self.precision()
    recall=self.recall()
    return 2*(prec*recall)/(prec+recall)
  

  def plotROC(self,y,y_prob):
    """
    Calculates the True positive rate and false positive rate of model

    Parameters
    ----------
    y : 1-dimensional numpy array of shape (n_samples,) 

    y_prob : 1-dimensional numpy array of shape (n_samples,)

    Returns
    -------
    TPR : True Positive rate

    FPR : False Positive rate
    """  
    thresh=0
    TPR=[]
    FPR=[]
    
    while(thresh<=1):
      y_pred=y_prob.copy()
      y_pred[y_pred>=thresh]=1
      y_pred[y_pred<thresh]=0
      y_pred=y_pred.astype(int)
     
      matrix=self.confusion_matrix(y,y_pred)
     
      FPR.append(matrix[0,1]/matrix[0].sum())
      TPR.append(matrix[1,1]/matrix[1].sum())
      
      thresh+=0.0002

    return FPR,TPR
      

''' Processing '''

preprocessor = MyPreProcessor()
X, y = preprocessor.pre_process(1)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.20, random_state=40)
gnb=GaussianNB()
gnb.fit(X_train,y_train)
# model=pickle.load(open('model','rb'))
metric=Evaluation_Metric(gnb,X_test,y_test)
metric.asses()












