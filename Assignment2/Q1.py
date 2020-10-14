def analysis(X_train, X_test, y_train, y_test):
  """
    performs Sklearn's Logistic Regression and TSNE visualization on the given dataset

    Parameters
    ----------
    X_train : 2-dimensional numpy array of shape (n_samples, n_features) 

    y_train : 1-dimensional numpy array of shape (n_samples,)

    X_test : 2-dimensional numpy array of shape (n_samples, n_features) 

    y_test : 1-dimensional numpy array of shape (n_samples,)

    Returns
    -------
    """
  logistic = LogisticRegression(max_iter=10000)
  logistic.fit(X_train,y_train)
  y_pred=logistic.predict(X_test)

  print("Accuracy:",accuracy_score(y_test,y_pred))

  tsne = TSNE(n_components=2, verbose=2, n_iter=1000)
  tsne_results = tsne.fit_transform(X_train)

  plt.figure(figsize=(16,10))
  sns.scatterplot(
    x=tsne_results[:,0], y=tsne_results[:,1],
    hue=y_train,
    palette=sns.color_palette("hls", 10),
    legend="full"
  )
  
  
''' Data preprocessing and Loading '''

preprocessor = MyPreProcessor()
X, y = preprocessor.pre_process(0)
print(X.shape,y.shape)

''' D part '''
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.20)
freq = np.bincount(y_train)
items = np.nonzero(freq)[0]
print("Training Sample")
print("size:",y_train.shape[0])
print(*list(zip(items,freq[items]))) 
freq = np.bincount(y_test)
items = np.nonzero(freq)[0]
print("Testing Sample")
print("size:",y_test.shape[0])
print(*list(zip(items,freq[items]))) 


''' E part '''
pca = PCA(n_components=50) 
X_train_pca = pca.fit_transform(X_train) 
X_test_pca = pca.transform(X_test) 
analysis(X_train_pca, X_test_pca, y_train, y_test)


''' F part '''
svd = TruncatedSVD(n_components=50)
X_train_svd=svd.fit_transform(X_train)
X_test_svd=svd.transform(X_test)
analysis(X_train_svd, X_test_svd, y_train, y_test)















