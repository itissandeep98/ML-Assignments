from sklearn import metrics 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from scratch import MyLinearRegression, MyLogisticRegression,MyPreProcessor


''' ------------ Dataset 1------------ '''
preprocessor = MyPreProcessor()
X, y = preprocessor.pre_process(0)

linear = MyLinearRegression()
linear.cross_validation(X, y, epoch=10000, alpha=0.01, lossfunc=2)
linear.plot_loss()

''' ------------ Dataset 2------------ '''
preprocessor = MyPreProcessor()
X, y = preprocessor.pre_process(1)

linear = MyLinearRegression()
linear.cross_validation(X, y, epoch=10000, alpha=0.001, lossfunc=2)
linear.plot_loss()

''' ------------ Dataset 3------------ '''

preprocessor = MyPreProcessor()
X, y = preprocessor.pre_process(2)
X_test=X[:X.shape[0]//10]
y_test=y[:y.shape[0]//10]

X_train=X[X.shape[0]//10:]
y_train=y[y.shape[0]//10:]

logistic = MyLogisticRegression()
logistic.fit(X_train, y_train,X_test,y_test,epoch=10000,alpha=0.01,type=1)
print(logistic.training_acc[-1],logistic.validation_acc[-1])
logistic.plot_loss()

''' ------------ SK learn----------- '''

preprocessor = MyPreProcessor()
x, y = preprocessor.pre_process(2)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=4)
logistic_regression = LogisticRegression()
logistic_regression.fit(x_train,y_train)
y_pred = logistic_regression.predict(x_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy Test:",accuracy)
y_pred1 = logistic_regression.predict(x_train)
accuracy = metrics.accuracy_score(y_train, y_pred1)
print("Accuracy Train:",accuracy)
print("Thetas:",logistic_regression.coef_)

