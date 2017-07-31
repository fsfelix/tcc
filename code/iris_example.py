from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)

clf = svm.SVC(kernel='rbf', C=1)
clf.fit(X_train, y_train)

clf.predict(X_test) # Retorna lista de labels
clf.score(X_test, y_test) # Retorna f-measure
