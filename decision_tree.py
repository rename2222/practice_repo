from sklearn.datasets import load_iris

iris = load_iris()

x=iris.data
y=iris.target 
y

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2)

xtrain.shape

from sklearn.tree import DecisionTreeClassifier

clf= DecisionTreeClassifier()

clf.fit(xtrain,ytrain)

ypre= clf.predict(xtest)

from sklearn.metrics import accuracy_score
accuracy_score(ytest,ypre)

from sklearn.tree import plot_tree
from matplotlib.pyplot import rcParams
rcParams['figure.figsize']=80,50
plot_tree(clf)
