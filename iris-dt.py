import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns

mlflow.set_tracking_uri('http://127.0.0.1:5000')

iris = load_iris()
x = iris.data 
y = iris.target

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=23)

max_depth = 1

mlflow.set_experiment('iris-dt')

with mlflow.start_run():

    dt = DecisionTreeClassifier(max_depth=max_depth)
    dt .fit(xtrain,ytrain)

    ypred = dt.predict(xtest)

    accuracy = accuracy_score(ytest,ypred)
   
    mlflow.log_metric('accuracy',accuracy)

    mlflow.log_param('max_depth',max_depth)

    cm = confusion_matrix(ytest,ypred)

    plt.figure(figsize=(5,5))

    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=iris.target_names,yticklabels=iris.target_names)

    plt.ylabel('Actual')
    plt.xlabel('predicted')
    plt.title('confusion matrix')

    plt.savefig('confusion_matrix.png')

    mlflow.sklearn.log_model(dt,'decision_tree')

    mlflow.log_artifact('confusion_matrix.png')

    mlflow.set_tag('author','akshat sharma')
    mlflow.set_tag('model','acha hai')

    print('accuracy',accuracy)

    mlflow.log_artifact(__file__)