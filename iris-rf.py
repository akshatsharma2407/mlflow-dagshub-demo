import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns

import dagshub
dagshub.init(repo_owner='akshatsharma2407', repo_name='mlflow-dagshub-demo', mlflow=True)

mlflow.set_tracking_uri('https://dagshub.com/akshatsharma2407/mlflow-dagshub-demo.mlflow')

iris = load_iris()
x = iris.data 
y = iris.target

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=23)

max_depth = 1
n_estimators = 100

mlflow.set_experiment('iris-rf')

with mlflow.start_run():

    rf = RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators)
    rf .fit(xtrain,ytrain)

    ypred = rf.predict(xtest)

    accuracy = accuracy_score(ytest,ypred)
   
    mlflow.log_metric('accuracy',accuracy)

    mlflow.log_param('max_depth',max_depth)
    mlflow.log_param('n estimators',n_estimators)

    cm = confusion_matrix(ytest,ypred)

    plt.figure(figsize=(5,5))

    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=iris.target_names,yticklabels=iris.target_names)

    plt.ylabel('Actual')
    plt.xlabel('predicted')
    plt.title('confusion matrix')

    plt.savefig('confusion_matrix.png')

    mlflow.sklearn.log_model(rf,'random_forest')

    mlflow.log_artifact('confusion_matrix.png')

    mlflow.set_tag('author','suryansh sharma')
    mlflow.set_tag('model','rf')

    print('accuracy',accuracy)

    mlflow.log_artifact(__file__)