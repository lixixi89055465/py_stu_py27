# coding: utf-8
# @Time    : 2020-07-26 15:21
# @Author  : lixiang
# @File    : taitanike.py
import pandas
titanic=pandas.read_csv('titanic_train.csv')
print titanic.head(5)
titanic['Age']=titanic['Age'].fillna(titanic['Age'].median())
print titanic.describe()
print titanic['Sex'].unique()
titanic.loc[titanic['Sex']=='male','Sex']=0
titanic.loc[titanic['Sex']=='female','Sex']=1
print titanic['Embarked'].unique()

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
predictors=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
alg=LinearRegression()
kf=KFold(n_splits=titanic.shape[0],random_state=None,shuffle=False)
predictions=[]
for train,test in kf.split():
    print train,test