import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn import svm
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

#import Dataset
data = pd.read_csv("adult.csv")
#print(data.info())
#print(data.head())
data.boxplot()
plt.show()

data.hist()
plt.show()

#Replace ? with Nan Values
data[data == '?'] = np.nan
#print(data.isnull().sum())

#Repalce Nan values with mode of column as categorical vaiable
for col in ['workclass', 'occupation', 'native.country']:
    data[col].fillna(data[col].mode()[0], inplace=True)
#print(data.isnull().sum())

#Split dataframe for target attribute
x = data.drop(['income'],axis=1)
y = data['income']

#Split Data into training and testing data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 21)

#Encode all categorical values using label encoder
category = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
for feature in category:
        le = preprocessing.LabelEncoder()
        x_train[feature] = le.fit_transform(x_train[feature])
        x_test[feature] = le.transform(x_test[feature])
#print(x_train.head())

#Scale data using Standard Scaler
scaler = StandardScaler()

x_train_copy = x_train.copy()
x_test_copy = x_test.copy()
cols = ['age','fnlwgt','education.num','capital.gain','capital.loss','hours.per.week']
feature1 = x_train_copy[cols]
feature2 = x_test_copy[cols]
feature1 = scaler.fit_transform(feature1)
feature2 = scaler.transform(feature2)
x_train[cols] = feature1
x_test[cols] = feature2


# x_train = pd.DataFrame(scaler.fit_transform(x_train), columns = x.columns)
# x_test = pd.DataFrame(scaler.transform(x_test), columns = x.columns)
print(x_train.head())

#Train Model and predict
reg = LogisticRegression()
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)

print('Logistic Regression accuracy score with all the features: ', accuracy_score(y_test, y_pred))
print("Classification Report for logistic regression with all feaatures :")
print(classification_report(y_test,y_pred))

#Now Use PCA for feature reduction
pca = PCA()
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
print(pca.explained_variance_ratio_)

#So 7 features will be used
#Train using 7 features
pca = PCA(n_components=7)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

#Logistic Regression
reg = LogisticRegression()
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)

print('Score with Pca & Logistic Regression: ', accuracy_score(y_test, y_pred))
print('Classification report for Logistic Regression')
print(classification_report(y_test,y_pred))
#SVM
reg = svm.SVC(kernel = 'linear')
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)

print('Score with SVM: ', accuracy_score(y_test,y_pred))
print('Classification report for SVM :')
print(classification_report(y_test,y_pred))
