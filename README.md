# Breast Cancer Detection By KNN, Decsion tree, Logistic regression ,Random Forest, SVM ,NBC

#### Importing python libraries so that we can work with DataFrame

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

### Importing the csv file(DataSet/DataFrame)
- Syntax to Load or import the DataSet(DataFrame)
- d = pd.read_csv(r'csv_FilePath Or Location\csv_fileName')
- d is a variable where we have stored Whole Data Set/frame

### Breast Cancer DataSet

d = pd.read_csv(r'A:\MTECH(Data Science)\DataSet\breast cancer.csv')
d

### Now to Analysis of Breast cancer's Data we have to performed Following Operations :-

## (1) d.head()
- This command will show the by default first 5 rows from the loaded DataSet.

d.head()

## (2) d.tail()
- This command will show the by default last 5 rows from the loaded DataSet.

d.tail()

## (3)  d.shape
- This command will show the Total No. of rows and Total No. of Columns of the loaded DataSet.

d.shape

## (4)  d.info()
- This command will provide us basic information about the DataFrame.

d.info()

## (5) d.duplicated()
- d.duplicated() is used to check duplicate value present or not?

d.duplicated().value_counts()

## (6) pd.isnull(d).sum()
- This command is used to check NULL values present or not in our dataframe.
- isnull command is used to check for null values.

pd.isnull(d).sum()

###### From above information we can say that there are no Null values.

### (7) Now we try to know how many people have breast cancer and how many people have not got breast cancer
- Malignant(M) = Breast cancer has happened
- Benign(B) = Breast cancer has not happened

d['diagnosis'].value_counts()

sns.countplot(x='diagnosis', data = d)

# correlation test
d.corr()

# visualize the correlation
plt.figure(figsize=(10,10))
sns.heatmap(d.iloc[:,1:10].corr(),annot=True,fmt=".0%")

### (8) We know that machine model can only understand binary format(0/1) so we have to convert M and B into 1 and 0
- LabelEncoder = is used to convert non-numerical values into numerical values or 
- LabelEncoder = is used to convert Categorical variables into numerical values.
- LabelEncoder = is used to convert String into Binary format

- we have to import sklearn to work with machine learning model
- Sklearn = sklearn is python library used for Machine learning

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder() # Call LabelEncoder fun. and assign into LE variable
d.iloc[:,1] = LE.fit_transform(d.iloc[:,1].values) # iloc is used for position based selection.

d.head()

### (9) Now we are checking relationship

sns.pairplot(d.iloc[:,1:5],hue='diagnosis') # hue = is used to visualize the data of different categories in one plot.

# Correlation
d.corr()

### (10) split the dataset into Training and Testing sets for Model Evaluation

X = d.iloc[:,2:31].values
Y = d.iloc[:,1].values

# Dependent(X) datasets
X

# Independent(Y) datasets
Y

### (11) Now spliting the data for trainning(80%) and testing(20%) dateset

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=0)

### (12) feature scaling(Normalized the Data)

from sklearn.preprocessing import StandardScaler
X_train=StandardScaler().fit_transform(X_train)
X_test=StandardScaler().fit_transform(X_test)

## (13) Applying Model
- KNN
- Decsion tree
- Logistic regression
- Random Forest
- SVM
- NBC

def models(X_train,Y_train):
        # 1.KNN
        from sklearn.neighbors import KNeighborsClassifier
        knc = KNeighborsClassifier(n_neighbors = 5 , metric = 'minkowski',p=2)
        knc.fit(X_train,Y_train)
        
        # 2.Decision Tree
        from sklearn.tree import DecisionTreeClassifier
        dtc = DecisionTreeClassifier(random_state=0,criterion="entropy")
        dtc.fit(X_train,Y_train)
        
        # 3.logistic regression
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(random_state=0)
        lr.fit(X_train,Y_train)
        
        # 4.Random Forest
        from sklearn.ensemble import RandomForestClassifier
        rfc = RandomForestClassifier(random_state=0,criterion="entropy",n_estimators=10)
        rfc.fit(X_train,Y_train)
        
        # 5.SVM
        from sklearn.svm import SVC
        svc = SVC(random_state=0)
        svc.fit(X_train,Y_train)
        
        # 6.NBC
        from sklearn.naive_bayes import GaussianNB
        nbc = GaussianNB()
        nbc.fit(X_train,Y_train)
        
        print('[0] K-Nearest Neighbors accuracy:',knc.score(X_train,Y_train))
        print('[1] Decision tree accuracy:',dtc.score(X_train,Y_train))
        print('[2] logistic regression accuracy:',lr.score(X_train,Y_train))
        print('[3] Random forest accuracy:',rfc.score(X_train,Y_train))
        print('[4] Support Vector Machine:',svc.score(X_train,Y_train))
        print('[5] Naive Bayes Classifier:',nbc.score(X_train,Y_train))
        
        return knc,dtc,lr,rfc,svc,nbc

model = models(X_train,Y_train)

### (14) Now Finally Testing the model's Accuracy , Precision , Recall and F1 score

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

for i in range(len(model)):
    print("Model",i)
    print(classification_report(Y_test,model[i].predict(X_test)))
    print('Accuracy : ',accuracy_score(Y_test,model[i].predict(X_test)))

## Conclusion :-
- Random forest is best model for this detection.

### Name - Aatish Kumar Baitha


### M.Tech(Data Science 2nd Year Student)

### My Linkedin Profile -
- www.linkedin.com/in/aatish-kumar-baitha-ba9523191

### My Blog
- https://computersciencedatascience.blogspot.com/

PLEASE CLICK bellow link or .ipynb file to see original view of this project.
https://github.com/Aatishkb/Breast-Cancer-Detection-By-KNN-Decsion-tree-Logistic-regression-Random-Forest-SVM-NBC/blob/main/Breast%20Cancer%20Detection%20(2).ipynb

# Thank you!
