# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

pd.set_option('display.max_columns', 12)

train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

combine = [train_df, test_df]

# print(test_df.info())

# print(train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))


# print(train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))

# print(train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))

# print(train_df[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))


# g = sns.FacetGrid(train_df, col='Survived')
# g.map(plt.hist, 'Age', bins=20)

# grid = sns.FacetGrid(train_df, col="Survived", row="Pclass", height=2.2, aspect=1.6)
# grid.map(plt.hist, 'Age', alpha=0.5, bins=20)

# grid2 = sns.FacetGrid(train_df, row="Embarked", height=2.2, aspect=1.6)
# grid2.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
# grid2.add_legend()

# grid3 = sns.FacetGrid(train_df, row="Embarked", col="Survived", height=2.2, aspect=1.6)
# grid3.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
# grid3.add_legend()

# plt.show()

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
guess_age = np.zeros((2, 3))


for dataset in combine:
	dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)
	dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
	dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
	dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
	dataset['Title'] = dataset['Title'].replace('Mme', 'Mr')
	dataset['Title'] = dataset['Title'].map(title_mapping)
	dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)


	for i in range(0 ,2):
		for j in range(0, 3):
			guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j + 1)]['Age'].dropna()

			age_guess = guess_df.median()
			guess_age[i, j] = int(age_guess/0.5 + 0.5)*0.5

	for i in range(0 ,2):
		for j in range(0, 3):
			dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j + 1), 'Age'] = guess_age[i, j]

	# dataset['Age'] = dataset['Age'].astype(int)

	dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
	dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
	dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
	dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
	dataset.loc[(dataset['Age'] > 64), 'Age'] = 4

	dataset['Age'] = dataset['Age'].astype(int)

	dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

	dataset['IsAlone'] = 0
	dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

	freq_port = dataset.Embarked.dropna().mode()[0]
	dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
	dataset['Embarked'] = dataset['Embarked'].map({'S' : 0, 'C': 1, 'Q': 2}).astype(int)

	dataset['Fare'].fillna(dataset['Fare'].dropna().median(), inplace=True)
	
	# dataset['FareBand'] = pd.qcut(dataset['Fare'], 5)

	dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0;
	dataset.loc[ (dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1;
	dataset.loc[ (dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2;
	dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3;

	dataset['Fare'] = dataset['Fare'].astype(int)
	

	# print(freq_port)
# print(train_df.describe)

train_df = train_df.drop(['Name', 'PassengerId', 'Parch', 'SibSp', 'FamilySize'], axis=1)
test_summit = test_df.drop(['Name', 'Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Name', 'PassengerId', 'Parch', 'SibSp', 'FamilySize'], axis=1)
print(train_df.head())
print(test_df.head())

X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]

X_test = test_df;
# Y_test = test_df["Survived"]

#Logistic
# logreg = LogisticRegression()
# logreg.fit(X_train, Y_train)
# Y_pred = logreg.predict(X_test)
# log = round(logreg.score(X_train, Y_train)*100, 2)
# print(log)

# coeff_df = pd.DataFrame(train_df.columns.delete(0))
# coeff_df.columns = ['Feature']
# coeff_df["Correlation"] = pd.Series(logreg.coef_[0])
# print(coeff_df.sort_values(by="Correlation", ascending=False))

# SVM
# svc = SVC()
# svc.fit(X_train, Y_train)
# Y_pred = svc.predict(X_test)
# log = round(svc.score(X_train, Y_train)*100, 2)
# print(log)
# knn
# knn = KNeighborsClassifier()
# knn.fit(X_train, Y_train)
# Y_pred = knn.predict(X_test)
# log = round(knn.score(X_train, Y_train)*100, 2)
# print(log)
# Naive Bayes
# gaussian = GaussianNB()
# gaussian.fit(X_train, Y_train)
# Y_pred = gaussian.predict(X_test)
# log = round(gaussian.score(X_train, Y_train)*100, 2)
# print(log)
# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
log = round(decision_tree.score(X_train, Y_train)*100, 2)
print(log)
#Random forest
# random_forest = RandomForestClassifier()
# random_forest.fit(X_train, Y_train)
# Y_pred = random_forest.predict(X_test)
# log = round(random_forest.score(X_train, Y_train)*100, 2)
# print(log)

submission = pd.DataFrame({
	"PassengerId" : test_summit["PassengerId"],
	"Survived": Y_pred
	})
submission.to_csv("./submission1.csv", index=False)
