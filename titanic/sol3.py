import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve



def detect_outlier(df, n, features):
	outlier_indices = []

	for col in features:
		q1 = np.percentile(df[col], 25)
		q3 = np.percentile(df[col], 75)

		iqr = q3 - q1
		outlier_step = 1.5 * iqr

		outlier_list_col = df[(df[col] < q1 - outlier_step) | (df[col] > q3 + outlier_step)].index

		outlier_indices.extend(outlier_list_col)

	outlier_indices = Counter(outlier_indices)
	multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)

	return multiple_outliers


train = pd.read_csv('./input/train.csv')
test = pd.read_csv('./input/test.csv')
idtest = test["PassengerId"]

print(train.info())
print(train.isnull().sum())

outlier_drop = detect_outlier(train, 2, ['Age', 'SibSp', 'Parch', 'Fare'])
# print(train.loc[outlier_drop])

train = train.drop(outlier_drop, axis=0).reset_index(drop=True)
train_length = len(train)

dataset = pd.concat(objs=[train, test], axis=0).reset_index(drop=True)

dataset = dataset.fillna(np.nan)

# print(dataset.isnull().sum())

# print(train.describe())

dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].mean())

# g = sns.distplot(dataset['Fare'], color="m")
# g = g.legend(loc="best")
# plt.show()

# print(dataset['Fare'].skew())

dataset["Fare"] = dataset["Fare"].map(lambda i: np.log(i) if i > 0 else 0)

# g = sns.distplot(dataset['Fare'], color="m")
# g = g.legend(loc="best")
# plt.show()

dataset['Embarked'].fillna(dataset['Embarked'].mode()[0])
dataset['Sex'] = dataset['Sex'].map({"male": 0, "female": 1})

# g = sns.heatmap(dataset[["Age", "Sex", "SibSp", "Parch", "Pclass"]].corr(), cmap="BrBG", annot=True)
# plt.show()

index_nan_age = list(dataset['Age'][dataset['Age'].isnull()].index)

for i in index_nan_age:
	age_med = dataset['Age'].median()
	age_pred = dataset['Age'][(dataset["SibSp"] == dataset.iloc[i]['SibSp']) &
							(dataset["Parch"] == dataset.iloc[i]['Parch']) &
							(dataset["Pclass"] == dataset.iloc[i]['Pclass'])].median()

	if not np.isnan(age_pred):
		dataset['Age'].iloc[i] = age_pred
	else:
		dataset['Age'].iloc[i] = age_med

dataset_name_title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]
dataset['Title'] = pd.Series(dataset_name_title)
# print(dataset['Title'].value_counts())

dataset["Title"] = dataset["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

dataset["Title"] = dataset["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr": 2, "Rare":3})

dataset["Title"] = dataset["Title"].astype(int)

print(dataset['Title'].value_counts())

# g = sns.factorplot(x = 'Title', y = 'Survived', data=dataset, kind="bar")
# plt.show()

dataset.drop(labels=["Name"], axis=1, inplace=True)

dataset['Fsize'] = dataset['SibSp'] + dataset['Parch'] + 1
dataset['Single'] = dataset['Fsize'].map(lambda s: 1 if s == 1 else 0)
dataset['SmallF'] = dataset['Fsize'].map(lambda s: 1 if s == 2 else 0)
dataset['MedF'] = dataset['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
dataset['LargeF'] = dataset['Fsize'].map(lambda s: 1 if s >= 5 else 0)


dataset = pd.get_dummies(dataset, columns=["Title"])
dataset = pd.get_dummies(dataset, columns=["Embarked"], prefix="Em")
dataset["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in dataset['Cabin']])

dataset = pd.get_dummies(dataset, columns=["Cabin"])

# print(dataset['Cabin'].value_counts())

Ticket = []

for i in list(dataset["Ticket"]):
	if not i.isdigit():
		Ticket.append(i.replace(".", "").replace("/", "").strip().split(" ")[0])
	else:
		Ticket.append('X')
dataset['Ticket'] = Ticket
dataset = pd.get_dummies(dataset, columns=["Ticket"], prefix="T")

dataset['Pclass'] = dataset['Pclass'].astype("category")
dataset = pd.get_dummies(dataset, columns=["Pclass"], prefix="Pc")

dataset.drop(labels=["PassengerId"], axis=1, inplace=True)

train = dataset[:train_length]
test = dataset[train_length:]

test.drop(labels=["Survived"], axis=1, inplace=True)
# print(train.info())
train['Survived'] = train['Survived'].astype(int)
print(train.info())
Y_train = train['Survived']
X_train = train.drop(labels=["Survived"], axis=1)

# random_state = 0
#
# classifiers = []
# classifiers.append(SVC(random_state=random_state))
# classifiers.append(DecisionTreeClassifier(random_state=random_state))
# classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
# classifiers.append(RandomForestClassifier(random_state=random_state))
# classifiers.append(ExtraTreesClassifier(random_state=random_state))
# classifiers.append(GradientBoostingClassifier(random_state=random_state))
# classifiers.append(MLPClassifier(random_state=random_state))
# classifiers.append(KNeighborsClassifier())
# classifiers.append(LogisticRegression(random_state = random_state))
# classifiers.append(LinearDiscriminantAnalysis())
#
# cv_results = []
#
# kfold = StratifiedKFold(n_splits=10)
#
# for classifier in classifiers:
# 	cv_results.append(cross_val_score(classifier, X_train, y=Y_train, scoring='accuracy', cv=kfold, n_jobs=4))
# 	# print(cross_val_score(classifier, X_train, y=Y_train, scoring='accuracy', cv=kfold, n_jobs=4))
# 	# cv_results.append(test_score)
#
# cv_means = []
# cv_std = []
#
# for cv_result in cv_results:
# 	cv_means.append(cv_result.mean())
# 	cv_std.append(cv_result.std())
#
# cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","AdaBoost",
# "RandomForest","ExtraTrees","GradientBoosting","MultipleLayerPerceptron","KNeighboors","LogisticRegression","LinearDiscriminantAnalysis"]})
#
# g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h")
# plt.show()test = dataset[train_length:]

# test.drop(labels=["Survived"], axis=1, inplace=True)
# print(train.info())
train['Survived'] = train['Survived'].astype(int)
print(train.info())

# print(dataset.head())

#AdaBoost best estimator
# ada_best = AdaBoostClassifier(algorithm='SAMME.R',
#                    base_estimator=DecisionTreeClassifier(class_weight=None,
#                                                          criterion='entropy',
#                                                          max_depth=None,
#                                                          max_features=None,
#                                                          max_leaf_nodes=None,
#                                                          min_impurity_decrease=0.0,
#                                                          min_impurity_split=None,
#                                                          min_samples_leaf=1,
#                                                          min_samples_split=2,
#                                                          min_weight_fraction_leaf=0.0,
#                                                          presort=False,
#                                                          random_state=None,
#                                                          splitter='random'),
#                    learning_rate=0.01, n_estimators=2, random_state=0)
##
DTC = DecisionTreeClassifier()
adaDTC = AdaBoostClassifier(DTC, random_state=7)

ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
				"base_estimator__splitter" :   ["best", "random"],
				"algorithm" : ["SAMME","SAMME.R"],
				"n_estimators" :[1,2],
				"learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}

gsadaDTC = GridSearchCV(adaDTC, param_grid=ada_param_grid, scoring="accuracy", n_jobs=4, verbose=1)
gsadaDTC.fit(X_train, Y_train)

ada_best = gsadaDTC.best_estimator_
# print(ada_best)

#ExtraTreeClassifier
# Extc_best = ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='gini',
#                      max_depth=None, max_features=10, max_leaf_nodes=None,
#                      min_impurity_decrease=0.0, min_impurity_split=None,
#                      min_samples_leaf=10, min_samples_split=3,
#                      min_weight_fraction_leaf=0.0, n_estimators=100,
#                      n_jobs=None, oob_score=False, random_state=None, verbose=0,
#                      warm_start=False)
##
gsExtC = ExtraTreesClassifier()

ex_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}
gsExtC = GridSearchCV(gsExtC, param_grid=ex_param_grid, scoring="accuracy", n_jobs=4, verbose=1)
gsExtC.fit(X_train, Y_train)

Extc_best = gsExtC.best_estimator_

#RFC
# rfc_best = RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
#                        max_depth=None, max_features=10, max_leaf_nodes=None,
#                        min_impurity_decrease=0.0, min_impurity_split=None,
#                        min_samples_leaf=3, min_samples_split=2,
#                        min_weight_fraction_leaf=0.0, n_estimators=300,
#                        n_jobs=None, oob_score=False, random_state=None,
#                        verbose=0, warm_start=False)
##
RFC = RandomForestClassifier()
rf_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}
gsRFC = GridSearchCV(RFC, param_grid=rf_param_grid, scoring="accuracy", n_jobs=4, verbose=1)
gsRFC.fit(X_train, Y_train)
# print(gsRFC.best_score_)
rfc_best = gsRFC.best_estimator_


#GBC
# gbc_best = GradientBoostingClassifier(criterion='friedman_mse', init=None,
#                            learning_rate=0.1, loss='deviance', max_depth=4,
#                            max_features=0.3, max_leaf_nodes=None,
#                            min_impurity_decrease=0.0, min_impurity_split=None,
#                            min_samples_leaf=100, min_samples_split=2,
#                            min_weight_fraction_leaf=0.0, n_estimators=300,
#                            n_iter_no_change=None, presort='auto',
#                            random_state=None, subsample=1.0, tol=0.0001,
#                            validation_fraction=0.1, verbose=0,
#                            warm_start=False)
##
GBC = GradientBoostingClassifier()

gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1]
              }
gsGBC = GridSearchCV(GBC, param_grid=gb_param_grid, scoring="accuracy", n_jobs=4, verbose=1)
gsGBC.fit(X_train, Y_train)

gbc_best = gsGBC.best_estimator_
# print(gsGBC.best_score_)

#SVC
# svc_best = SVC(C=50, cache_size=200, class_weight=None, coef0=0.0,
#     decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
#     max_iter=-1, probability=True, random_state=None, shrinking=True, tol=0.001,
#     verbose=False)
##
SVC = SVC(probability=True)
svc_param_grid = {'kernel': ['rbf'],
                  'gamma': [ 0.001, 0.01, 0.1, 1],
                  'C': [1, 10, 50, 100,200,300, 1000]}

gsSVC = GridSearchCV(SVC, param_grid=svc_param_grid, scoring="accuracy", n_jobs=4, verbose=1)
gsSVC.fit(X_train, Y_train)
svc_best = gsSVC.best_estimator_

votingC = VotingClassifier(estimators=[('ada', ada_best), ('rfc', rfc_best), ("extc", Extc_best), ("svc", svc_best)], voting='soft', n_jobs=4)

votingC.fit(X_train, Y_train)
# print(votingC.score(X_train, y=Y_train))

test_survived = pd.Series(votingC.predict(test), name="Survived")
submit = pd.concat([idtest, test_survived], axis=1)

submit.to_csv("submission3.csv", index=False)