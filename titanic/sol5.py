# data analysis and wrangling
import pandas as pd
import numpy as np
# from fancyimpute import KNN
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
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestRegressor

pd.set_option('display.max_columns', 12)

train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

train_df_index = 891

title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4}
guess_age = np.zeros((2, 3))

combine = [train_df, test_df]
dataset = pd.concat(combine)
# train_df = dataset.iloc[:891, :]

# print(train_df.info())

# print(test_df.loc[test_df['Fare'].isnull()])
#
# # for dataset in combine:
# Title Section
dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)
dataset['Title'] = dataset['Title'].replace(['Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer'], 'Rare')
dataset['Title'] = dataset['Title'].replace(['Mlle', 'Ms', 'Lady', 'Dona'], 'Miss')
dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
dataset['Title'] = dataset['Title'].map(title_mapping)

# Sex Section
dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

# Embarked Section
# print(train_df.loc[train_df['Embarked'].isnull()])
dataset.loc[dataset['PassengerId'] == 62, 'Embarked'] = dataset.Embarked.dropna().mode()[0]
dataset.loc[dataset['PassengerId'] == 830, 'Embarked'] = dataset.Embarked.dropna().mode()[0]
dataset['Embarked'] = dataset['Embarked'].map({'S' : 0, 'C': 1, 'Q': 2}).astype(int)


# Fare
dataset.loc[dataset['PassengerId'] == 1044, 'Fare'] = dataset[(dataset['Pclass'] == 3) & (dataset['Embarked'] == 0)]['Fare'].dropna().mean()
dataset['Fare'] = dataset['Fare'].astype(int)

def cabin_estimator(i):
    a = 0
    if i < 14:
        a = 'G'
    elif i >= 14 & i < 25:
        a = 'F'
    elif i >= 25 & i < 38:
        a = 'T'
    elif i >= 38 & i < 49:
        a = 'A'
    elif i >= 49 & i < 53:
        a = 'D'
    elif i >= 53 & i < 60:
        a = 'E'
    elif i >= 60 & i < 115:
        a = 'C'
    else:
        a = 'B'
    return a

dataset.Cabin.fillna("N", inplace=True)
dataset.Cabin = [i[0] for i in dataset['Cabin']]
# print(dataset.groupby('Cabin')['Fare'].mean().sort_values())

dataset['Cabin'] = dataset[['Cabin', 'Fare']].apply(lambda x: cabin_estimator(x[1]) if (x[0] == 'N') else x[0], axis=1)
# dataset = pd.concat([null_cabin, fill_cabin], axis=0)
dataset['Cabin'] = dataset['Cabin'].map({'G': 0, 'F': 1, 'T': 2, 'A': 3, 'D': 4, 'E': 5, 'C': 6, 'B': 7}).astype(int)

# Family Size
dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
dataset.loc[dataset['FamilySize'] == 1, 'FamilySize'] = 0 # single
dataset.loc[(dataset['FamilySize'] > 1) & (dataset['FamilySize'] <= 4), 'FamilySize'] = 1 # medium size
dataset.loc[dataset['FamilySize'] > 4, 'FamilySize'] = 2 # large size

# g1 = sns.distplot(dataset['Age'].dropna())
# dataset['Age'].fillna(np.nan)

#Ticket Bin
ticket_count = dataset['Ticket'].value_counts().reset_index(name='TicketBin').rename(columns={'index': 'Ticket'})
# train_df['TicketBin'] = train_df['Ticket'].value_counts()
dataset = pd.merge(dataset, ticket_count, on=['Ticket'], how='left')

dataset.loc[dataset['TicketBin'] == 1, 'TicketBin'] = 0
dataset.loc[(dataset['TicketBin'] >= 2) & (dataset['TicketBin'] <= 4), 'TicketBin'] = 1
dataset.loc[(dataset['TicketBin'] > 4), 'TicketBin'] = 2

# Age imputation
# Xcol = ['Pclass', 'Sex', 'Title', 'FamilySize', 'Fare', 'Embarked']
# X_incomplete = dataset.loc[:, Xcol + ['Age']]
#
# X_filled = KNN(k=8).fit_transform(X_incomplete)
# # print(X_filled)
# X_filled = X_filled[:, -1:].reshape(1309)
# # dataset['Age'] = X_filled
# dataset['Age'] = np.round(X_filled)
# dataset['Age'] = dataset['Age'].astype(int)

# guess_age = np.zeros((2, 5))
#
# for i in range(0, 2):
#     for j in range(0, 5):
#         guess_df = dataset[(dataset['Sex'] == i) & (dataset['Title'] == j)]['Age'].dropna()
#
#         age_guess = guess_df.mean()
#         if(not np.isnan(age_guess)):
#             guess_age[i, j] = int(age_guess)
#
# for i in range(0, 2):
#     for j in range(0, 5):
#         dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Title == j), 'Age'] = guess_age[i, j]
#
# dataset['Age'] = dataset['Age'].astype(int)

def complete_age(df):
    temp_train = df.loc[df['Age'].notnull()]
    temp_test = df.loc[df['Age'].isnull()]
    # print(temp_train)
    y = temp_train.Age.values
    x = temp_train.drop(['Name', 'PassengerId', 'Parch', 'SibSp', 'Ticket', 'Age', 'Survived'], axis=1).values
    rfr = RandomForestRegressor(n_estimators=1500, n_jobs=-1)
    rfr.fit(x, y)
    predicted_age = rfr.predict(temp_test.drop(['Name', 'PassengerId', 'Parch', 'SibSp', 'Ticket', 'Age', 'Survived'], axis=1).values)
    df.loc[df['Age'].isnull(), 'Age'] = predicted_age

    return df

fig, ax = plt.subplots()
g = sns.distplot(dataset.Age.dropna(), ax=ax)
ax.set_xlim(0,100)
ax.set_xticks(np.arange(0, 100, 5))
complete_age(dataset)
dataset['Age'] = dataset['Age'].astype(int)
g = sns.distplot(dataset.Age, ax=ax)
plt.show()

dataset.loc[dataset['Age'] <= 10, 'AgeBin'] = 0
dataset.loc[(dataset['Age'] > 10) & (dataset['Age'] <= 50), 'AgeBin'] = 1
dataset.loc[(dataset['Age'] > 50), 'AgeBin'] = 2

# print(dataset['Age'])
# g = sns.countplot(x='TicketBin', data=dataset, hue='Survived')
# plt.show()

# g = sns.heatmap(dataset.corr(), annot=True)
# plt.show()

train_df = dataset.iloc[:891, :]
test_df = dataset.iloc[891:, :]

train_df = train_df.drop(['Name', 'PassengerId', 'Parch', 'SibSp', 'Ticket', 'Age'], axis=1)
pass_id = test_df['PassengerId']
test_df = test_df.drop(['Name', 'PassengerId', 'Parch', 'SibSp', 'Ticket', 'Age'], axis=1)

X_train = train_df.drop('Survived', axis=1)
Y_train = train_df['Survived']

X_test = test_df.drop('Survived', axis=1)

# Random forest
# RFC = RandomForestClassifier()
# rf_param_grid = {"max_depth": [None],
#               "max_features": [1, 3, 6, 8],
#               "min_samples_split": [2, 3, 8],
#               "min_samples_leaf": [1, 3, 8],
#               "bootstrap": [False],
#               "n_estimators" :[100,300],
#               "criterion": ["gini"]}
# gsRFC = GridSearchCV(RFC, param_grid=rf_param_grid, scoring="accuracy", n_jobs=4, verbose=1)
# gsRFC.fit(X_train, Y_train)
# # print(gsRFC.best_score_)
# rfc_best = gsRFC.best_estimator_
# print(rfc_best)

# n_estimators = [140, 145, 150, 155, 160]
# max_depth = range(1, 10)
# criterions = ['gini', 'entropy']
# cv = StratifiedShuffleSplit(n_splits=10, test_size=.3, random_state=15)
# paramerters = {'n_estimators': n_estimators,
#                'max_depth': max_depth,
#                'criterion': criterions}
#
# grid = GridSearchCV(estimator=RandomForestClassifier(max_features='auto'), param_grid=paramerters, cv=cv, n_jobs=-1)
# grid.fit(X_train, Y_train)
# print (grid.best_score_)
# print (grid.best_params_)
# print (grid.best_estimator_)


# print(X_train.describe())
# print(Y_train.describe())
# print(train_df.head(100))
random_forest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                       max_depth=4, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=155,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
log = round(random_forest.score(X_train, Y_train)*100, 2)
print(log)
# #
submission = pd.DataFrame({
    "PassengerId": pass_id,
    "Survived": Y_pred.astype(int)
})

submission.to_csv("./submission5.csv", index=False)