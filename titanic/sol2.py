#/usr/local/bin/python3
# data analysis and wrangling
import pandas as pd
import numpy as np
import random

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
import graphviz
# from pandas.tools.plotting import scatter_matrix


data_raw = pd.read_csv('./input/train.csv')
data_val = pd.read_csv('./input/test.csv')

data1 = data_raw.copy(deep = True)

data_cleaner = [data1, data_val]

# print(data_val.head(10))
# print(data1.isnull().sum())
# print(data_val.isnull().sum())

for dataset in data_cleaner:
	dataset['Age'].fillna(dataset['Age'].median(), inplace=True)
	dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)
	dataset['Fare'].fillna(dataset['Fare'].median(), inplace=True)

drop_column = ['PassengerId', "Cabin", "Ticket"]
data1.drop(drop_column, axis=1, inplace=True)

# print(data1.info())

for dataset in data_cleaner:
	dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch']  + 1
	dataset['IsAlone'] = 1
	dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0

	dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
	dataset['FareBin'] = pd.qcut(dataset['Fare'].astype(int), 4)
	dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)


magic_number = 10


titles_name = (data1['Title'].value_counts() < magic_number)
# print(titles_name)
data1['Title'] = data1['Title'].apply(lambda x: 'Misc' if titles_name.loc[x] == True else x)
# print(data1['Title'].value_counts())
# print(data1['AgeBin'])


label = LabelEncoder()

for dataset in data_cleaner:
	dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])
	dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked'])
	dataset['Title_Code'] = label.fit_transform(dataset['Title'])
	dataset['AgeBin_Code'] = label.fit_transform(dataset['AgeBin'])
	dataset['FareBin_Code'] = label.fit_transform(dataset['FareBin'])


Target = ['Survived']

data1_x = ['Sex','Pclass', 'Embarked', 'Title','SibSp', 'Parch', 'Age', 'Fare', 'FamilySize', 'IsAlone']
data1_x_calc = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code','SibSp', 'Parch', 'Age', 'Fare']

data1_x_bin = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code']

data1_dummy = pd.get_dummies(data1[data1_x])
data1_x_dummy = data1_dummy.columns.tolist()

# print(data1_dummy.columns)

# print(data1.info())
# print(data_val.info())

train1_x, test1_x, train1_y, test1_y = model_selection.train_test_split(data1[data1_x_calc], data1[Target], random_state=0)
train1_x_bin, test1_x_bin, train1_y_bin, test1_y_bin = model_selection.train_test_split(data1[data1_x_bin], data1[Target] , random_state = 0)
train1_x_dummy, test1_x_dummy, train1_y_dummy, test1_y_dummy = model_selection.train_test_split(data1_dummy[data1_x_dummy], data1[Target], random_state = 0)


# print(train1_x_bin)

# for x in data1_x:
# 	if data1[x].dtype != 'float64':
# 		print("Survival correlation by: ", x)
# 		print(data1[[x, Target[0]]].groupby(x, as_index = False).mean())
# 		print("-"*10)


# print(data1['Sex'].value_counts())
# h = sns.FacetGrid(data1, row = 'Sex', col = 'Pclass', hue='Survived')
# h.map(plt.hist, 'Age', alpha = .75)
# h.add_legend()
# plt.show()


# def heat_map(df):
# 	_, ax = plt.subplots(figsize=(14, 12))
# 	color_map = sns.diverging_palette(220, 10, as_cmap=True)
# 	_ = sns.heatmap(
# 			df.corr(),
# 			cmap = color_map,
# 			square=True,
# 			cbar_kws={'shrink': .9},
# 			ax=ax,
# 			annot=True,
# 			linewidths=0.1, vmax=0.1, linecolor='white',
# 			annot_kws={'fontsize': 8}
# 		)

# 	plt.show()

# heat_map(data1)

# for index, row in data1.iterrows():
# 	if random.random() > .5:
# 		data1.set_value(index, 'Random_predict', 1)
# 	else:
# 		data1.set_value(index, 'Random_predict', 0)
# data1['Random_score'] = 0

# data1.loc[(data1['Survived'] == data1['Random_predict']), 'Random_score'] = 1

# print(data1['Random_score'].mean()*100)

# pivot_table = data1[data1.Sex=='female'].groupby(['Sex', 'Pclass', 'Embarked', 'FareBin'])['Survived'].mean()
# print(pivot_table)



def tree(df):
	Model = pd.DataFrame(data = {'Predict': []})
	male_title = ['Master']
	for index, row in df.iterrows():
		Model.loc[index, 'Predict'] = 0

		if (df.loc[index, 'Sex'] == 'female'):
			Model.loc[index, 'Predict'] = 1
		if (df.loc[index, 'Sex'] == 'female' and
			(df.loc[index, 'Pclass'] == 3) and
			(df.loc[index, 'Embarked'] == 'S') and
			(df.loc[index, 'Fare'] > 8) ):
			Model.loc[index, 'Predict'] = 0
		if ((df.loc[index, 'Sex'] == 'female') and
			(df.loc[index, 'Title'] in male_title)):
			Model.loc[index, 'Predict'] = 1
	return Model;

# tree_predict = tree(data1)
# print(metrics.accuracy_score(data1['Survived'], tree_predict))
cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0)
# dtree = tree.DecisionTreeClassifier(random_state = 0)
# base_results = model_selection.cross_validate(dtree, data1[data1_x_bin], data1[Target], cv = cv_split)

#Origin D-Tree
# dtree.fit(data1[data1_x_bin], data1[Target])
# print("DTREE Parameters: ", dtree.get_params())
# print("DTREE Training w/bin score mean: {:.2f}", format(base_results['train_score'].mean()))
# print("DTREE Test w/bin score mean: {:.2f}", format(base_results['test_score'].mean()))
# print("OLD DTREE Columns: ", data1[data1_x_bin].columns.values)
# print("-"*10)

# tune by hyper-parameters
# param_grid = {
# 	'criterion': ['entropy', 'gini'],
# 	'max_depth': [2, 4, 6, 8, 10, None],
# 	'random_state': [0],

# }

# tune_model = model_selection.GridSearchCV(tree.DecisionTreeClassifier(), param_grid=param_grid, scoring='roc_auc', cv = cv_split)
# tune_model.fit(data1[data1_x_bin], data1[Target])


# print("AFTER DTREE Parameters: ", tune_model.best_params_)
# print("AFTER DTREE Training w/bin score mean: {:.2f}", format(tune_model.cv_results_['mean_train_score'][tune_model.best_index_]))
# print("AFTER DTREE Test w/bin score mean: {:.2f}", format(tune_model.cv_results_['mean_test_score'][tune_model.best_index_]))

# print("-"*10)

#feature selection
# dtree_rfe = feature_selection.RFECV(dtree, step = 1, scoring = 'accuracy', cv = cv_split)
# dtree_rfe.fit(data1[data1_x_bin], data1[Target])

# X_rfe = data1[data1_x_bin].columns.values[dtree_rfe.get_support()]
# rfe_results = model_selection.cross_validate(dtree, data1[X_rfe], data1[Target], cv = cv_split)

# print("RFE_TREE Parameters: ", dtree.get_params())
# print("RFE_TREE Training w/bin score mean: {:.2f}", format(rfe_results['train_score'].mean()))
# print("RFE_TREE Test w/bin score mean: {:.2f}", format(rfe_results['test_score'].mean()))
# print("NEW RFE_TREE Columns: ", X_rfe)

# print("-"*10)

#rfe_tune_model

# param_grid = {
# 	'criterion': ['entropy', 'gini'],
# 	'max_depth': [2, 4, 6, 8, 10, None],
# 	'random_state': [0],

# }

# RFE TUNE MODEL
# rfe_tune_model = model_selection.GridSearchCV(tree.DecisionTreeClassifier(), param_grid=param_grid, scoring='roc_auc', cv = cv_split)
# rfe_tune_model.fit(data1[X_rfe], data1[Target])


# print("RFE_TUNE_TREE Parameters: ", rfe_tune_model.best_params_)
# print("RFE_TUNE_TREE Training w/bin score mean: {:.2f}", format(rfe_tune_model.cv_results_['mean_train_score'][rfe_tune_model.best_index_]))
# print("RFE_TUNE_TREE Test w/bin score mean: {:.2f}", format(rfe_tune_model.cv_results_['mean_test_score'][rfe_tune_model.best_index_]))


# dot_data = tree.export_graphviz(dtree, out_file=None, feature_names = data1_x_bin, class_names = True, filled=True, rounded=True)
# graph = graphviz.Source(dot_data)
# print(graph.render())

# vote_est = [
#     #Ensemble Methods: http://scikit-learn.org/stable/modules/ensemble.html
#     ('ada', ensemble.AdaBoostClassifier()),
#     ('bc', ensemble.BaggingClassifier()),
#     ('etc',ensemble.ExtraTreesClassifier()),
#     ('gbc', ensemble.GradientBoostingClassifier()),
#     ('rfc', ensemble.RandomForestClassifier()),

#     #Gaussian Processes: http://scikit-learn.org/stable/modules/gaussian_process.html#gaussian-process-classification-gpc
#     ('gpc', gaussian_process.GaussianProcessClassifier()),
    
#     #GLM: http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
#     ('lr', linear_model.LogisticRegressionCV()),
    
#     #Navies Bayes: http://scikit-learn.org/stable/modules/naive_bayes.html
#     ('bnb', naive_bayes.BernoulliNB()),
#     ('gnb', naive_bayes.GaussianNB()),
    
#     #Nearest Neighbor: http://scikit-learn.org/stable/modules/neighbors.html
#     ('knn', neighbors.KNeighborsClassifier()),
    
#     #SVM: http://scikit-learn.org/stable/modules/svm.html
#     ('svc', svm.SVC(probability=True)),
    
#     #xgboost: http://xgboost.readthedocs.io/en/latest/model.html
#    ('xgb', XGBClassifier())

# ]

# vote_hard = ensemble.VotingClassifier(estimators = vote_est, voting='hard') 
# vote_hard_cv = model_selection.cross_validate(vote_hard, data1[data1_x_bin], data1[Target], cv = cv_split)
# vote_hard.fit(data1[data1_x_bin], data1[Target])

# # print("VOTING_CLASSIFIER Parameters: ", dtree.get_params())
# # print("VOTING_CLASSIFIER Training w/bin score mean: {:.2f}", format(vote_hard_cv['train_score'].mean()))
# print("HARD_VOTING_CLASSIFIER Test w/bin score mean: {:.2f}", format(vote_hard_cv['test_score'].mean()))


# vote_soft = ensemble.VotingClassifier(estimators = vote_est, voting='soft') 
# vote_soft_cv = model_selection.cross_validate(vote_soft, data1[data1_x_bin], data1[Target], cv = cv_split)
# vote_soft.fit(data1[data1_x_bin], data1[Target])

# # print("VOTING_CLASSIFIER Parameters: ", dtree.get_params())
# # print("VOTING_CLASSIFIER Training w/bin score mean: {:.2f}", format(vote_hard_cv['train_score'].mean()))
# print("SOFT_VOTING_CLASSIFIER Test w/bin score mean: {:.2f}", format(vote_soft_cv['test_score'].mean()))


data_val['Survived'] = tree(data_val).astype(int)
submit = data_val[['PassengerId', 'Survived']]
submit.to_csv("submission2.csv", index=False)
print(submit.info())
