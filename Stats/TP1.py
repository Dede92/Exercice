#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd 
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt 
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

## 1
# Affiche les donnees du fichier
data_df = pd.read_csv('Galton.txt', delimiter='\t')
# affichel es noms des colonnes
columns = data_df.columns
# Affiche les donnes de la colonne
family = data_df['Family']

## 2
# Ajoute une nouvelle colonne de donnee
data_df['MeanParents'] = pd.Series(0.5*(data_df['Father'] + 1.08 * data_df['Mother']), index = data_df.index)
# Affiche les donnees
# print data_df

## 3
# Cree un plot des donnees
# data_df.plot(x='MeanParents', y='Height', kind='scatter')
# plt.show()

#4
# http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
meanParents_X_train = []
for data in data_df['MeanParents']:
	meanParents_X_train.append([data])

height_Y_train = []
for data in data_df['Height']:
	height_Y_train.append([data])

# Create linear regression object
regr = linear_model.LinearRegression()
# Train the model using the training sets
regr.fit(meanParents_X_train, height_Y_train)

# 5
# The coefficients
coefYteta = regr.coef_
print('Coefficients: \n', coefYteta)
# Intercept
interceptYteta = regr.intercept_
print('Intercept: \n', interceptYteta)
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((regr.predict(meanParents_X_train) - height_Y_train) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(meanParents_X_train, height_Y_train))

# Plot outputs
plt.scatter(meanParents_X_train, height_Y_train,  color='black', label="Scatter")
plt.plot(meanParents_X_train, regr.predict(meanParents_X_train), color='blue', linewidth=3, label='Regression')

plt.xticks(())
plt.yticks(())

plt.legend()

plt.show()

plt.figure()
plt.plot(meanParents_X_train, regr.predict(meanParents_X_train), label="Prediction")
plt.plot(meanParents_X_train , meanParents_X_train*coefYteta + interceptYteta, label="Coef")
plt.legend()
plt.show()

# 6 Calcul des résidus
plt.figure()
yTeta = meanParents_X_train*coefYteta + interceptYteta
residus = height_Y_train - yTeta
# print residus
plt.hist(residus)
plt.show()
# le graphe n'est pas gaussien, l'hypothese n'est pas credible

# 7 
# Create linear regression object
regr2 = linear_model.LinearRegression()
# Train the model using the training sets
regr2.fit(height_Y_train, meanParents_X_train)

# The coefficients
alpha1 = regr2.coef_
print('Alpha coefficients: \n', alpha1)
# Intercept
alpha0 = regr2.intercept_
print('Alpha intercept: \n', alpha0)
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((regr2.predict(height_Y_train) - meanParents_X_train) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr2.score(height_Y_train, meanParents_X_train))

# Plot outputs
plt.scatter(height_Y_train, meanParents_X_train,  color='black', label="Scatter2")
plt.plot(height_Y_train, regr.predict(height_Y_train), color='blue', linewidth=3, label='Regression2')

plt.xticks(())
plt.yticks(())

plt.legend()

plt.show()

# verification numerique

var_x = data_df['MeanParents'].var(axis=1)
var_y = data_df['Height'].var(axis=1)
mean_x = data_df['MeanParents'].mean(axis=1)
mean_y = data_df['Height'].mean(axis=1)

alpha0_other = mean_x + (mean_y * var_x)*(interceptYteta-mean_y)/(mean_x * var_y)
print 'Alpha0 : ' + str(alpha0) + ' ' + 'Alpha0 other: ' + str(alpha0_other)

alpha1_other = var_x * coefYteta / var_y
print 'Alpha1 : ' + str(alpha1) + ' ' + 'Alpha1 other: ' + str(alpha1_other)

#8 
# b = X[['A','B']]
# b.shape => taille

FatherMother_train = data_df[['Father', 'Mother']]
# Create linear regression object
multipleRegression = linear_model.LinearRegression()
# Train the model using the training sets
multipleRegression.fit(FatherMother_train, height_Y_train)

# The coefficients
multipleCoef = multipleRegression.coef_
print('Alpha coefficients: \n', multipleCoef)
# Intercept
multipleIntercept = multipleRegression.intercept_
print('Alpha intercept: \n', multipleIntercept)

# 9

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(data_df['Father'], data_df['Mother'], data_df['Height'])


ax.set_xlabel('Father')
ax.set_ylabel('Mother')
ax.set_zlabel('Kid\'s Height')



XX = np.arange(50, 80, 0.5)
YY = np.arange(50, 80, 0.5)
xx, yy = np.meshgrid(XX, YY)
zz = multipleIntercept[0] + multipleCoef[0][0] * xx + multipleCoef[0][1] * yy

ax.plot_wireframe(xx, yy, zz, rstride=10, cstride=10, alpha=0.3)

plt.show()

#  la covariance est liee à la correlation des variables, plus la cov est grande plus la cov est grande