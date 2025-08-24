# import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


# loading the dataset
df = pd.read_csv('dataset/Iris.csv')

# dropping ID column
df.drop(columns = ['Id'], inplace = True)

# previewing dataset
print('Dataset Preview:- ')
print(df.head())

# prepare features and labels
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# encoding labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# normalize features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# split data
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_encoded, test_size = 0.2, random_state = 42)

# train and evaluate for different K
k_values = range(1, 16)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    accuracies.append(accuracy_score(y_test, y_pred))
    print(f'K = {k}, Accuracy = {accuracies[-1]:.4f}')

# plot accuracy vs. K
plt.figure(figsize = (8, 5))
plt.plot(k_values, accuracies, marker = 'o')
plt.title('K vs. Accuracy')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Accuracy')
plt.grid()
plt.show()

# Best K
best_k = k_values[np.argmax(accuracies)]
print(f'Best K Values: {best_k}')

# train with best K
final_knn = KNeighborsClassifier(n_neighbors = best_k)
final_knn.fit(x_train, y_train)
y_pred = final_knn.predict(x_test)

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = label_encoder.classes_)
disp.plot(cmap = 'Blues')
plt.title('Confusion Matrix')
plt.show()

# Decision boundary visualization (first two features)
x_plot = x_scaled[:, :2]  # Only first two features
x_train_p, x_test_p, y_train_p, y_test_p = train_test_split(
    x_plot, y_encoded, test_size=0.2, random_state=42
)

knn_plot = KNeighborsClassifier(n_neighbors=best_k)
knn_plot.fit(x_train_p, y_train_p)

x_min, x_max = x_plot[:, 0].min() - 1, x_plot[:, 0].max() + 1
y_min, y_max = x_plot[:, 1].min() - 1, x_plot[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))
Z = knn_plot.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Set2)
plt.scatter(x_plot[:, 0], x_plot[:, 1], c=y_encoded, cmap=plt.cm.Set1, edgecolor='k', s=50)
plt.xlabel(df.columns[0])
plt.ylabel(df.columns[1])
plt.title(f'KNN Decision Boundary (k = {best_k})')
plt.show()
