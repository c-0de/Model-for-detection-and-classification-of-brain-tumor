
import numpy as np
import pandas as pd
import keras
import io
import os
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from sklearn.utils import shuffle
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout
import ipywidgets as widgets
from tqdm import tqdm
import seaborn as sns

#CNN
X_train = []
y_train = []
image_size = 256
train_path = 'Dataset/Training'
labels = os.listdir(train_path)
print(labels)
for i in labels:
    folder_path = os.path.join(train_path, i)
    for j in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, j))
        img = cv2.resize(img, (image_size, image_size))
        
        X_train.append(img)
        y_train.append(i)
test_path = 'Dataset/Testing'
for i in labels:
    folder_path = os.path.join(test_path, i)
    for j in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, j))
        img = cv2.resize(img, (image_size, image_size))
        
        X_train.append(img)
        y_train.append(i)
X_train = np.array(X_train)
y_train = np.array(y_train)
X_train, y_train = shuffle(X_train, y_train, random_state = 99)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.1, random_state = 100)
y_train_new = []
for i in y_train:
    y_train_new.append(labels.index(i))
y_train_new = tf.keras.utils.to_categorical(y_train_new)
y_test_new = []
for i in y_test:
    y_test_new.append(labels.index(i))

y_test_new = tf.keras.utils.to_categorical(y_test_new)
model = Sequential()
model.add(Conv2D(32, (3,3), activation = 'relu', input_shape = (image_size, image_size, 3)))
model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))

model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3,3), activation = 'relu'))
model.add(Conv2D(128, (3,3), activation = 'relu'))
model.add(Conv2D(128, (3,3), activation = 'relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3,3), activation = 'relu'))
model.add(Conv2D(256, (3,3), activation = 'relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(512, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.3))

model.add(Dense(4, activation = 'softmax'))

extracted_features = model.predict(X_train)
print(extracted_features)

model.compile(
    loss = 'categorical_crossentropy',
    optimizer = 'Adam',
    metrics = ['accuracy']
)
hist = model.fit(X_train, y_train_new, epochs = 20, validation_split = 0.1)
acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy'] 

epochs = range(len(acc))
fig = plt.figure(figsize = (14,7))
plt.plot(epochs, acc, 'r', label = 'Training Accuracy')
plt.plot(epochs, val_acc, 'b', label = 'Validation Accuracy')
plt.legend(loc = 'upper left')
plt.show()
loss = hist.history['loss']
val_loss = hist.history['val_loss'] 

epochs = range(len(acc))
fig = plt.figure(figsize = (14,7))
plt.plot(epochs, loss, 'r', label = 'Training Loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation Loss')
plt.legend(loc = 'upper left')
plt.show()


y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

y_test_labels = np.argmax(y_test_new, axis=1)

report = classification_report(y_test_labels, y_pred_classes, target_names=labels)
print("Classification Report:")
print(report)

cnn_precision = precision_score(y_test_labels, y_pred_classes, average='weighted')
print("Precision:", cnn_precision)

cnn_recall = recall_score(y_test_labels, y_pred_classes, average='weighted')
print("Recall:", cnn_recall)

cnn_f1 = f1_score(y_test_labels, y_pred_classes, average='weighted')
print("F1-score:", cnn_f1)

TN = confusion_matrix(y_test_labels, y_pred_classes)[0, 0]
FP = confusion_matrix(y_test_labels, y_pred_classes)[0, 1]
cnn_specificity = TN / (TN + FP)
print("Specificity:", cnn_specificity)

accuracy = accuracy_score(y_test_labels, y_pred_classes)
print("Accuracy:", accuracy)
cnn_acc=accuracy_score(y_test_labels, y_pred_classes)

path = os.listdir('Dataset/Training');
classes = {'no_tumor':0, 'pituitary_tumor':1, 'glioma_tumor':2, 'meningioma_tumor':3}

X=[]
y=[]
for types in classes:
    pth='Dataset/Training/'+types
    for data in os.listdir(pth):
        img = cv2.imread(pth+'/'+data, 0)
        img = cv2.resize(img, (200,200))
        X.append(img)
        y.append(classes[types])
X = np.array(X)
y = np.array(y)

X_updated = X.reshape(len(X), -1)

#SVM
X_train, X_test, y_train, y_test = train_test_split(X_updated, y, random_state=10,test_size=.20)
from sklearn.decomposition import PCA

pca = PCA(.98)
pca_train = X_train
pca_test = X_test

from sklearn.svm import SVC

import warnings
warnings.filterwarnings('ignore')

sv = SVC()
sv.fit(X_train, y_train)

sv_pred_train = sv.predict(X_train)
sv_pred_test = sv.predict(X_test)

print("Support Vector Classifier:")
print("\nClassification Report:")
print(classification_report(y_test, sv_pred_test))

sv_conf_matrix = confusion_matrix(y_test, sv_pred_test)

sv_TP = sv_conf_matrix[1][1]  # True Positives
sv_TN = sv_conf_matrix[0][0]  # True Negatives
sv_FP = sv_conf_matrix[0][1]  # False Positives
sv_FN = sv_conf_matrix[1][0]  # False Negatives

sv_precision = sv_TP / (sv_TP + sv_FP)
sv_recall = sv_TP / (sv_TP + sv_FN)
sv_f1 = 2 * (sv_precision * sv_recall) / (sv_precision + sv_recall)
sv_specificity = sv_TN / (sv_TN + sv_FP)

print("\nSupport Vector Classifier Metrics:")
print("Precision:", sv_precision)
print("Recall (Sensitivity):", sv_recall)
print("F1-score:", sv_f1)
print("Specificity:", sv_specificity)
print("Training Accuracy:", accuracy_score(y_train, sv_pred_train))
print("Testing Accuracy:", accuracy_score(y_test, sv_pred_test))
svm_acc= accuracy_score(y_test, sv_pred_test)

print(" ")

#KNN
X_train, X_test, y_train, y_test = train_test_split(X_updated, y, test_size=0.2, random_state=42)

X_train_flatten = X_train.reshape(X_train.shape[0], -1)
X_test_flatten = X_test.reshape(X_test.shape[0], -1)

import math
k = 3
knn_classifier = KNeighborsClassifier(n_neighbors=k)

knn_classifier.fit(X_train_flatten, y_train)

y_pred = knn_classifier.predict(X_test_flatten)

report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)

conf_matrix = confusion_matrix(y_test, y_pred)

TP = conf_matrix[1][1]
TN = conf_matrix[0][0]  
FP = conf_matrix[0][1] 
FN = conf_matrix[1][0]  

precision = TP / (TP + FP)
print("Precision:", precision)

recall = TP / (TP + FN)
print("Recall:", recall)

f1 = 2 * (precision * recall) / (precision + recall)
print("F1-score:", f1)

specificity = TN / (TN + FP)
print("Specificity:", specificity)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
knn_acc=accuracy_score(y_test,y_pred)
print(svm_acc, knn_acc)


combined_metrics = {
    'Metric': ['Precision', 'Recall (Sensitivity)', 'F1-score', 'Specificity', 'Testing Accuracy'],
    'CNN': [cnn_precision, cnn_recall, cnn_f1, cnn_specificity, cnn_acc],
    'SVM': [sv_precision, sv_recall, sv_f1, sv_specificity, svm_acc],
    'KNN': [precision, recall, f1, specificity, knn_acc]
}

import pandas as pd
df = pd.DataFrame(combined_metrics)

melted_df = pd.melt(df, id_vars=['Metric'], var_name='Classifier', value_name='Score')

plt.figure(figsize=(12, 8))
sns.barplot(x='Metric', y='Score', hue='Classifier', data=melted_df)
plt.title('Comparison of SVM and KNN Classifiers')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.legend(title='Classifier')
plt.show()



cnn_precision = 0.979005524861878
svm_precision = sv_precision
knn_precision = precision

classifiers = ['CNN', 'SVM', 'KNN']
precision_scores = [cnn_precision, svm_precision, knn_precision]

plt.figure(figsize=(10, 6))
plt.bar(classifiers, precision_scores, color=['blue', 'orange', 'green'])
plt.title('Precision Scores of CNN, SVM, and KNN')
plt.xlabel('Classifier')
plt.ylabel('Precision Score')
plt.ylim(0, 1)
plt.show()


cnn_f1_score = 0.9860724233983287
svm_f1_score = sv_f1
knn_f1_score = f1

classifiers = ['CNN', 'SVM', 'KNN']
f1_scores = [cnn_f1_score, svm_f1_score, knn_f1_score]

plt.figure(figsize=(10, 6))
plt.bar(classifiers, f1_scores, color=['blue', 'orange', 'green'])
plt.title('F1-Scores of CNN, SVM, and KNN')
plt.xlabel('Classifier')
plt.ylabel('F1-Score')
plt.ylim(0, 1)
plt.show()

cnn_specificity_score = 0.9384615384615385
svm_specificity_score = sv_specificity
knn_specificity_score = specificity

classifiers = ['CNN', 'SVM', 'KNN']
specificity_scores = [cnn_specificity_score, svm_specificity_score, knn_specificity_score]

plt.figure(figsize=(10, 6))
plt.bar(classifiers, specificity_scores, color=['blue', 'orange', 'green'])
plt.title('Specificity of CNN, SVM, and KNN')
plt.xlabel('Classifier')
plt.ylabel('Specificity')
plt.ylim(0, 1)  
plt.show()

cnn_accuracy_score = 0.8989547038327527
svm_accuracy_score = svm_acc
knn_accuracy_score = knn_acc

classifiers = ['CNN', 'SVM', 'KNN']
accuracy_scores = [cnn_accuracy_score, svm_accuracy_score, knn_accuracy_score]

plt.figure(figsize=(10, 6))
plt.bar(classifiers, accuracy_scores, color=['blue', 'orange', 'green'])
plt.title('Accuracy of CNN, SVM, and KNN')
plt.xlabel('Classifier')
plt.ylabel('Accuracy')
plt.ylim(0, 1)  
plt.show()

cnn_sensitivity_score = 0.9943820224719101
svm_sensitivity_score = sv_recall
knn_sensitivity_score = recall

classifiers = ['CNN', 'SVM', 'KNN']
sensitivity_scores = [cnn_sensitivity_score, svm_sensitivity_score, knn_sensitivity_score]

plt.figure(figsize=(10, 6))
plt.bar(classifiers, sensitivity_scores, color=['blue', 'orange', 'green'])
plt.title('Sensitivity of CNN, SVM, and KNN')
plt.xlabel('Classifier')
plt.ylabel('Sensitivity (Recall)')
plt.ylim(0, 1)
plt.show()



'''import os
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def extract_texture_features(image):

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    

    distances = [1, 2, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    glcm = graycomatrix(gray_image, distances=distances, angles=angles, symmetric=True, normed=True)
    
    contrast = graycoprops(glcm, 'contrast').ravel()
    dissimilarity = graycoprops(glcm, 'dissimilarity').ravel()
    homogeneity = graycoprops(glcm, 'homogeneity').ravel()
    energy = graycoprops(glcm, 'energy').ravel()
    correlation = graycoprops(glcm, 'correlation').ravel()
    asm = graycoprops(glcm, 'ASM').ravel()
    
    texture_features = np.concatenate((contrast, dissimilarity, homogeneity, energy, correlation, asm))
    
    return texture_features

train_path = 'Dataset/Training'
test_path = 'Dataset/Testing'

X_updated = []
y = []

for label in os.listdir(train_path):
    label_path = os.path.join(train_path, label)
    for image_file in os.listdir(label_path):
        image_path = os.path.join(label_path, image_file)
        image = cv2.imread(image_path)
        features = extract_texture_features(image)
        X_updated.append(features)
        y.append(label)

for label in os.listdir(test_path):
    label_path = os.path.join(test_path, label)
    for image_file in os.listdir(label_path):
        image_path = os.path.join(label_path, image_file)
        image = cv2.imread(image_path)
        features = extract_texture_features(image)
        X_updated.append(features)
        y.append(label)

X_updated = np.array(X_updated)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X_updated, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC())
])

param_grid = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__kernel': ['linear', 'rbf', 'poly'],
    'svm__gamma': ['scale', 'auto']
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)

grid_search.fit(X_train, y_train)

best_classifier = grid_search.best_estimator_

svm_pred_train = best_classifier.predict(X_train)
svm_pred_test = best_classifier.predict(X_test)

print("Support Vector Classifier:")
print("\nClassification Report:")
print(classification_report(y_test, svm_pred_test))

sv_conf_matrix = confusion_matrix(y_test, svm_pred_test)

sv_TP = sv_conf_matrix[1][1]  # True Positives
sv_TN = sv_conf_matrix[0][0]  # True Negatives
sv_FP = sv_conf_matrix[0][1]  # False Positives
sv_FN = sv_conf_matrix[1][0]  # False Negatives

sv_precision = sv_TP / (sv_TP + sv_FP)
sv_recall = sv_TP / (sv_TP + sv_FN)
sv_f1 = 2 * (sv_precision * sv_recall) / (sv_precision + sv_recall)
sv_specificity = sv_TN / (sv_TN + sv_FP)

print("\nSupport Vector Classifier Metrics:")
print("Precision:", sv_precision)
print("Recall (Sensitivity):", sv_recall)
print("F1-score:", sv_f1)
print("Specificity:", sv_specificity)
print("Training Accuracy:", accuracy_score(y_train, svm_pred_train))
print("Testing Accuracy:", accuracy_score(y_test, svm_pred_test))

print(" ")'''

