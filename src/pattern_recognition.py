from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

# Import data for digits recognition
digits = load_digits()
data = scale(digits.data)
labels = digits.target
print(data.shape)
print(labels.shape)

X_train, X_test, Y_train, Y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)

# Classifier representation
classifier = {
    "knn": {
        "accuracy": 0.0,
        "recall": 0.0,
        "cm": []
    },
    "sbd": {
        "accuracy": 0.0,
        "recall": 0.0,
        "cm": []
    },
    "dt": {
        "accuracy": 0.0,
        "recall": 0.0,
        "cm": []
    }
}

# KNN Classifier
count = 0
nob = 15
predictions = []
accuracies_knn = []
recalls = []
for i in range(1, nob):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, Y_train)

    # Predict Output
    predicted = knn.predict(X_test)

    # Accuracy Score of KNN
    a = accuracy_score(Y_test, predicted)
    recall = recall_score(Y_test, predicted, average='macro')
    accuracies_knn.append(a)
    recalls.append(recall)
    predictions.append(predicted)

    print('Accuracy Score for KNN with number of neighbours {} is {}: \n'.format(i, a))

print('Best accuracy:', accuracies_knn)
knn_classifier = classifier['knn']
knn_classifier['accuracy'] = max(accuracies_knn)
knn_classifier['recall'] = recalls[accuracies_knn.index(max(accuracies_knn))]

# Confusion Matrix of KNN
cm_knn = confusion_matrix(Y_test, predictions[accuracies_knn.index(max(accuracies_knn))])
knn_classifier['cm'] = cm_knn

# SGD Classifier
sbd = SGDClassifier(loss="hinge", penalty="l2", max_iter=100, random_state=0)
sbd.fit(X_train, Y_train)
prediction_sbd = sbd.predict(X_test)
sbd_classifier = classifier['sbd']

# Accuracy Score of Stochastic Gradient Descent
acc_sbd = accuracy_score(Y_test, prediction_sbd)
print("Accuracy Score  of  SBD is: {}".format(acc_sbd))
sbd_classifier['accuracy'] = acc_sbd

# Recall for SBD
recall_sbd = recall_score(Y_test, prediction_sbd, average='macro')
sbd_classifier['recall'] = recall_sbd

# Confusion Matrix of SBD
cm_sbd = confusion_matrix(Y_test, prediction_sbd)
sbd_classifier['cm'] = cm_sbd

# Decision Tree Classifier
dt = DecisionTreeClassifier(max_depth=15, random_state=0, criterion="gini")
dt.fit(X_train, Y_train)
prediction_dt = dt.predict(X_test)
dt_classifier = classifier['dt']

# Accuracy Score of Decision Tree
acc_dt = accuracy_score(Y_test, prediction_dt)
print("Accuracy Score  of  SBD is: {}".format(acc_dt))
dt_classifier['accuracy'] = acc_dt

# Recall for DT
recall_dt = recall_score(Y_test, prediction_dt, average='macro')
dt_classifier['recall'] = recall_dt

# Confusion Matrix of DT
cm_dt = confusion_matrix(Y_test, prediction_dt)
dt_classifier['cm'] = cm_dt
print(classifier)

print('COMP9517 Week 5 Lab - z5222766')
print()

# Results
print('Test size: {}'.format(0.25))
print('KNN Accuracy: {}       Recall: {}'.format(classifier['knn']['accuracy'], classifier['knn']['recall']))
print('SBD Accuracy: {}       Recall: {}'.format(classifier['sbd']['accuracy'], classifier['sbd']['recall']))
print('DT Accuracy: {}        Recall: {}'.format(classifier['dt']['accuracy'], classifier['dt']['recall']))

max_accuracy_classifier = max(classifier.keys(), key=(lambda k: classifier[k]['accuracy']))
print('Confusion matrix of the best classifier is {}'.format(max_accuracy_classifier))
print(classifier[max_accuracy_classifier]['cm'])
