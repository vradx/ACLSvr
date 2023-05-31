from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import sklearn.metrics
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from imblearn.pipeline import make_pipeline
from skopt import BayesSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek


import datetime
from tqdm import tqdm


def SVM_classifier(iterations, Xtrain, Ytrain, Xtest):

    params = {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
        'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
    }
    new_params = {'svc__' + key: params[key] for key in params}
    kf = KFold(n_splits=10, shuffle=False)
    # imba_pipeline = make_pipeline(SVC())
    # imba_pipeline = make_pipeline(SMOTE(random_state=0), SVC()) # Upsample minority class
    imba_pipeline = make_pipeline(SMOTETomek(random_state=0), SVC())  # Upsample minority class & Downsample majority
    optimizer = BayesSearchCV(
        imba_pipeline,
        search_spaces=new_params,
        n_iter=iterations,
        cv=kf,
        n_jobs=-1,
        scoring='f1',
        verbose=0,
        random_state=0,
    )
    print('\nTotal iterations are: ', optimizer.total_iterations)
    print('\nThe software was launched at: ', startTime)
    for _ in tqdm(range(iterations)): # Progress bar
        optimizer.fit(Xtrain, Ytrain)
    Ypred = optimizer.predict(Xtest)
    print("\nVal. accuracy is: %s" % optimizer.best_score_)
    print("\nThe best parameters are: %s" % optimizer.best_params_)

    return (Ypred)


startTime = datetime.datetime.now()

# Set prints larger
desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 10)  # 5 features + 1 target label atm

# File Fetch
df = pd.read_csv(r'D:\Dataset\_4_FOURTH PREPROCESSING\_Task_1-7_removeRows_0726_1225\feature_df.csv', header=0, dtype=np.float64)
# df = pd.read_csv(r'D:\Dataset\_SECOND PREPROCESSING\_ZERO_PADDING\Task_Healthy_vs_ALL_intoClass3_1500\feature_df.csv', header=0, dtype=np.float64)
# print(df.head())


# Divide labels from Features
labelsTable = df.iloc[:, -1]

for i in range(len(labelsTable)):
    if labelsTable[i] == 1:
        labelsTable[i] = 0
    elif labelsTable[i] != 1:
        labelsTable[i] = 1


# Drop the labels column and the first row
featTable = df
featTable = featTable.drop(featTable.columns[-1], axis=1)

col_names = df.columns
print(col_names)

# EDA
df.info()
# Double check for nulls
print(df.isnull().sum())
# How many sounds are adventitious?
print(df['Labels'].value_counts())
# How many sounds are adventitious in percentage?
print(df['Labels'].value_counts() / float(len(df)))
# Outliers
print(round(df.describe(), 6))


# Visualize outliers for base df
plt.figure(figsize=(24, 20))

plt.subplot(4, 4, 1)
fig = df.boxplot(column='Variance')
fig.set_title('')
fig.set_ylabel('Variance ')

plt.subplot(4, 4, 2)
fig = df.boxplot(column='SMA_Coarse')
fig.set_title('')
fig.set_ylabel('SMA_Coarse')

plt.subplot(4, 4, 3)
fig = df.boxplot(column='Range')
fig.set_title('')
fig.set_ylabel('Range')

plt.subplot(4, 4, 4)
fig = df.boxplot(column='Spectrum_Mean')
fig.set_title('')
fig.set_ylabel('Spectrum_Mean')
plt.show()

plt.subplot(4, 4, 5)
fig = df.boxplot(column='SMA_Fine')
fig.set_title('')
fig.set_ylabel('SMA_Fine')

plt.subplot(4, 4, 6)
fig = df.boxplot(column='ZCR')
fig.set_title('')
fig.set_ylabel('ZCR')

plt.subplot(4, 4, 7)
fig = df.boxplot(column='Spec_Kurtosis')
fig.set_title('')
fig.set_ylabel('Spec_Kurtosis')

plt.subplot(4, 4, 8)
fig = df.boxplot(column='RMS')
fig.set_title('')
fig.set_ylabel('RMS')

plt.show()


# Visualize Distributions for base df
plt.figure(figsize=(24, 20))

plt.subplot(4, 4, 1)
fig = df['Variance'].hist(bins=100)
fig.set_xlabel('Variance')
fig.set_ylabel('Variance Count')

plt.subplot(4, 4, 2)
fig = df['SMA_Coarse'].hist(bins=100)
fig.set_xlabel('SMA_Coarse')
fig.set_ylabel('SMA_Coarse Count')

plt.subplot(4, 4, 3)
fig = df['Range'].hist(bins=100)
fig.set_xlabel('Range')
fig.set_ylabel('Range Count')

plt.subplot(4, 4, 4)
fig = df['Spectrum_Mean'].hist(bins=100)
fig.set_xlabel('Spectrum_Mean')
fig.set_ylabel('Spectrum_Mean Count')

plt.subplot(4, 4, 5)
fig = df['SMA_Fine'].hist(bins=100)
fig.set_xlabel('SMA_Fine')
fig.set_ylabel('SMA_Fine Count')

plt.subplot(4, 4, 6)
fig = df['SMA_Fine'].hist(bins=100)
fig.set_xlabel('ZCR')
fig.set_ylabel('ZCR Count')

plt.subplot(4, 4, 7)
fig = df['SMA_Fine'].hist(bins=100)
fig.set_xlabel('Spec_Kurtosis')
fig.set_ylabel('Spec_Kurtosis Count')

plt.subplot(4, 4, 8)
fig = df['SMA_Fine'].hist(bins=100)
fig.set_xlabel('RMS')
fig.set_ylabel('RMS Count')

plt.show()


# CAREFUL IT'S HEAVY TO LOAD
g = sns.pairplot(df, hue='Labels', vars=['Variance', 'SMA_Coarse', 'Range',
                                         'Spectrum_Mean', 'SMA_Fine', 'ZCR',
                                         'Spec_Kurtosis', 'RMS'])

# Remove the numbers from axis labels
g.set(xticklabels=[], yticklabels=[])

# Show the plot
plt.show()


# Correlation map
plt.figure(figsize=(20, 10))
sns.set(font_scale=2)
sns.heatmap(df.corr(), annot=True)
plt.show()


# # Train/Test partitioning
X_train, X_test, y_train, y_test = train_test_split(featTable, labelsTable, test_size=0.2, random_state=5)

# Necessary for the pipeline
y_train = y_train.astype(np.int32)
y_train = np.array((y_train))

# Feature Scaling
cols = X_train.columns
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])
print(X_train.describe())

# Function Call
y_pred = SVM_classifier(10, X_train, y_train, X_test)


y_test.value_counts()

confusion = confusion_matrix(y_test, y_pred)
print('Confusion Matrix\n')
print(confusion)


print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_test, y_pred)))

print('Micro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='micro')))
print('Micro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='micro')))
print('Micro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='micro')))

print('Macro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='macro')))
print('Macro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='macro')))
print('Macro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='macro')))

print('Weighted Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='weighted')))
print('Weighted Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='weighted')))
print('Weighted F1-score: {:.2f}'.format(f1_score(y_test, y_pred, average='weighted')))

# print('\nClassification Report\n')
# print(classification_report(y_test, y_pred, target_names=['1', '3']))
# print(classification_report(y_test, y_pred, target_names=['0', '1', '2', '3', '4', '5', '6', '7']))


# check null accuracy score
# null_accuracy = (93526 / (93526 + 4040))
# # null_accuracy = (null_acc[0:0] / (null_acc[0:0] + null_acc[0:1]))
# print('Null accuracy score:', null_accuracy)
# print('Null accuracy score: {0:0.4f}'.format(null_accuracy))


# CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix\n\n', cm)
print('\nTrue Positives(TP) = ', cm[0, 0])
print('\nTrue Negatives(TN) = ', cm[1, 1])
print('\nFalse Positives(FP) = ', cm[0, 1])
print('\nFalse Negatives(FN) = ', cm[1, 0])

cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'],
                         index=['Predict Positive:1', 'Predict Negative:0'])

# sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
# plt.show()


# CLASSIFICATION REPORT
# print(classification_report(y_test, y_pred))
TP = cm[0, 0]
TN = cm[1, 1]
FP = cm[0, 1]
FN = cm[1, 0]

classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)
print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))

classification_error = (FP + FN) / float(TP + TN + FP + FN)
print('Classification error : {0:0.4f}'.format(classification_error))

precision = TP / float(TP + FP)
print('Precision : {0:0.4f}'.format(precision))
recall = TP / float(TP + FN)
print('Recall or Sensitivity : {0:0.4f}'.format(recall))
true_positive_rate = TP / float(TP + FN)
print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))
false_positive_rate = FP / float(FP + TN)
print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))
specificity = TN / (TN + FP)
print('Specificity : {0:0.4f}'.format(specificity))


f1 = sklearn.metrics.f1_score(y_test, y_pred)
print('f1: ', f1)
rocauc = sklearn.metrics.roc_auc_score(y_test, y_pred)
print('auc: ', rocauc)

precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
threshold = 0.5
y_pred_binary = (y_pred >= threshold).astype(int)
print('prc: ', rocauc)


# auprc = sklearn.metrics.precision_recall_curve(y_test, y_pred, pos_label=1)
# print('prc: ', auprc)



endTime = datetime.datetime.now()
runningTime = endTime - startTime
print('\nThe software finished at: ', endTime)
print('\nTotal running time: ', runningTime)