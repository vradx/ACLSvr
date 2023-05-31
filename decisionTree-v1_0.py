from sklearn import tree
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import sklearn.metrics
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# File Fetch
df = pd.read_csv(r'D:\Dataset\_4_FOURTH PREPROCESSING\_Task_1-7_removeRows_loudNorm_0726_1225\feature_df.csv', header=0, dtype=np.float64)
# print(df.head())

# Divide labels from Features
labelsTable = df.iloc[:, -1]

# Drop the labels column and the first row
featTable = df
featTable = featTable.drop(featTable.columns[-1], axis=1)

df_col_names = df.columns[:5]
print(df_col_names)
targ_col_names = df.columns[-1]
print(targ_col_names)


# Features count
plt.figure(figsize=(8, 8))
sns.countplot(x = df['Labels'])
plt.title('Count of Normal Respiratory Sounds and Adventitious Sounds')
plt.show()


for i in range(len(labelsTable)):
    if labelsTable[i] == 1:
        labelsTable[i] = 0
    elif labelsTable[i] != 1:
        labelsTable[i] = 1

# Decision tree 0
# Split your data into train and test:
var_train, var_test, res_train, res_test = train_test_split(featTable, labelsTable, test_size=0.2, random_state=5)




# Train your decision tree on train set:
decision_tree = tree.DecisionTreeClassifier(max_depth=8, criterion='entropy')
# decision_tree = decision_tree.fit(var_train, res_train)
# Progress Bar
with tqdm(total=100) as pbar:
    for i in range(100):
        decision_tree = decision_tree.fit(var_train, res_train)
        pbar.update(1)


# Test model performance by calculating accuracy on test set:
res_pred = decision_tree.predict(var_test)
score = accuracy_score(res_test, res_pred)
print('Model performance by calculating accuracy: ', score)

# Or you could directly use decision_tree.score:
dscore = decision_tree.score(var_test, res_test)
print('Model performance by calculating decision_tree: ', dscore)

tree.plot_tree(decision_tree)
plt.show()


# CM
confusion = confusion_matrix(res_test, res_pred)
print('Confusion Matrix\n')
print(confusion)

print('\nAccuracy: {:.2f}\n'.format(accuracy_score(res_test, res_pred)))

print('Micro Precision: {:.2f}'.format(precision_score(res_test, res_pred, average='micro')))
print('Micro Recall: {:.2f}'.format(recall_score(res_test, res_pred, average='micro')))
print('Micro F1-score: {:.2f}\n'.format(f1_score(res_test, res_pred, average='micro')))

print('Macro Precision: {:.2f}'.format(precision_score(res_test, res_pred, average='macro')))
print('Macro Recall: {:.2f}'.format(recall_score(res_test, res_pred, average='macro')))
print('Macro F1-score: {:.2f}\n'.format(f1_score(res_test, res_pred, average='macro')))

print('Weighted Precision: {:.2f}'.format(precision_score(res_test, res_pred, average='weighted')))
print('Weighted Recall: {:.2f}'.format(recall_score(res_test, res_pred, average='weighted')))
print('Weighted F1-score: {:.2f}'.format(f1_score(res_test, res_pred, average='weighted')))

print('\nClassification Report\n')
print(classification_report(res_test, res_pred, target_names=['0', '1']))


cm = confusion_matrix(res_test, res_pred)
print('Confusion matrix\n\n', cm)
print('\nTrue Positives(TP) = ', cm[0, 0])
print('\nTrue Negatives(TN) = ', cm[1, 1])
print('\nFalse Positives(FP) = ', cm[0, 1])
print('\nFalse Negatives(FN) = ', cm[1, 0])

cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'],
                         index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
plt.show()

f1 = sklearn.metrics.f1_score(res_test, res_pred)
print('f1: ', f1)
rocauc = sklearn.metrics.roc_auc_score(res_test, res_pred)
print('auc: ', rocauc)

auprc = sklearn.metrics.precision_recall_curve(res_test, res_pred, pos_label=1)
print('prc: ', auprc)