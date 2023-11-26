import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

dataset = pd.read_csv(r"A:\PROJECTS\DIABETES\diabetes.csv")

dataset.head()

dataset.shape

plt.figure(figsize=(8, 6))
sns.heatmap(dataset.corr(), annot=True, linewidths=2)
plt.show()

dataset.isna().any()

dataset.info()

dataset.describe()

zero_not_accepted = ["Glucose", "BloodPressure", "SkinThickness",
                    "Insulin", "BMI"]
for col in zero_not_accepted:
    dataset[col].replace(0, np.nan, inplace=True)
 
dataset.head(n=10)

plt.figure(figsize=(9,5))
ax = sns.barplot(x=dataset.isna().sum(),
           y=dataset.columns, orient='h')
for p in ax.patches:
    ax.annotate(text=f"{p.get_width():.0f}", 
                xy=(p.get_width(), p.get_y()+p.get_height()/2),
                xytext=(5, 0), textcoords='offset points', 
                ha="left", va="center",
               )
plt.grid(False)
plt.show()

for col in zero_not_accepted:
    dataset[col].replace(np.nan, dataset[col].mean(), inplace=True)

dataset.describe()

plt.figure(figsize=(20,20))
sns.pairplot(data=dataset, hue="Outcome", diag_kind="hist")
plt.show()

dataset["Outcome"].value_counts()

X = dataset.iloc[:, :-1].to_numpy()
y = dataset.iloc[:, -1].to_numpy()

print(X)

print(y)

from imblearn.combine import SMOTETomek
smk = SMOTETomek(random_state=0)
X_r, y_r = smk.fit_resample(X, y)
from collections import Counter
print(f"Initial counts: {Counter(y)}")
print(f"Resampled Counts: {Counter(y_r)}")

print(X_r.shape, y_r.shape)

X = X_r
y = y_r

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                   test_size=0.2,
                                                   random_state=0)

print(X_train.shape)
print(X_train)

print(y_train.shape)
print(y_train)

print(X_test.shape)
print(X_test)

print(y_test.shape)
print(y_test)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

pd.DataFrame(X_train, columns=dataset.columns[:-1]).describe()

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def disp_cm(y_test, y_pred)->float:
    acc_score = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(f"Accuracy: {acc_score:0.3f}")
    plt.xlabel("Predicted labels")
    plt.ylabel("Actual labels")
    plt.show()
    return acc_score
def judge_clf(classifier, X_train=X_train, y_train=y_train,
              X_test=X_test, y_test=y_test)->float:
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    acc_score = disp_cm(y_test, y_pred)
    print(classification_report(y_test, y_pred))
    return acc_score

from sklearn.linear_model import LogisticRegression
log_clf = LogisticRegression(random_state=0)
log_acc = judge_clf(log_clf)

from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
knn_acc = judge_clf(knn_clf)

from sklearn.svm import SVC
lsvm = SVC(kernel="linear", random_state=0)
lsvm_acc = judge_clf(lsvm)

ksvm = SVC(kernel="rbf", random_state=0)
ksvm_acc = judge_clf(ksvm)

from sklearn.naive_bayes import GaussianNB
nb_clf = GaussianNB()
naiveb_acc = judge_clf(nb_clf)

from sklearn.tree import DecisionTreeClassifier
dtree_clf = DecisionTreeClassifier(criterion="entropy",
                                             random_state=0
                                  )
dtree_acc = judge_clf(dtree_clf)

from sklearn.ensemble import RandomForestClassifier
rfor_clf = RandomForestClassifier(n_estimators=100,
                                            criterion="entropy",
                                            random_state=0)
rfor_acc = judge_clf(rfor_clf)

from xgboost import XGBClassifier
xgb_clf = XGBClassifier(use_label_encoder=False,
                       verbosity=0)
xgb_acc = judge_clf(xgb_clf)

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

models = pd.DataFrame({
    'Model': ["Logistic Regr", "KNN", "SVM-Linear", "SVM-RBF", 
             "Naive Bayes", "Decision-Tree", "Radom Forest", "XGB"],
    'Accuracy Score': [log_acc, knn_acc, lsvm_acc, ksvm_acc, 
                       naiveb_acc, dtree_acc, rfor_acc, xgb_acc]
})

models.sort_values(by = 'Accuracy Score', ascending = False, ignore_index=True)

from sklearn.model_selection import cross_val_score
def perform_kfold(clf, X_train=X_train, y_train=y_train)->(float, float):
   
    scores = cross_val_score(estimator=clf, X=X_train, y=y_train, 
                            cv=10 #number of folds
                            )
    
    mean_score = scores.mean()
    std_score = scores.std()
    print(f"Mean Accuracy: {mean_score*100:0.3f} %")
    print(f"Standard Deviation: {std_score*100:0.3f} %")
    
    return mean_score, std_score

log_macc, log_std = perform_kfold(log_clf)

knn_macc, knn_std = perform_kfold(knn_clf)

lsvm_macc, lsvm_std = perform_kfold(lsvm)

ksvm_macc, ksvm_std = perform_kfold(ksvm)

naiveb_macc, naiveb_std = perform_kfold(nb_clf)

rfor_macc, rfor_std = perform_kfold(rfor_clf)

xgb_macc, xgb_std = perform_kfold(xgb_clf)

#mlp_macc, mlp_std = perform_kfold(mlp_clf)

dtree_macc, dtree_std = perform_kfold(dtree_clf)

models_1 = pd.DataFrame({
    "Model": ["Logistic Regr", "KNN", "SVM-Linear", "SVM-RBF", 
             "Naive Bayes", "Decision-Tree", "Radom Forest", "XGB"],
    "Mean Accuracy Score": [log_macc, knn_macc, lsvm_macc, ksvm_macc, 
                       naiveb_macc, dtree_macc, rfor_macc, xgb_macc],
    "Stadard Dev": [log_std, knn_std, lsvm_std, ksvm_std, 
                       naiveb_std, dtree_std, rfor_std, xgb_std
            ]
})

models_1.sort_values(by = 'Mean Accuracy Score', ascending = False,
                     ignore_index=True)

import matplotlib.pyplot as plt
fig = plt.figure(figsize =(12, 8))
models_1.plot(x="Model", y=["Stadard Dev","Mean Accuracy Score"], kind="bar",legend=False)
plt.show()