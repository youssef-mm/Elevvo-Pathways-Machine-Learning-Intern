import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_score, recall_score, f1_score


# Dataset
df = pd.read_csv("loan_approval_dataset.csv")
print(df)
print(df.head())
print(df.describe())
print(df.info())


#Handle Missing Value
Cols_num = df.select_dtypes(include=['int64','float64']).columns
df[Cols_num] = df[Cols_num].fillna(df[Cols_num].median())

cols_categ = df.select_dtypes(include=['object']).columns
df[cols_categ] = df[cols_categ].fillna(df[cols_categ].mode())

#Encode Categorical Features 
encode = LabelEncoder()
for col in [' education',' self_employed',' loan_status']:
    df[col] = encode.fit_transform(df[col])


#Define Features
x = df.drop(columns=[' loan_status', 'loan_id'])  
y = df[' loan_status']  


#Scale Numerical Features
scaler = StandardScaler()
x_scaler = scaler.fit_transform(x)


#Train and Test Split
x_train,x_test,y_tain,y_test = train_test_split(x_scaler,y,test_size=0.2,random_state=37,stratify=y)


#Handle Imbalance using SMOTE
smoteHandle = SMOTE(random_state=37)
x_train_sm,y_tain_sm = smoteHandle.fit_resample(x_train,y_tain)

print('Before SMOTE',np.bincount(y_tain))
print('After SMOTE',np.bincount(y_tain_sm))


#Train Models
#LogisticRegression
Log_Model =LogisticRegression(max_iter=1333,random_state=37)
Log_Model.fit(x_train_sm,y_tain_sm)
y_Pred_Log = Log_Model.predict(x_test)

#DecisionTree
Tree_model = DecisionTreeClassifier(max_depth=7,class_weight="balanced",random_state=37)
Tree_model.fit(x_train_sm,y_tain_sm)
y_pred_tree = Tree_model.predict(x_test)


#Evaluation
print('LogisticRegression')
print(classification_report(y_test,y_Pred_Log))
ConfusionMatrixDisplay.from_predictions(y_test,y_Pred_Log)
plt.title("ConfusionMatrix and LogisticRegression")
plt.show()

print("DecisionTree")
print(classification_report(y_test,y_pred_tree))
ConfusionMatrixDisplay.from_predictions(y_test,y_pred_tree)
plt.title("ConfusionMatrix and DecisionTree")
plt.show()


#Metrics for LogisticRegression and DecisionTree
metrics = {'Logistic Regression': {
        'Precision': precision_score(y_test, y_Pred_Log),
        'Recall': recall_score(y_test, y_Pred_Log),
        'F1': f1_score(y_test, y_Pred_Log)},
    
    'Decision Tree': {'Precision': precision_score(y_test, y_pred_tree),
        'Recall': recall_score(y_test, y_pred_tree),
        'F1': f1_score(y_test, y_pred_tree)}}

Metrics_df = pd.DataFrame(metrics).T

Metrics_df.plot(kind='bar',figsize=(7,5))
plt.title("The Metrics between models")
plt.ylabel("SCORE")
plt.ylim(0,1)
plt.legend('Lower Right')
plt.show()
