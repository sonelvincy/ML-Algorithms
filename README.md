# ML-Algorithms

# Importing the libraries

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
%matplotlib inline

# Ignore harmless warnings 

import warnings 
warnings.filterwarnings("ignore")

# Set to display all the columns in dataset

pd.set_option("display.max_columns", None)

# Import psql to run queries 

#import pandasql as psql


# Load the Concrete dataset

freq = pd.read_csv(r"C:\Users\Asus\Downloads\freMTPL2freq.csv", header=0)

# Copy the file to back-up file

freq_bk = freq.copy()

# Display first 5 records

freq.head()


freq.info()


# Displaying Duplicate values with in dataset

freq_dup = freq[freq.duplicated(keep='last')]

# Display the duplicate records

freq_dup


freq.shape


freq.nunique()


freq.isnull().sum()


freq['Area'].value_counts()

freq['Area'] = freq['Area'].replace('C', '0')
freq['Area'] = freq['Area'].replace('D', '1')
freq['Area'] = freq['Area'].replace('E', '2')
freq['Area'] = freq['Area'].replace('A', '3')
freq['Area'] = freq['Area'].replace('B', '4')
freq['Area'] = freq['Area'].replace('F', '5')

freq['Area'] = freq['Area'].astype(int)


freq['VehBrand'].value_counts()
del freq['VehBrand']


freq['VehGas'].value_counts()
freq['VehGas'] = freq['VehGas'].replace('Regular', '0')
freq['VehGas'] = freq['VehGas'].replace('Diesel', '1')
freq['VehGas'] = freq['VehGas'].astype(int)


freq['VehGas'].value_counts()


del freq['Region']


freq.info()

freq.columns


col=['IDpol', 'ClaimNb', 'Exposure', 'Area', 'VehPower', 'VehAge', 'DrivAge',
       'BonusMalus', 'Density']


       # Count the target or dependent variable by '0' & '1' and their proportion  
# (> 10 : 1, then the dataset is imbalance data) 
      
freq_count = freq.VehGas.value_counts() 
print('Class 0:', freq_count[0]) 
print('Class 1:', freq_count[1]) 
print('Proportion:', round(freq_count[0] /freq_count[1], 2), ': 1') 
print('Total counts in VehGas:', len(freq))


# Identify the independent and Target variables

IndepVar = []
for col in freq.columns:
    if col != 'VehGas':
        IndepVar.append(col)

TargetVar = 'VehGas'

x = freq[IndepVar]
y = freq[TargetVar]


# Split the data into train and test

from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=143)

# Display the shape of the train_data and test_data

x_train.shape, x_test.shape, y_train.shape, y_test.shape


# Feature Scaling - Each independent variable is in different range. The process of transforming all the 
# features in the given data set to a fixed range is known as ‘Scaling’

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

APPLYING DECISION TREE ALGORITHM:

# To build the 'Decision Tree' model with random sampling

from sklearn.tree import DecisionTreeClassifier

# Create an object for model

ModelDT = DecisionTreeClassifier()
#ModelDT = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, 
#                                 min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, 
#                                 random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, 
#                                 class_weight=None, ccp_alpha=0.0)

# Train the model with train data 

ModelDT.fit(x_train,y_train)

# Predict the model with test data set

y_pred = ModelDT.predict(x_test)
y_pred_prob = ModelDT.predict_proba(x_test)

# Confusion matrix in sklearn

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# actual values

actual = y_test

# predicted values

predicted = y_pred

# confusion matrix

matrix = confusion_matrix(actual,predicted, labels=[1,0],sample_weight=None, normalize=None)
print('Confusion matrix : \n', matrix)

# outcome values order in sklearn

tp, fn, fp, tn = confusion_matrix(actual,predicted,labels=[1,0]).reshape(-1)
print('Outcome values : \n', tp, fn, fp, tn)

# classification report for precision, recall f1-score and accuracy

C_Report = classification_report(actual,predicted,labels=[1,0])

print('Classification report : \n', C_Report)

# calculating the metrics
sensitivity = round(tp/(tp+fn), 3)
specificity = round(tn/(tn+fp), 3)
accuracy = round((tp+tn)/(tp+fp+tn+fn), 3)
balanced_accuracy = round((sensitivity+specificity)/2, 3)
precision = round(tp/(tp+fp), 3)
f1Score = round((2*tp/(2*tp + fp + fn)), 3)

# Check if any of the denominators are zero to avoid division by zero
if tp+fp+tn+fn == 0:
    accuracy = 0
    sensitivity = 0
    specificity = 0
    precision = 0
    f1Score = 0
else:
    sensitivity = round(tp/(tp+fn), 3)
    specificity = round(tn/(tn+fp), 3)
    accuracy = round((tp+tn)/(tp+fp+tn+fn), 3)
    balanced_accuracy = round((sensitivity+specificity)/2, 3)
    precision = round(tp/(tp+fp), 3)
    f1Score = round((2*tp/(2*tp + fp + fn)), 3)
    
    # Matthews Correlation Coefficient (MCC)
    mx = (tp+fp) * (tp+fn) * (tn+fp) * (tn+fn)
    if mx <= 0:
        MCC = 0
    else:
        MCC = round(((tp * tn) - (fp * fn)) / sqrt(mx), 3)

print('Accuracy :', round(accuracy*100, 2),'%')
print('Precision :', round(precision*100, 2),'%')
print('Recall :', round(sensitivity*100,2), '%')
print('F1 Score :', f1Score)
print('Specificity or True Negative Rate :', round(specificity*100,2), '%'  )
print('Balanced Accuracy :', round(balanced_accuracy*100, 2),'%')
print('MCC :', MCC)



# Area under ROC curve 

from sklearn.metrics import roc_curve, roc_auc_score

print('roc_auc_score:', round(roc_auc_score(actual, predicted), 3))

# ROC Curve

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
model_roc_auc = roc_auc_score(actual, predicted)
fpr, tpr, thresholds = roc_curve(actual, ModelDT.predict_proba(x_test)[:,1])
plt.figure()
# plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot(fpr, tpr, label= 'Classification Model' % model_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
print('-------------------------------------------------------------------------------')


# To get feature importance

from matplotlib import pyplot

importance = ModelDT.feature_importances_

# Summarize feature importance

for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
    
# Plot feature importance

pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()


Results = pd.DataFrame({'VehGas_A':y_test, 'VehGas _B':y_pred})

# Merge two Dataframes on index of both the dataframes

ResultsFinal = freq_bk.merge(Results, left_index=True, right_index=True)
ResultsFinal.sample(5)


del ResultsFinal['VehGas_A']
ResultsFinal.head()


# Load the results dataset

EMResults = pd.read_csv(r"C:\Users\Asus\Downloads\EMResults (1).csv", header=0)

# Display the first 5 records

EMResults.head()

COMPARE WITH OTHER ALGORITHMS:

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier

# Assuming you have defined x_train, x_test, y_train, and y_test before this point

# Create objects of classification algorithm with default hyper-parameters

ModelLR = LogisticRegression()
ModelDC = DecisionTreeClassifier()
ModelRF = RandomForestClassifier()
ModelET = ExtraTreesClassifier()
ModelKNN = KNeighborsClassifier(n_neighbors=5)

MM = [ModelLR, ModelDC, ModelRF, ModelET, ModelKNN]

results_list = []

for models in MM:
    
    # Fit the model
    models.fit(x_train, y_train)
    
    # Prediction
    y_pred = models.predict(x_test)
    
    # Print the model name
    print('Model Name:', models.__class__.__name__)
    
    # Confusion matrix
    matrix = confusion_matrix(y_test, y_pred, labels=[1, 0])
    print('Confusion matrix:\n', matrix)
    
    # Classification report
    print('Classification report:\n', classification_report(y_test, y_pred, labels=[1, 0]))
    
    # Calculate metrics
    tp, fn, fp, tn = matrix.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    specificity = tn / (tn + fp)
    balanced_accuracy = (recall + specificity) / 2
    mcc = ((tp * tn) - (fp * fn)) / (((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5)
    roc_auc = roc_auc_score(y_test, y_pred)
    
    print('Accuracy:', round(accuracy * 100, 2), '%')
    print('Precision:', round(precision * 100, 2), '%')
    print('Recall:', round(recall * 100, 2), '%')
    print('F1 Score:', round(f1_score, 3))
    print('Specificity:', round(specificity * 100, 2), '%')
    print('Balanced Accuracy:', round(balanced_accuracy * 100, 2), '%')
    print('MCC:', round(mcc, 3))
    print('ROC AUC Score:', round(roc_auc, 3))
    print('---')
    
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, models.predict_proba(x_test)[:,1])
    plt.plot(fpr, tpr, label=models.__class__.__name__)
    
    # Append results to the list
    results_list.append({
        'Model Name': models.__class__.__name__,
        'True Positive': tp,
        'False Negative': fn,
        'False Positive': fp,
        'True Negative': tn,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1_score,
        'Specificity': specificity,
        'Balanced Accuracy': balanced_accuracy,
        'MCC': mcc,
        'ROC AUC Score': roc_auc
    })

# Plot ROC Curve
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Convert the list to a DataFrame
EMResults = pd.DataFrame(results_list)


EMResults.head(10)
