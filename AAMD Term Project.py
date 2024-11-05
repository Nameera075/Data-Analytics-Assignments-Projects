#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,roc_curve, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
import statsmodels.api as sm


# In[2]:


HR = pd.read_csv(r'C:\Users\namee\Downloads\HR_data.csv')
HR.head()


# In[3]:


HR.describe()


# In[4]:


HR.corr()


# In[5]:


HR.info()


# In[6]:


left = HR.groupby('left')
left.mean()


# In[7]:


g = sns.FacetGrid(HR, col="left",  row="promotion_last_5years")
g.map(sns.scatterplot, "satisfaction_level", "last_evaluation")


# # K-Means Clustering

# In[8]:


X = HR.drop(['sales','salary'] , axis=1)[HR.left == 1]


# In[9]:


X.head()


# In[10]:


scaler = MinMaxScaler()
# transform data
XS = scaler.fit_transform(X)
XS[1:5]


# In[11]:


#Finding the optimum number of clusters for k-means classification
wss = []
sscore = []
for i in range(2, 15):
    kmeans = KMeans(n_clusters = i, max_iter = 300, random_state = 99)
    kmeans.fit(XS)
    wss.append(kmeans.inertia_)
    Y = kmeans.fit_predict(XS)
    sscore.append(silhouette_score(XS, Y))


# In[12]:


plt.plot(range(2, 15), wss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') #within cluster sum of squares
plt.show()


# In[13]:


# optimal no of clusters and why


# In[14]:


kmeans = KMeans(n_clusters = 3, max_iter = 300, random_state = 99)
Y = kmeans.fit_predict(XS)


# In[15]:


Y


# In[16]:


#### Plotting any two features for visualization


# In[17]:


plt.scatter(x=X.satisfaction_level,y=X.last_evaluation)
plt.title("The actual dataset")
plt.xlabel('Satisfaction Level')
plt.ylabel('Last Evaluation')


# In[18]:


plt.figure(figsize=(9,5))
plt.xlabel('Satisfaction Level')
plt.ylabel('Last Evaluation')
plt.title('Employees who left')


plt.scatter(XS[Y == 0, 0], XS[Y==0, 1], s=50, c='purple')
plt.scatter(XS[Y==1, 0], XS[Y==1, 1], s=50, c='orange')
plt.scatter(XS[Y==2, 0], XS[Y==2, 1], s=50, c='green')


# # Machie Learning Models

# In[19]:


from sklearn.linear_model import LogisticRegression


# In[20]:


X.head()


# ## 1. Logistic Regression

# In[21]:


### Model 1 with 20% testing data and 80% training data


# In[22]:


X_LR = HR.drop(["left"], axis=1)
Y_LR = HR['left']


# In[23]:


X_LR.columns


# In[24]:


X_LR = pd.get_dummies(X_LR, columns=['sales', 'salary'])


# In[25]:


from sklearn.model_selection import train_test_split
x_train_LR,x_test_LR,y_train_LR,y_test_LR = train_test_split(X_LR,Y_LR,test_size=0.2)


# In[26]:


logistic_model = LogisticRegression(class_weight='balanced', max_iter=1500)
logistic_model = logistic_model.fit(x_train_LR, y_train_LR)


# In[27]:


# Hard Prediction (Class Label)
predicted = logistic_model.predict(x_test_LR)
predicted


# In[28]:


#Soft Prediction (Probabilty Scores)
predic_prob = logistic_model.predict_proba(x_test_LR)
predic_prob


# In[29]:


sns.regplot(x='satisfaction_level', y='left', data= HR , logistic=True, color='c', ci = None)


# ### Classification Report

# In[30]:


from sklearn import metrics
print (metrics.confusion_matrix(y_test_LR, predicted))


# In[31]:


print (metrics.classification_report(y_test_LR, predicted))


# ### Confusion Matrix 

# In[32]:


cm_LR = confusion_matrix(y_test_LR, predicted)
cp_LR = ConfusionMatrixDisplay(cm_LR, display_labels=logistic_model.classes_)
cp_LR.plot()


# ### AUC-ROC Curve

# In[33]:


score = roc_auc_score(y_test_LR, predicted)
print("AUC-ROC score:", score)


# In[34]:


import matplotlib.pyplot as plt

def plot_roc_curve(true_y_LR, y_prob_LR):
    """
    plots the roc curve based of the probabilities
    """

    fpr, tpr, thresholds = roc_curve(true_y_LR, y_prob_LR)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')


# In[35]:


plot_roc_curve(y_test_LR, predicted)


# # 2. Decision Tree

# ## Model

# In[36]:


X_DT = HR.drop(["left"], axis=1)
Y_DT = HR['left']


# In[37]:


X_DT = pd.get_dummies(X_DT, columns=['sales', 'salary'])


# In[38]:


dt = tree.DecisionTreeClassifier(random_state=5)
dtmodel = dt.fit(X_DT, Y_DT)


# In[39]:


X_train_DT, X_test_DT, Y_train_DT, Y_test_DT = train_test_split(X_DT, Y_DT, test_size=0.2, random_state=5)
dtmodel = dt.fit(X_train_DT, Y_train_DT)
train_pred = dtmodel.predict(X_train_DT)
acc = accuracy_score(train_pred, Y_train_DT)
print("Accuracy of predicting training data: ")
print(acc *100)


# In[40]:


test_pred = dtmodel.predict(X_test_DT)
acc = accuracy_score(test_pred, Y_test_DT)
print("Accuracy of predicting testing data: ")
print(acc *100)


# In[41]:


print(tree.export_text(dtmodel))


# In[42]:


X_DT = X_DT.astype("str")
Y_DT = Y_DT.astype("str")


# In[70]:


# printing dT
fig = plt.figure(figsize=(14, 10))
_ = tree.plot_tree(dtmodel, feature_names=list(X_DT.columns), class_names=Y_DT.value_counts().index, filled=True)


# ## Classification Report

# In[44]:


print(classification_report(test_pred,Y_test_DT))


# ## Confusion Matrix

# In[45]:


cm_DT = confusion_matrix(Y_test_DT, test_pred)
cp_DT = ConfusionMatrixDisplay(cm_DT, display_labels=dtmodel.classes_)
cp_DT.plot()


# ## AUO ROC Curve

# In[46]:


score = roc_auc_score(Y_test_DT,test_pred)
print("AUC-ROC score:", score)


# In[47]:


import matplotlib.pyplot as plt

def plot_roc_curve(true_y_DT, y_prob_DT):
    """
    plots the roc curve based of the probabilities
    """

    fpr, tpr, thresholds = roc_curve(true_y_DT, y_prob_DT)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')


# In[48]:


plot_roc_curve(Y_test_DT,test_pred)


# # 3. Random Forest

# In[49]:


X_RF = HR.drop(["left"], axis=1)
Y_RF = HR['left']


# In[50]:


X_RF = pd.get_dummies(X_RF, columns=['sales', 'salary'])


# In[51]:


### 

rf = RandomForestClassifier(n_estimators=100)
rfmodel = rf.fit(X_train_DT, Y_train_DT)
train_pred = rfmodel.predict(X_train_DT)
acc = accuracy_score(train_pred, Y_train_DT)
print("Accuracy of random forest training is: ")
print(acc * 100)


# In[52]:


test_pred = rfmodel.predict(X_test_DT)
acc = accuracy_score(test_pred, Y_test_DT)
print("Accuracy of random forest testing is: ")
print(acc * 100)


# ## Classification Report

# In[53]:


print(classification_report(test_pred,Y_test_DT))


# ## Confusion Matrix

# In[54]:


cm_RF = confusion_matrix(Y_test_DT, test_pred)
cp_RF = ConfusionMatrixDisplay(cm_RF, display_labels=rfmodel.classes_)
cp_RF.plot()


# ## AUC-ROC curve

# In[55]:


score = roc_auc_score(Y_test_DT,test_pred)
print("AUC-ROC score:", score)


# In[56]:


import matplotlib.pyplot as plt

def plot_roc_curve(true_y_DT, y_prob_DT):
    """
    plots the roc curve based of the probabilities
    """

    fpr, tpr, thresholds = roc_curve(true_y_DT, y_prob_DT)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')


# In[57]:


plot_roc_curve(Y_test_DT,test_pred)


# # Learning Curve

# In[58]:


### Learning Curve for Logistic Regression

train_sizes, train_scores, test_scores = learning_curve(estimator= logistic_model, X=X_LR, y=Y_LR, cv=10)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)


# In[59]:


plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training Accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, color='green', marker='+', markersize=5, linestyle='--', label='Validation Accuracy')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.title('Learning Curve')
plt.xlabel('Training Data Size')
plt.ylabel('Model error')
plt.grid()
plt.legend(loc='lower right')
plt.show()


# In[60]:


### Learning Curve for Decision Tree

train_sizes, train_scores, test_scores = learning_curve(estimator= dt, X=X_DT, y=Y_DT, cv=10)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)


# In[61]:


plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training Accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, color='green', marker='+', markersize=5, linestyle='--', label='Validation Accuracy')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.title('Learning Curve')
plt.xlabel('Training Data Size')
plt.ylabel('Model error')
plt.grid()
plt.legend(loc='lower right')
plt.show()


# In[62]:


### Learning Curve for Random Forest

train_sizes, train_scores, test_scores = learning_curve(estimator= RandomForestClassifier(), X=X_RF, y=Y_RF, cv=10)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)


# In[63]:


plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training Accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(train_sizes, test_mean, color='green', marker='+', markersize=5, linestyle='--', label='Validation Accuracy')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.title('Learning Curve')
plt.xlabel('Training Data Size')
plt.ylabel('Model error')
plt.grid()
plt.legend(loc='lower right')
plt.show()


# # Decision Boundary

# In[64]:


import numpy as np
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression

X_LR, Y_LR = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=1, cluster_std=3)

min1, max1 = X_LR[:, 0].min()-1, X_LR[:, 0].max()+1
min2, max2 = X_LR[:, 1].min()-1, X_LR[:, 1].max()+1

x1grid = np.arange(min1, max1, 0.1)
x2grid = np.arange(min2, max2, 0.1)

xx, yy = np.meshgrid(x1grid, x2grid)

r1, r2 = xx.flatten(), yy.flatten()
r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))

grid = np.hstack((r1,r2))

model = logistic_model

model.fit(X_LR, Y_LR)

yhat = model.predict(grid)

zz = yhat.reshape(xx.shape)

plt.contourf(xx, yy, zz, cmap='Paired')

for class_value in range(2):
    
    row_ix = np.where(Y_LR == class_value)
    
    plt.scatter(X_LR[row_ix, 0], X_LR[row_ix, 1])
    plt.xlabel('Employees who will not churn')
    plt.ylabel('Employees who will churn')


# #  Precision Recall Curve

# In[65]:


#calculating precision and recall for Logistic Regression
precision = precision_score(y_test_LR, predicted)
recall = recall_score(y_test_LR, predicted)
 
print('Precision: ',precision)
print('Recall: ',recall)

precision, recall, thresholds = precision_recall_curve(y_test_LR, predicted)

fig, ax = plt.subplots()
ax.plot(recall, precision, color='purple')
ax.set_title('Precision-Recall Curve')
ax.set_ylabel('Precision')
ax.set_xlabel('Recall')
plt.show()


# # Top 10 features that were most significant

# In[66]:


X_df = pd.get_dummies(HR, drop_first=True)
X_df.head()


# In[67]:


X_df = X_df.drop(['left'], axis=1)


# In[68]:


X = sm.add_constant(X_df)
y = HR['left']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 11)

# logit = sm.Logit(y_train, X_train)
# logit_model = logit.fit()
# logit_model.summary2()

model = sm.OLS( y_train, X_train ).fit()
model.summary2()


# 1.   Last Evaluation
# 2.   Time Spent at Company
# 3.   Number of Projects
# 4.   Satisfaction Levels
# 5.   Average Monthly Hours
# 6.   time_spend_company
# 7.   Work Accidient
# 8.   salary _ Low 
# 9.   salary _ medium
# 10.  promotion last five years
# 
