# Predicting Credit Card Approvals

### - Goal: Build an automatic credit card approval predictor
### - Data: [Credit Card Approval dataset](http://archive.ics.uci.edu/ml/datasets/credit+approval) 

- Since this data is confidential, the contributor of this dataset has anonymized the feature names.
- However, this [blog](http://rstudio-pubs-static.s3.amazonaws.com/73039_9946de135c0a49daa7a0a9eda4a67a72.html) gives a good overview of the probable features (i.e. Gender, Age, Debt, Married, BankCustomer, EducationLevel, Ethnicity, YearsEmployed, PriorDefault, Employed, CreditScore, DriversLicense, Citizen, ZipCode, Income, ApprovalStatus).

### - Python Libraries Used
- pandas, numpy, sklearn

### - File Descriptions
- datasets/cc_approvals.data: the dataset described above
- Predicting-Credit-Card-Approvals.ipynb: Jupyter Notebook used for predicting credit card approvals

### - Acknowledgement
- This work is based on [DataCamp projects](https://www.datacamp.com/projects) and DataCamp has [agreed](https://support.datacamp.com/hc/en-us/articles/360006091334-DataCamp-Projects-An-Overview) to add projects to a personal portfolio.

### - Table of Contents
1. [Exploratory Data Analysis](#exploratory-data-analysis)
2. [Data Preprocessing and Feature Engineering](#data-preprocessing-and-feature-engineering)
    - 2.1 [Missing Values](#missing-values)
    - 2.2 [Feature Engineering](#feature-engineering)
3. [Modeling and Evaluation](#modeling-and-evaluation)
    - 3.1 [Logistic Regression](#logistic-regression)
    - 3.2 [GridSearchCV](#gridsearchcv)

## Exploratory Data Analysis


```python
# Import pandas
import pandas as pd

# Load dataset
cc_apps = pd.read_csv('datasets/cc_approvals.data', header = None)

# Inspect data
cc_apps.head()
```




<div>
<!--
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
-->
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>b</td>
      <td>30.83</td>
      <td>0.000</td>
      <td>u</td>
      <td>g</td>
      <td>w</td>
      <td>v</td>
      <td>1.25</td>
      <td>t</td>
      <td>t</td>
      <td>1</td>
      <td>f</td>
      <td>g</td>
      <td>00202</td>
      <td>0</td>
      <td>+</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a</td>
      <td>58.67</td>
      <td>4.460</td>
      <td>u</td>
      <td>g</td>
      <td>q</td>
      <td>h</td>
      <td>3.04</td>
      <td>t</td>
      <td>t</td>
      <td>6</td>
      <td>f</td>
      <td>g</td>
      <td>00043</td>
      <td>560</td>
      <td>+</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a</td>
      <td>24.50</td>
      <td>0.500</td>
      <td>u</td>
      <td>g</td>
      <td>q</td>
      <td>h</td>
      <td>1.50</td>
      <td>t</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>g</td>
      <td>00280</td>
      <td>824</td>
      <td>+</td>
    </tr>
    <tr>
      <th>3</th>
      <td>b</td>
      <td>27.83</td>
      <td>1.540</td>
      <td>u</td>
      <td>g</td>
      <td>w</td>
      <td>v</td>
      <td>3.75</td>
      <td>t</td>
      <td>t</td>
      <td>5</td>
      <td>t</td>
      <td>g</td>
      <td>00100</td>
      <td>3</td>
      <td>+</td>
    </tr>
    <tr>
      <th>4</th>
      <td>b</td>
      <td>20.17</td>
      <td>5.625</td>
      <td>u</td>
      <td>g</td>
      <td>w</td>
      <td>v</td>
      <td>1.71</td>
      <td>t</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>s</td>
      <td>00120</td>
      <td>0</td>
      <td>+</td>
    </tr>
  </tbody>
</table>
</div>



- The dataset has a mixture of numerical and non-numerical features.


```python
# Print summary statistics
cc_apps_description = cc_apps.describe()
print(cc_apps_description)

print('\n')

# Print DataFrame information
cc_apps_info = cc_apps.info()
print(cc_apps_info)

print('\n')

# Inspect missing values in the dataset
cc_apps.tail(20)
```

                   2           7          10             14
    count  690.000000  690.000000  690.00000     690.000000
    mean     4.758725    2.223406    2.40000    1017.385507
    std      4.978163    3.346513    4.86294    5210.102598
    min      0.000000    0.000000    0.00000       0.000000
    25%      1.000000    0.165000    0.00000       0.000000
    50%      2.750000    1.000000    0.00000       5.000000
    75%      7.207500    2.625000    3.00000     395.500000
    max     28.000000   28.500000   67.00000  100000.000000
    
    
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 690 entries, 0 to 689
    Data columns (total 16 columns):
     #   Column  Non-Null Count  Dtype  
    ---  ------  --------------  -----  
     0   0       690 non-null    object 
     1   1       690 non-null    object 
     2   2       690 non-null    float64
     3   3       690 non-null    object 
     4   4       690 non-null    object 
     5   5       690 non-null    object 
     6   6       690 non-null    object 
     7   7       690 non-null    float64
     8   8       690 non-null    object 
     9   9       690 non-null    object 
     10  10      690 non-null    int64  
     11  11      690 non-null    object 
     12  12      690 non-null    object 
     13  13      690 non-null    object 
     14  14      690 non-null    int64  
     15  15      690 non-null    object 
    dtypes: float64(2), int64(2), object(12)
    memory usage: 86.4+ KB
    None
    
    





<div>
<!--
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
-->
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>670</th>
      <td>b</td>
      <td>47.17</td>
      <td>5.835</td>
      <td>u</td>
      <td>g</td>
      <td>w</td>
      <td>v</td>
      <td>5.500</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>g</td>
      <td>00465</td>
      <td>150</td>
      <td>-</td>
    </tr>
    <tr>
      <th>671</th>
      <td>b</td>
      <td>25.83</td>
      <td>12.835</td>
      <td>u</td>
      <td>g</td>
      <td>cc</td>
      <td>v</td>
      <td>0.500</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>g</td>
      <td>00000</td>
      <td>2</td>
      <td>-</td>
    </tr>
    <tr>
      <th>672</th>
      <td>a</td>
      <td>50.25</td>
      <td>0.835</td>
      <td>u</td>
      <td>g</td>
      <td>aa</td>
      <td>v</td>
      <td>0.500</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>t</td>
      <td>g</td>
      <td>00240</td>
      <td>117</td>
      <td>-</td>
    </tr>
    <tr>
      <th>673</th>
      <td>?</td>
      <td>29.50</td>
      <td>2.000</td>
      <td>y</td>
      <td>p</td>
      <td>e</td>
      <td>h</td>
      <td>2.000</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>g</td>
      <td>00256</td>
      <td>17</td>
      <td>-</td>
    </tr>
    <tr>
      <th>674</th>
      <td>a</td>
      <td>37.33</td>
      <td>2.500</td>
      <td>u</td>
      <td>g</td>
      <td>i</td>
      <td>h</td>
      <td>0.210</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>g</td>
      <td>00260</td>
      <td>246</td>
      <td>-</td>
    </tr>
    <tr>
      <th>675</th>
      <td>a</td>
      <td>41.58</td>
      <td>1.040</td>
      <td>u</td>
      <td>g</td>
      <td>aa</td>
      <td>v</td>
      <td>0.665</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>g</td>
      <td>00240</td>
      <td>237</td>
      <td>-</td>
    </tr>
    <tr>
      <th>676</th>
      <td>a</td>
      <td>30.58</td>
      <td>10.665</td>
      <td>u</td>
      <td>g</td>
      <td>q</td>
      <td>h</td>
      <td>0.085</td>
      <td>f</td>
      <td>t</td>
      <td>12</td>
      <td>t</td>
      <td>g</td>
      <td>00129</td>
      <td>3</td>
      <td>-</td>
    </tr>
    <tr>
      <th>677</th>
      <td>b</td>
      <td>19.42</td>
      <td>7.250</td>
      <td>u</td>
      <td>g</td>
      <td>m</td>
      <td>v</td>
      <td>0.040</td>
      <td>f</td>
      <td>t</td>
      <td>1</td>
      <td>f</td>
      <td>g</td>
      <td>00100</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <th>678</th>
      <td>a</td>
      <td>17.92</td>
      <td>10.210</td>
      <td>u</td>
      <td>g</td>
      <td>ff</td>
      <td>ff</td>
      <td>0.000</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>g</td>
      <td>00000</td>
      <td>50</td>
      <td>-</td>
    </tr>
    <tr>
      <th>679</th>
      <td>a</td>
      <td>20.08</td>
      <td>1.250</td>
      <td>u</td>
      <td>g</td>
      <td>c</td>
      <td>v</td>
      <td>0.000</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>g</td>
      <td>00000</td>
      <td>0</td>
      <td>-</td>
    </tr>
    <tr>
      <th>680</th>
      <td>b</td>
      <td>19.50</td>
      <td>0.290</td>
      <td>u</td>
      <td>g</td>
      <td>k</td>
      <td>v</td>
      <td>0.290</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>g</td>
      <td>00280</td>
      <td>364</td>
      <td>-</td>
    </tr>
    <tr>
      <th>681</th>
      <td>b</td>
      <td>27.83</td>
      <td>1.000</td>
      <td>y</td>
      <td>p</td>
      <td>d</td>
      <td>h</td>
      <td>3.000</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>g</td>
      <td>00176</td>
      <td>537</td>
      <td>-</td>
    </tr>
    <tr>
      <th>682</th>
      <td>b</td>
      <td>17.08</td>
      <td>3.290</td>
      <td>u</td>
      <td>g</td>
      <td>i</td>
      <td>v</td>
      <td>0.335</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>t</td>
      <td>g</td>
      <td>00140</td>
      <td>2</td>
      <td>-</td>
    </tr>
    <tr>
      <th>683</th>
      <td>b</td>
      <td>36.42</td>
      <td>0.750</td>
      <td>y</td>
      <td>p</td>
      <td>d</td>
      <td>v</td>
      <td>0.585</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>g</td>
      <td>00240</td>
      <td>3</td>
      <td>-</td>
    </tr>
    <tr>
      <th>684</th>
      <td>b</td>
      <td>40.58</td>
      <td>3.290</td>
      <td>u</td>
      <td>g</td>
      <td>m</td>
      <td>v</td>
      <td>3.500</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>t</td>
      <td>s</td>
      <td>00400</td>
      <td>0</td>
      <td>-</td>
    </tr>
    <tr>
      <th>685</th>
      <td>b</td>
      <td>21.08</td>
      <td>10.085</td>
      <td>y</td>
      <td>p</td>
      <td>e</td>
      <td>h</td>
      <td>1.250</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>g</td>
      <td>00260</td>
      <td>0</td>
      <td>-</td>
    </tr>
    <tr>
      <th>686</th>
      <td>a</td>
      <td>22.67</td>
      <td>0.750</td>
      <td>u</td>
      <td>g</td>
      <td>c</td>
      <td>v</td>
      <td>2.000</td>
      <td>f</td>
      <td>t</td>
      <td>2</td>
      <td>t</td>
      <td>g</td>
      <td>00200</td>
      <td>394</td>
      <td>-</td>
    </tr>
    <tr>
      <th>687</th>
      <td>a</td>
      <td>25.25</td>
      <td>13.500</td>
      <td>y</td>
      <td>p</td>
      <td>ff</td>
      <td>ff</td>
      <td>2.000</td>
      <td>f</td>
      <td>t</td>
      <td>1</td>
      <td>t</td>
      <td>g</td>
      <td>00200</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <th>688</th>
      <td>b</td>
      <td>17.92</td>
      <td>0.205</td>
      <td>u</td>
      <td>g</td>
      <td>aa</td>
      <td>v</td>
      <td>0.040</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>f</td>
      <td>g</td>
      <td>00280</td>
      <td>750</td>
      <td>-</td>
    </tr>
    <tr>
      <th>689</th>
      <td>b</td>
      <td>35.00</td>
      <td>3.375</td>
      <td>u</td>
      <td>g</td>
      <td>c</td>
      <td>h</td>
      <td>8.290</td>
      <td>f</td>
      <td>f</td>
      <td>0</td>
      <td>t</td>
      <td>g</td>
      <td>00000</td>
      <td>0</td>
      <td>-</td>
    </tr>
  </tbody>
</table>
</div>



- The dataset contains both numeric and non-numeric data.
- The dataset also contains values from several ranges. 
- The dataset has missing values, which are labeled with '?'.

## Data Preprocessing and Feature Engineering
### Missing Values
- I will replace the missing values with NaN and impute the missing values with mean imputation.


```python
# Import numpy
import numpy as np

# Replace the '?'s with NaN
cc_apps = cc_apps.replace('?', np.NaN)

# Impute the missing values with mean imputation
cc_apps.fillna(cc_apps.mean(), inplace = True)

# Count the number of NaNs in the dataset to verify
cc_apps.isnull().sum()
```




    0     12
    1     12
    2      0
    3      6
    4      6
    5      9
    6      9
    7      0
    8      0
    9      0
    10     0
    11     0
    12     0
    13    13
    14     0
    15     0
    dtype: int64



- The missing values present in the numeric columns are taken care of, but the columns with non-numeric data (columns 0, 1, 3, 4, 5, 6, and 13) still have some missing values to be imputed. 
- These missing values will be imputed with the most frequent values as present in the respective columns. 


```python
# Iterate over each column of cc_apps
for col in cc_apps.columns:
    # Check if the column is of object type
    if cc_apps[col].dtypes == 'object':
        # Impute with the most frequent value
        cc_apps = cc_apps.fillna(cc_apps[col].value_counts().index[0])
        
# Count the number of NaNs again to verify 
cc_apps.isnull().sum()
```




    0     0
    1     0
    2     0
    3     0
    4     0
    5     0
    6     0
    7     0
    8     0
    9     0
    10    0
    11    0
    12    0
    13    0
    14    0
    15    0
    dtype: int64



### Feature Engineering
- I will convert all the non-numeric values into numeric one using the label encoding technique.


```python
# Import LabelEncoder
from sklearn.preprocessing import LabelEncoder

# Instantiate LabelEncoder
le = LabelEncoder()

# Iterate over all the values of each column and extract their dtypes
for col in cc_apps.columns:
    # Compare if the dtype is object
    if cc_apps[col].dtypes == 'object':
        # Use LabelEncoder to do the numeric transformation
        cc_apps[col] = le.fit_transform(cc_apps[col].to_numpy())
```

- To scale the feature values to a uniform range, I will first split the dataset into train and test set.
- Also, I will drop unimportant features like 'DriversLicense(11)' and 'ZipCode(13)'.


```python
# Import train_test_split
from sklearn.model_selection import train_test_split

# Drop unimportant features and convert the DataFrame to a NumPy array
cc_apps = cc_apps.drop([11, 13], axis = 1)
cc_apps = cc_apps.to_numpy()

# Segregate features and labels into separate variables
X, y = cc_apps[:,:13], cc_apps[:,13]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
```

- Now the dataset is splited, I will apply the scaling. This is because ideally, no information from the test data should be used to scale the training data or should be used to direct the training process of a machine learning model. 


```python
# Import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

# Instantiate MinMaxScaler and use it to rescale X_train and X_test
scaler = MinMaxScaler(feature_range = (0, 1))
rescaledX_train = scaler.fit_transform(X_train)
rescaledX_test = scaler.transform(X_test)
```

## Modeling and Evaluation
### Logistic Regression
- I will use a Logistic Regression model and take a look at the model's confusion matrix. Checking confusion matrix is important because in the case of predicting credit card applications, it is equally important to see if the model is able to predict the approval status of the applications as denied that originally got denied.


```python
# Import LogisticRegression
from sklearn.linear_model import LogisticRegression

# Instantiate a LogisticRegression classifier
logreg = LogisticRegression()

# Fit logreg to the train set
logreg.fit(rescaledX_train, y_train)
```




    LogisticRegression()




```python
# Import confusion_matrix
from sklearn.metrics import confusion_matrix

# Use logreg to predict instances from the test set and store it
y_pred = logreg.predict(rescaledX_test)

# Get the accuracy score of logreg model and print it
print('Accuracy of logistic regression classifier: ', logreg.score(rescaledX_test, y_test))

# Print the confusion matrix of the logreg model
print(confusion_matrix(y_test, y_pred))
```

    Accuracy of logistic regression classifier:  0.8421052631578947
    [[94  9]
     [27 98]]


### GridSearchCV
- To improve the model performance (accuracy score of 84%), I will perform a grid search of logistic regression's hyperparameters 'tol' and 'max_iter'.
- I will also perform a cross-validation of 5 folds.


```python
# Import GridSearchCV
from sklearn.model_selection import GridSearchCV

# Define the grid of values for tol and max_iter
tol = [0.01, 0.001, 0.0001]
max_iter = [100, 150, 200]

# Create a dictionary where tol and max_iter are keys and the lists of their values are corresponding values
param_grid = dict(tol = tol, max_iter = max_iter)
```

- Finally, I will find the best performing model!


```python
# Instantiate GridSearchCV
grid_model = GridSearchCV(estimator = logreg, param_grid = param_grid, cv = 5)

# Use scaler to rescale X and assign it to rescaledX
rescaledX = scaler.fit_transform(X)

# Fit data to grid_model
grid_model_result = grid_model.fit(rescaledX, y)

# Summarize results
best_score, best_params = grid_model_result.best_score_, grid_model_result.best_params_
print("Best: %f using %s" % (best_score, best_params))
```

    Best: 0.850725 using {'max_iter': 100, 'tol': 0.01}


- Reference: [DataCamp](https://www.datacamp.com/)
