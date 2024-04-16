# Week 1 Assignment: Data Validation

[Tensorflow Data Validation (TFDV)](https://cloud.google.com/solutions/machine-learning/analyzing-and-validating-data-at-scale-for-ml-using-tfx) is an open-source library that helps to understand, validate, and monitor production machine learning (ML) data at scale. Common use-cases include comparing training, evaluation and serving datasets, as well as checking for training/serving skew. You have seen the core functionalities of this package in the previous ungraded lab and you will get to practice them in this week's assignment.

In this lab, you will use TFDV in order to:

* Generate and visualize statistics from a dataframe
* Infer a dataset schema
* Calculate, visualize and fix anomalies

Let's begin!

## Table of Contents

- [1 - Setup and Imports](#1)
- [2 - Load the Dataset](#2)
  - [2.1 - Read and Split the Dataset](#2-1)
    - [2.1.1 - Data Splits](#2-1-1)
    - [2.1.2 - Label Column](#2-1-2)
- [3 - Generate and Visualize Training Data Statistics](#3)
  - [3.1 - Removing Irrelevant Features](#3-1)
  - [Exercise 1 - Generate Training Statistics](#ex-1)
  - [Exercise 2 - Visualize Training Statistics](#ex-2)
- [4 - Infer a Data Schema](#4)
  - [Exercise 3: Infer the training set schema](#ex-3)
- [5 - Calculate, Visualize and Fix Evaluation Anomalies](#5)
  - [Exercise 4: Compare Training and Evaluation Statistics](#ex-4)
  - [Exercise 5: Detecting Anomalies](#ex-5)
  - [Exercise 6: Fix evaluation anomalies in the schema](#ex-6)
- [6 - Schema Environments](#6)
  - [Exercise 7: Check anomalies in the serving set](#ex-7)
  - [Exercise 8: Modifying the domain](#ex-8)
  - [Exercise 9: Detecting anomalies with environments](#ex-9)
- [7 - Check for Data Drift and Skew](#7)
- [8 - Display Stats for Data Slices](#8)
- [9 - Freeze the Schema](#8)


```python
# IMPORTANT: This will check your notebook's metadata for grading.
# Please do not continue the lab unless the output of this cell tells you to proceed. 
!python add_metadata.py --filename C2W1_Assignment.ipynb
```

    [32mGrader metadata detected! You can proceed with the lab![0m


_**NOTE:** To prevent errors from the autograder, you are not allowed to edit or delete non-graded cells in this notebook . Please only put your solutions in between the `### START CODE HERE` and `### END CODE HERE` code comments, and also refrain from adding any new cells. **Once you have passed this assignment** and want to experiment with any of the non-graded code, you may follow the instructions at the bottom of this notebook._

<a name='1'></a>
## 1 - Setup and Imports


```python
# grader-required-cell

# Import packages
import os
import pandas as pd
import tensorflow as tf
import tempfile, urllib, zipfile
import tensorflow_data_validation as tfdv


from tensorflow.python.lib.io import file_io
from tensorflow_data_validation.utils import slicing_util
from tensorflow_metadata.proto.v0.statistics_pb2 import DatasetFeatureStatisticsList, DatasetFeatureStatistics

# Set TF's logger to only display errors to avoid internal warnings being shown
tf.get_logger().setLevel('ERROR')
```

<a name='2'></a>
## 2 - Load the Dataset
You will be using the [Diabetes 130-US hospitals for years 1999-2008 Data Set](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008) donated to the University of California, Irvine (UCI) Machine Learning Repository. The dataset represents 10 years (1999-2008) of clinical care at 130 US hospitals and integrated delivery networks. It includes over 50 features representing patient and hospital outcomes.

This dataset has already been included in your Jupyter workspace so you can easily load it.

<a name='2-1'></a>
### 2.1 Read and Split the Dataset


```python
# grader-required-cell

# Read CSV data into a dataframe and recognize the missing data that is encoded with '?' string as NaN
df = pd.read_csv('data/diabetic_data.csv', header=0, na_values = '?')

# Preview the dataset
df.head()
```




<div>
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
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>encounter_id</th>
      <th>patient_nbr</th>
      <th>race</th>
      <th>gender</th>
      <th>age</th>
      <th>weight</th>
      <th>admission_type_id</th>
      <th>discharge_disposition_id</th>
      <th>admission_source_id</th>
      <th>time_in_hospital</th>
      <th>...</th>
      <th>citoglipton</th>
      <th>insulin</th>
      <th>glyburide-metformin</th>
      <th>glipizide-metformin</th>
      <th>glimepiride-pioglitazone</th>
      <th>metformin-rosiglitazone</th>
      <th>metformin-pioglitazone</th>
      <th>change</th>
      <th>diabetesMed</th>
      <th>readmitted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2278392</td>
      <td>8222157</td>
      <td>Caucasian</td>
      <td>Female</td>
      <td>[0-10)</td>
      <td>NaN</td>
      <td>6</td>
      <td>25</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>NO</td>
    </tr>
    <tr>
      <th>1</th>
      <td>149190</td>
      <td>55629189</td>
      <td>Caucasian</td>
      <td>Female</td>
      <td>[10-20)</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>3</td>
      <td>...</td>
      <td>No</td>
      <td>Up</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Ch</td>
      <td>Yes</td>
      <td>&gt;30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>64410</td>
      <td>86047875</td>
      <td>AfricanAmerican</td>
      <td>Female</td>
      <td>[20-30)</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>2</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>NO</td>
    </tr>
    <tr>
      <th>3</th>
      <td>500364</td>
      <td>82442376</td>
      <td>Caucasian</td>
      <td>Male</td>
      <td>[30-40)</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>2</td>
      <td>...</td>
      <td>No</td>
      <td>Up</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Ch</td>
      <td>Yes</td>
      <td>NO</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16680</td>
      <td>42519267</td>
      <td>Caucasian</td>
      <td>Male</td>
      <td>[40-50)</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>...</td>
      <td>No</td>
      <td>Steady</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Ch</td>
      <td>Yes</td>
      <td>NO</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 50 columns</p>
</div>



<a name='2-1-1'></a>
#### Data splits

In a production ML system, the model performance can be negatively affected by anomalies and divergence between data splits for training, evaluation, and serving. To emulate a production system, you will split the dataset into:

* 70% training set 
* 15% evaluation set
* 15% serving set

You will then use TFDV to visualize, analyze, and understand the data. You will create a data schema from the training dataset, then compare the evaluation and serving sets with this schema to detect anomalies and data drift/skew.

<a name='2-1-2'></a>
#### Label Column

This dataset has been prepared to analyze the factors related to readmission outcome. In this notebook, you will treat the `readmitted` column as the *target* or label column. 

The target (or label) is important to know while splitting the data into training, evaluation and serving sets. In supervised learning, you need to include the target in the training and evaluation datasets. For the serving set however (i.e. the set that simulates the data coming from your users), the **label column needs to be dropped** since that is the feature that your model will be trying to predict.

The following function returns the training, evaluation and serving partitions of a given dataset:


```python
# grader-required-cell

def prepare_data_splits_from_dataframe(df):
    '''
    Splits a Pandas Dataframe into training, evaluation and serving sets.

    Parameters:
            df : pandas dataframe to split

    Returns:
            train_df: Training dataframe(70% of the entire dataset)
            eval_df: Evaluation dataframe (15% of the entire dataset) 
            serving_df: Serving dataframe (15% of the entire dataset, label column dropped)
    '''
    
    # 70% of records for generating the training set
    train_len = int(len(df) * 0.7)
    
    # Remaining 30% of records for generating the evaluation and serving sets
    eval_serv_len = len(df) - train_len
    
    # Half of the 30%, which makes up 15% of total records, for generating the evaluation set
    eval_len = eval_serv_len // 2
    
    # Remaining 15% of total records for generating the serving set
    serv_len = eval_serv_len - eval_len 
 
    # Split the dataframe into the three subsets
    train_df = df.iloc[:train_len].reset_index(drop=True)
    eval_df = df.iloc[train_len: train_len + eval_len].reset_index(drop=True)
    serving_df = df.iloc[train_len + eval_len: train_len + eval_len + serv_len].reset_index(drop=True)
 
    # Serving data emulates the data that would be submitted for predictions, so it should not have the label column.
    serving_df = serving_df.drop(['readmitted'], axis=1)

    return train_df, eval_df, serving_df
```


```python
# grader-required-cell

# Split the datasets
train_df, eval_df, serving_df = prepare_data_splits_from_dataframe(df)
print('Training dataset has {} records\nValidation dataset has {} records\nServing dataset has {} records'.format(len(train_df),len(eval_df),len(serving_df)))
```

    Training dataset has 71236 records
    Validation dataset has 15265 records
    Serving dataset has 15265 records


**Note:** Depending on your use case, you might need to shuffle the dataset first before calling the function above. For example, some datasets are arranged by geographical location so directly splitting them might result in training sets that only come from locations that are not found in the evaluation and serving sets. That will likely result in a model that will only do well on the locations found in the training set. One way to avoid that kind of problem is to shuffle the dataset. For instance, you can do something like this:

```
# Shuffle the dataset
shuffled_df = df.sample(frac=1, random_state=48)

# Split the dataset
train_df, eval_df, serving_df = prepare_data_splits_from_dataframe(shuffled_df)
```

For grading purposes however, you will not shuffle the dataset in this lab. After you pass the assignment later and you want more practice, you can try to shuffle the data like above and observe the difference in the outputs. You might notice more anomalies so you will need to adjust the code on how to handle them.

<a name='3'></a>
## 3 - Generate and Visualize Training Data Statistics

In this section, you will be generating descriptive statistics from the dataset. This is usually the first step when dealing with a dataset you are not yet familiar with. It is also known as performing an *exploratory data analysis* and its purpose is to understand the data types, the data itself and any possible issues that need to be addressed.

It is important to mention that **exploratory data analysis should be perfomed on the training dataset** only. This is because getting information out of the evaluation or serving datasets can be seen as "cheating" since this data is used to emulate data that you have not collected yet and will try to predict using your ML algorithm. **In general, it is a good practice to avoid leaking information from your evaluation and serving data into your model.**

<a name='3-1'></a>
### Removing Irrelevant Features

Before you generate the statistics, you may want to drop irrelevant features from your dataset. You can do that with TFDV with the [tfdv.StatsOptions](https://www.tensorflow.org/tfx/data_validation/api_docs/python/tfdv/StatsOptions) class. It is usually **not a good idea** to drop features without knowing what information they contain. However there are times when this can be fairly obvious.

One of the important parameters of the `StatsOptions` class is `feature_allowlist`, which defines the features to include while calculating the data statistics. You can check the [documentation](https://www.tensorflow.org/tfx/data_validation/api_docs/python/tfdv/StatsOptions#args) to learn more about the class arguments.

In this case, you will omit the statistics for `encounter_id` and `patient_nbr` since they are part of the internal tracking of patients in the hospital and they don't contain valuable information for the task at hand.


```python
# grader-required-cell

# Define features to remove
features_to_remove = {'encounter_id', 'patient_nbr'}

# Collect features to include while computing the statistics
approved_cols = [col for col in df.columns if (col not in features_to_remove)]

# Instantiate a StatsOptions class and define the feature_allowlist property
stats_options = tfdv.StatsOptions(feature_allowlist=approved_cols)

# Review the features to generate the statistics
for feature in stats_options.feature_allowlist:
    print(feature)
```

    race
    gender
    age
    weight
    admission_type_id
    discharge_disposition_id
    admission_source_id
    time_in_hospital
    payer_code
    medical_specialty
    num_lab_procedures
    num_procedures
    num_medications
    number_outpatient
    number_emergency
    number_inpatient
    diag_1
    diag_2
    diag_3
    number_diagnoses
    max_glu_serum
    A1Cresult
    metformin
    repaglinide
    nateglinide
    chlorpropamide
    glimepiride
    acetohexamide
    glipizide
    glyburide
    tolbutamide
    pioglitazone
    rosiglitazone
    acarbose
    miglitol
    troglitazone
    tolazamide
    examide
    citoglipton
    insulin
    glyburide-metformin
    glipizide-metformin
    glimepiride-pioglitazone
    metformin-rosiglitazone
    metformin-pioglitazone
    change
    diabetesMed
    readmitted


<a name='ex-1'></a>
### Exercise 1: Generate Training Statistics 

TFDV allows you to generate statistics from different data formats such as CSV or a Pandas DataFrame. 

Since you already have the data stored in a DataFrame you can use the function [`tfdv.generate_statistics_from_dataframe()`](https://www.tensorflow.org/tfx/data_validation/api_docs/python/tfdv/generate_statistics_from_dataframe) which, given a DataFrame and `stats_options`, generates an object of type `DatasetFeatureStatisticsList`. This object includes the computed statistics of the given dataset.

Complete the cell below to generate the statistics of the training set. Remember to pass the training dataframe and the `stats_options` that you defined above as arguments.


```python
# grader-required-cell

### START CODE HERE
#following is the incorrecto way of removing the feature 
train_stats = tfdv.generate_statistics_from_dataframe(train_df, stats_options=stats_options)
### END CODE HERE
```


```python
# grader-required-cell

# TEST CODE

# get the number of features used to compute statistics
print(f"Number of features used: {len(train_stats.datasets[0].features)}")

# check the number of examples used
print(f"Number of examples used: {train_stats.datasets[0].num_examples}")

# check the column names of the first and last feature
print(f"First feature: {train_stats.datasets[0].features[0].path.step[0]}")
print(f"Last feature: {train_stats.datasets[0].features[-1].path.step[0]}")
```

    Number of features used: 48
    Number of examples used: 71236
    First feature: race
    Last feature: readmitted


**Expected Output:**

```
Number of features used: 48
Number of examples used: 71236
First feature: race
Last feature: readmitted
```

<a name='ex-2'></a>
### Exercise 2: Visualize Training Statistics

Now that you have the computed statistics in the `DatasetFeatureStatisticsList` instance, you will need a way to **visualize** these to get actual insights. TFDV provides this functionality through the method [`tfdv.visualize_statistics()`](https://www.tensorflow.org/tfx/data_validation/api_docs/python/tfdv/visualize_statistics).

Using this function in an interactive Python environment such as this one will output a very nice and convenient way to interact with the descriptive statistics you generated earlier. 

**Try it out yourself!** Remember to pass in the generated training statistics in the previous exercise as an argument.


```python
# grader-required-cell

### START CODE HERE
tfdv.visualize_statistics(train_stats)
### END CODE HERE
```


<iframe id='facets-iframe' width="100%" height="500px"></iframe>
        <script>
        facets_iframe = document.getElementById('facets-iframe');
        facets_html = '<script src="https://cdnjs.cloudflare.com/ajax/libs/webcomponentsjs/1.3.3/webcomponents-lite.js"><\/script><link rel="import" href="https://raw.githubusercontent.com/PAIR-code/facets/master/facets-dist/facets-jupyter.html"><facets-overview proto-input="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"></facets-overview>';
        facets_iframe.srcdoc = facets_html;
         facets_iframe.id = "";
         setTimeout(() => {
           facets_iframe.setAttribute('height', facets_iframe.contentWindow.document.body.offsetHeight + 'px')
         }, 1500)
         </script>


<a name='4'></a>
## 4 - Infer a data schema

A schema defines the **properties of the data** and can thus be used to detect errors. Some of these properties include:

- which features are expected to be present
- feature type
- the number of values for a feature in each example
- the presence of each feature across all examples
- the expected domains of features

The schema is expected to be fairly static, whereas statistics can vary per data split. So, you will **infer the data schema from only the training dataset**. Later, you will generate statistics for evaluation and serving datasets and compare their state with the data schema to detect anomalies, drift and skew.

<a name='ex-3'></a>
### Exercise 3: Infer the training set schema

Schema inference is straightforward using [`tfdv.infer_schema()`](https://www.tensorflow.org/tfx/data_validation/api_docs/python/tfdv/infer_schema). This function needs only the **statistics** (an instance of `DatasetFeatureStatisticsList`) of your data as input. The output will be a Schema [protocol buffer](https://developers.google.com/protocol-buffers) containing the results.

A complimentary function is [`tfdv.display_schema()`](https://www.tensorflow.org/tfx/data_validation/api_docs/python/tfdv/display_schema) for displaying the schema in a table. This accepts a **Schema** protocol buffer as input.

Fill the code below to infer the schema from the training statistics using TFDV and display the result.


```python
# grader-required-cell

### START CODE HERE
# Infer the data schema by using the training statistics that you generated
schema = tfdv.infer_schema(statistics=train_stats)


# Display the data schema
tfdv.display_schema(schema)
### END CODE HERE
```


<div>
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
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Type</th>
      <th>Presence</th>
      <th>Valency</th>
      <th>Domain</th>
    </tr>
    <tr>
      <th>Feature name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>'race'</th>
      <td>STRING</td>
      <td>optional</td>
      <td>single</td>
      <td>'race'</td>
    </tr>
    <tr>
      <th>'gender'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'gender'</td>
    </tr>
    <tr>
      <th>'age'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'age'</td>
    </tr>
    <tr>
      <th>'weight'</th>
      <td>STRING</td>
      <td>optional</td>
      <td>single</td>
      <td>'weight'</td>
    </tr>
    <tr>
      <th>'admission_type_id'</th>
      <td>INT</td>
      <td>required</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <th>'discharge_disposition_id'</th>
      <td>INT</td>
      <td>required</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <th>'admission_source_id'</th>
      <td>INT</td>
      <td>required</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <th>'time_in_hospital'</th>
      <td>INT</td>
      <td>required</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <th>'payer_code'</th>
      <td>STRING</td>
      <td>optional</td>
      <td>single</td>
      <td>'payer_code'</td>
    </tr>
    <tr>
      <th>'medical_specialty'</th>
      <td>STRING</td>
      <td>optional</td>
      <td>single</td>
      <td>'medical_specialty'</td>
    </tr>
    <tr>
      <th>'num_lab_procedures'</th>
      <td>INT</td>
      <td>required</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <th>'num_procedures'</th>
      <td>INT</td>
      <td>required</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <th>'num_medications'</th>
      <td>INT</td>
      <td>required</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <th>'number_outpatient'</th>
      <td>INT</td>
      <td>required</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <th>'number_emergency'</th>
      <td>INT</td>
      <td>required</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <th>'number_inpatient'</th>
      <td>INT</td>
      <td>required</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <th>'diag_1'</th>
      <td>BYTES</td>
      <td>optional</td>
      <td>single</td>
      <td>-</td>
    </tr>
    <tr>
      <th>'diag_2'</th>
      <td>BYTES</td>
      <td>optional</td>
      <td>single</td>
      <td>-</td>
    </tr>
    <tr>
      <th>'diag_3'</th>
      <td>BYTES</td>
      <td>optional</td>
      <td>single</td>
      <td>-</td>
    </tr>
    <tr>
      <th>'number_diagnoses'</th>
      <td>INT</td>
      <td>required</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <th>'max_glu_serum'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'max_glu_serum'</td>
    </tr>
    <tr>
      <th>'A1Cresult'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'A1Cresult'</td>
    </tr>
    <tr>
      <th>'metformin'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'metformin'</td>
    </tr>
    <tr>
      <th>'repaglinide'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'repaglinide'</td>
    </tr>
    <tr>
      <th>'nateglinide'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'nateglinide'</td>
    </tr>
    <tr>
      <th>'chlorpropamide'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'chlorpropamide'</td>
    </tr>
    <tr>
      <th>'glimepiride'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'glimepiride'</td>
    </tr>
    <tr>
      <th>'acetohexamide'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'acetohexamide'</td>
    </tr>
    <tr>
      <th>'glipizide'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'glipizide'</td>
    </tr>
    <tr>
      <th>'glyburide'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'glyburide'</td>
    </tr>
    <tr>
      <th>'tolbutamide'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'tolbutamide'</td>
    </tr>
    <tr>
      <th>'pioglitazone'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'pioglitazone'</td>
    </tr>
    <tr>
      <th>'rosiglitazone'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'rosiglitazone'</td>
    </tr>
    <tr>
      <th>'acarbose'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'acarbose'</td>
    </tr>
    <tr>
      <th>'miglitol'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'miglitol'</td>
    </tr>
    <tr>
      <th>'troglitazone'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'troglitazone'</td>
    </tr>
    <tr>
      <th>'tolazamide'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'tolazamide'</td>
    </tr>
    <tr>
      <th>'examide'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'examide'</td>
    </tr>
    <tr>
      <th>'citoglipton'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'citoglipton'</td>
    </tr>
    <tr>
      <th>'insulin'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'insulin'</td>
    </tr>
    <tr>
      <th>'glyburide-metformin'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'glyburide-metformin'</td>
    </tr>
    <tr>
      <th>'glipizide-metformin'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'glipizide-metformin'</td>
    </tr>
    <tr>
      <th>'glimepiride-pioglitazone'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'glimepiride-pioglitazone'</td>
    </tr>
    <tr>
      <th>'metformin-rosiglitazone'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'metformin-rosiglitazone'</td>
    </tr>
    <tr>
      <th>'metformin-pioglitazone'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'metformin-pioglitazone'</td>
    </tr>
    <tr>
      <th>'change'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'change'</td>
    </tr>
    <tr>
      <th>'diabetesMed'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'diabetesMed'</td>
    </tr>
    <tr>
      <th>'readmitted'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'readmitted'</td>
    </tr>
  </tbody>
</table>
</div>



<div>
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
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Values</th>
    </tr>
    <tr>
      <th>Domain</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>'race'</th>
      <td>'AfricanAmerican', 'Asian', 'Caucasian', 'Hispanic', 'Other'</td>
    </tr>
    <tr>
      <th>'gender'</th>
      <td>'Female', 'Male', 'Unknown/Invalid'</td>
    </tr>
    <tr>
      <th>'age'</th>
      <td>'[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'</td>
    </tr>
    <tr>
      <th>'weight'</th>
      <td>'&gt;200', '[0-25)', '[100-125)', '[125-150)', '[150-175)', '[175-200)', '[25-50)', '[50-75)', '[75-100)'</td>
    </tr>
    <tr>
      <th>'payer_code'</th>
      <td>'BC', 'CH', 'CM', 'CP', 'DM', 'HM', 'MC', 'MD', 'MP', 'OG', 'OT', 'PO', 'SI', 'SP', 'UN', 'WC'</td>
    </tr>
    <tr>
      <th>'medical_specialty'</th>
      <td>'AllergyandImmunology', 'Anesthesiology', 'Anesthesiology-Pediatric', 'Cardiology', 'Cardiology-Pediatric', 'Dentistry', 'Dermatology', 'Emergency/Trauma', 'Endocrinology', 'Family/GeneralPractice', 'Gastroenterology', 'Gynecology', 'Hematology', 'Hematology/Oncology', 'Hospitalist', 'InfectiousDiseases', 'InternalMedicine', 'Nephrology', 'Neurology', 'Obsterics&amp;Gynecology-GynecologicOnco', 'Obstetrics', 'ObstetricsandGynecology', 'Oncology', 'Ophthalmology', 'Orthopedics', 'Orthopedics-Reconstructive', 'Osteopath', 'Otolaryngology', 'OutreachServices', 'Pathology', 'Pediatrics', 'Pediatrics-AllergyandImmunology', 'Pediatrics-CriticalCare', 'Pediatrics-EmergencyMedicine', 'Pediatrics-Endocrinology', 'Pediatrics-Hematology-Oncology', 'Pediatrics-InfectiousDiseases', 'Pediatrics-Neurology', 'Pediatrics-Pulmonology', 'Perinatology', 'PhysicalMedicineandRehabilitation', 'PhysicianNotFound', 'Podiatry', 'Proctology', 'Psychiatry', 'Psychiatry-Addictive', 'Psychiatry-Child/Adolescent', 'Psychology', 'Pulmonology', 'Radiologist', 'Radiology', 'Rheumatology', 'Speech', 'SportsMedicine', 'Surgeon', 'Surgery-Cardiovascular', 'Surgery-Cardiovascular/Thoracic', 'Surgery-Colon&amp;Rectal', 'Surgery-General', 'Surgery-Maxillofacial', 'Surgery-Neuro', 'Surgery-Pediatric', 'Surgery-Plastic', 'Surgery-PlasticwithinHeadandNeck', 'Surgery-Thoracic', 'Surgery-Vascular', 'SurgicalSpecialty', 'Urology'</td>
    </tr>
    <tr>
      <th>'max_glu_serum'</th>
      <td>'&gt;200', '&gt;300', 'None', 'Norm'</td>
    </tr>
    <tr>
      <th>'A1Cresult'</th>
      <td>'&gt;7', '&gt;8', 'None', 'Norm'</td>
    </tr>
    <tr>
      <th>'metformin'</th>
      <td>'Down', 'No', 'Steady', 'Up'</td>
    </tr>
    <tr>
      <th>'repaglinide'</th>
      <td>'Down', 'No', 'Steady', 'Up'</td>
    </tr>
    <tr>
      <th>'nateglinide'</th>
      <td>'Down', 'No', 'Steady', 'Up'</td>
    </tr>
    <tr>
      <th>'chlorpropamide'</th>
      <td>'Down', 'No', 'Steady', 'Up'</td>
    </tr>
    <tr>
      <th>'glimepiride'</th>
      <td>'Down', 'No', 'Steady', 'Up'</td>
    </tr>
    <tr>
      <th>'acetohexamide'</th>
      <td>'No', 'Steady'</td>
    </tr>
    <tr>
      <th>'glipizide'</th>
      <td>'Down', 'No', 'Steady', 'Up'</td>
    </tr>
    <tr>
      <th>'glyburide'</th>
      <td>'Down', 'No', 'Steady', 'Up'</td>
    </tr>
    <tr>
      <th>'tolbutamide'</th>
      <td>'No', 'Steady'</td>
    </tr>
    <tr>
      <th>'pioglitazone'</th>
      <td>'Down', 'No', 'Steady', 'Up'</td>
    </tr>
    <tr>
      <th>'rosiglitazone'</th>
      <td>'Down', 'No', 'Steady', 'Up'</td>
    </tr>
    <tr>
      <th>'acarbose'</th>
      <td>'Down', 'No', 'Steady', 'Up'</td>
    </tr>
    <tr>
      <th>'miglitol'</th>
      <td>'Down', 'No', 'Steady', 'Up'</td>
    </tr>
    <tr>
      <th>'troglitazone'</th>
      <td>'No', 'Steady'</td>
    </tr>
    <tr>
      <th>'tolazamide'</th>
      <td>'No', 'Steady', 'Up'</td>
    </tr>
    <tr>
      <th>'examide'</th>
      <td>'No'</td>
    </tr>
    <tr>
      <th>'citoglipton'</th>
      <td>'No'</td>
    </tr>
    <tr>
      <th>'insulin'</th>
      <td>'Down', 'No', 'Steady', 'Up'</td>
    </tr>
    <tr>
      <th>'glyburide-metformin'</th>
      <td>'Down', 'No', 'Steady', 'Up'</td>
    </tr>
    <tr>
      <th>'glipizide-metformin'</th>
      <td>'No', 'Steady'</td>
    </tr>
    <tr>
      <th>'glimepiride-pioglitazone'</th>
      <td>'No'</td>
    </tr>
    <tr>
      <th>'metformin-rosiglitazone'</th>
      <td>'No'</td>
    </tr>
    <tr>
      <th>'metformin-pioglitazone'</th>
      <td>'No'</td>
    </tr>
    <tr>
      <th>'change'</th>
      <td>'Ch', 'No'</td>
    </tr>
    <tr>
      <th>'diabetesMed'</th>
      <td>'No', 'Yes'</td>
    </tr>
    <tr>
      <th>'readmitted'</th>
      <td>'&lt;30', '&gt;30', 'NO'</td>
    </tr>
  </tbody>
</table>
</div>



```python
# grader-required-cell

# TEST CODE

# Check number of features
print(f"Number of features in schema: {len(schema.feature)}")

# Check domain name of 2nd feature
print(f"Second feature in schema: {list(schema.feature)[1].domain}")
```

    Number of features in schema: 48
    Second feature in schema: gender


**Expected Output:**

```
Number of features in schema: 48
Second feature in schema: gender
```

**Be sure to check the information displayed before moving forward.**

<a name='5'></a>
## 5 - Calculate, Visualize and Fix Evaluation Anomalies


It is important that the schema of the evaluation data is consistent with the training data since the data that your model is going to receive should be consistent to the one you used to train it with.

Moreover, it is also important that the **features of the evaluation data belong roughly to the same range as the training data**. This ensures that the model will be evaluated on a similar loss surface covered during training.

<a name='ex-4'></a>
### Exercise 4: Compare Training and Evaluation Statistics

Now you are going to generate the evaluation statistics and compare it with training statistics. You can use the [`tfdv.generate_statistics_from_dataframe()`](https://www.tensorflow.org/tfx/data_validation/api_docs/python/tfdv/generate_statistics_from_dataframe) function for this. But this time, you'll need to pass the **evaluation data**. For the `stats_options` parameter, the list you used before works here too.

Remember that to visualize the evaluation statistics you can use [`tfdv.visualize_statistics()`](https://www.tensorflow.org/tfx/data_validation/api_docs/python/tfdv/visualize_statistics). 

However, it is impractical to visualize both statistics separately and do your comparison from there. Fortunately, TFDV has got this covered. You can use the `visualize_statistics` function and pass additional parameters to overlay the statistics from both datasets (referenced as left-hand side and right-hand side statistics). Let's see what these parameters are:

- `lhs_statistics`: Required parameter. Expects an instance of `DatasetFeatureStatisticsList `.


- `rhs_statistics`: Expects an instance of `DatasetFeatureStatisticsList ` to compare with `lhs_statistics`.


- `lhs_name`: Name of the `lhs_statistics` dataset.


- `rhs_name`: Name of the `rhs_statistics` dataset.

For this case, remember to define the `lhs_statistics` protocol with the `eval_stats`, and the optional `rhs_statistics` protocol with the `train_stats`.

Additionally, check the function for the protocol name declaration, and define the lhs and rhs names as `'EVAL_DATASET'` and `'TRAIN_DATASET'` respectively.


```python
# grader-required-cell

### START CODE HERE
# Generate evaluation dataset statistics
# HINT: Remember to use the evaluation dataframe and to pass the stats_options (that you defined before) as an argument
eval_stats = tfdv.generate_statistics_from_dataframe(eval_df, stats_options=stats_options)

# Compare evaluation data with training data 
# HINT: Remember to use both the evaluation and training statistics with the lhs_statistics and rhs_statistics arguments
# HINT: Assign the names of 'EVAL_DATASET' and 'TRAIN_DATASET' to the lhs and rhs protocols
tfdv.visualize_statistics(lhs_statistics=eval_stats, 
                          rhs_statistics=train_stats,
                          lhs_name='EVAL_DATASET',
                          rhs_name='TRAIN_DATASET')

                      
### END CODE HERE
```


<iframe id='facets-iframe' width="100%" height="500px"></iframe>
        <script>
        facets_iframe = document.getElementById('facets-iframe');
        facets_html = '<script src="https://cdnjs.cloudflare.com/ajax/libs/webcomponentsjs/1.3.3/webcomponents-lite.js"><\/script><link rel="import" href="https://raw.githubusercontent.com/PAIR-code/facets/master/facets-dist/facets-jupyter.html"><facets-overview proto-input="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AAA/QAohCCQQJCISSW5mZWN0aW91c0Rpc2Vhc2VzKQAAAAAAADpACiUIJRAlIhZQZWRpYXRyaWNzLVB1bG1vbm9sb2d5KQAAAAAAADlACjMIJhAmIiRPYnN0ZXJpY3MmR3luZWNvbG9neS1HeW5lY29sb2dpY09uY28pAAAAAAAAOUAKHAgnECciDU9waHRoYWxtb2xvZ3kpAAAAAAAANkAKJwgoECgiGEFuZXN0aGVzaW9sb2d5LVBlZGlhdHJpYykAAAAAAAAzQAoYCCkQKSIJUGF0aG9sb2d5KQAAAAAAADFAChoIKhAqIgtIb3NwaXRhbGlzdCkAAAAAAAAwQAobCCsQKyIMUmhldW1hdG9sb2d5KQAAAAAAACxAChkILBAsIgpPYnN0ZXRyaWNzKQAAAAAAAChACiQILRAtIhVTdXJnZXJ5LU1heGlsbG9mYWNpYWwpAAAAAAAAJkAKIwguEC4iFFN1cmdlcnktQ29sb24mUmVjdGFsKQAAAAAAACZACh0ILxAvIg5BbmVzdGhlc2lvbG9neSkAAAAAAAAmQAojCDAQMCIUUGVkaWF0cmljcy1OZXVyb2xvZ3kpAAAAAAAAJEAKIAgxEDEiEVN1cmdlcnktUGVkaWF0cmljKQAAAAAAACBACioIMhAyIhtQc3ljaGlhdHJ5LUNoaWxkL0Fkb2xlc2NlbnQpAAAAAAAAHEAKHwgzEDMiEE91dHJlYWNoU2VydmljZXMpAAAAAAAAHEAKIwg0EDQiFENhcmRpb2xvZ3ktUGVkaWF0cmljKQAAAAAAABxACiMINRA1IhRBbGxlcmd5YW5kSW1tdW5vbG9neSkAAAAAAAAcQAogCDYQNiIRU3VyZ2ljYWxTcGVjaWFsdHkpAAAAAAAAGEAKIAg3EDciEVBoeXNpY2lhbk5vdEZvdW5kKQAAAAAAABhACi0IOBA4Ih5QZWRpYXRyaWNzLUhlbWF0b2xvZ3ktT25jb2xvZ3kpAAAAAAAAEEAKGAg5EDkiCURlbnRpc3RyeSkAAAAAAAAQQAorCDoQOiIcUGVkaWF0cmljcy1FbWVyZ2VuY3lNZWRpY2luZSkAAAAAAAAIQAouCDsQOyIfUGVkaWF0cmljcy1BbGxlcmd5YW5kSW1tdW5vbG9neSkAAAAAAAAIQAovCDwQPCIgU3VyZ2VyeS1QbGFzdGljd2l0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"></facets-overview>';
        facets_iframe.srcdoc = facets_html;
         facets_iframe.id = "";
         setTimeout(() => {
           facets_iframe.setAttribute('height', facets_iframe.contentWindow.document.body.offsetHeight + 'px')
         }, 1500)
         </script>



```python
# grader-required-cell

# TEST CODE

# get the number of features used to compute statistics
print(f"Number of features: {len(eval_stats.datasets[0].features)}")

# check the number of examples used
print(f"Number of examples: {eval_stats.datasets[0].num_examples}")

# check the column names of the first and last feature
print(f"First feature: {eval_stats.datasets[0].features[0].path.step[0]}")
print(f"Last feature: {eval_stats.datasets[0].features[-1].path.step[0]}")
```

    Number of features: 48
    Number of examples: 15265
    First feature: race
    Last feature: readmitted


**Expected Output:**

```
Number of features: 48
Number of examples: 15265
First feature: race
Last feature: readmitted
```

<a name='ex-5'></a>
### Exercise 5: Detecting Anomalies ###

At this point, you should ask if your evaluation dataset matches the schema from your training dataset. For instance, if you scroll through the output cell in the previous exercise, you can see that the categorical feature **glimepiride-pioglitazone** has 1 unique value in the training set while the evaluation dataset has 2. You can verify with the built-in Pandas `describe()` method as well.


```python
# grader-required-cell

train_df["glimepiride-pioglitazone"].describe()
```




    count     71236
    unique    1    
    top       No   
    freq      71236
    Name: glimepiride-pioglitazone, dtype: object




```python
# grader-required-cell

eval_df["glimepiride-pioglitazone"].describe()
```




    count     15265
    unique    2    
    top       No   
    freq      15264
    Name: glimepiride-pioglitazone, dtype: object



It is possible but highly inefficient to visually inspect and determine all the anomalies. So, let's instead use TFDV functions to detect and display these.

You can use the function [`tfdv.validate_statistics()`](https://www.tensorflow.org/tfx/data_validation/api_docs/python/tfdv/validate_statistics) for detecting anomalies and [`tfdv.display_anomalies()`](https://www.tensorflow.org/tfx/data_validation/api_docs/python/tfdv/display_anomalies) for displaying them.

The `validate_statistics()` method has two required arguments:
- an instance of `DatasetFeatureStatisticsList`
- an instance of `Schema`

Fill in the following graded function which, given the statistics and schema, displays the anomalies found.


```python
# grader-required-cell

def calculate_and_display_anomalies(statistics, schema):
    '''
    Calculate and display anomalies.

            Parameters:
                    statistics : Data statistics in statistics_pb2.DatasetFeatureStatisticsList format
                    schema : Data schema in schema_pb2.Schema format

            Returns:
                    display of calculated anomalies
    '''
    ### START CODE HERE
    # HINTS: Pass the statistics and schema parameters into the validation function 
    anomalies = tfdv.validate_statistics(statistics=statistics, schema=schema)
    
    # HINTS: Display input anomalies by using the calculated anomalies
    tfdv.display_anomalies(anomalies)
    ### END CODE HERE
```

You should see detected anomalies in the `medical_specialty` and `glimepiride-pioglitazone` features by running the cell below.


```python
# grader-required-cell

# Check evaluation data for errors by validating the evaluation data staticss using the previously inferred schema
calculate_and_display_anomalies(eval_stats, schema=schema)
```


<div>
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
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Anomaly short description</th>
      <th>Anomaly long description</th>
    </tr>
    <tr>
      <th>Feature name</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>'glimepiride-pioglitazone'</th>
      <td>Unexpected string values</td>
      <td>Examples contain values missing from the schema: Steady (&lt;1%).</td>
    </tr>
    <tr>
      <th>'medical_specialty'</th>
      <td>Unexpected string values</td>
      <td>Examples contain values missing from the schema: Neurophysiology (&lt;1%).</td>
    </tr>
  </tbody>
</table>
</div>


<a name='ex-6'></a>
### Exercise 6: Fix evaluation anomalies in the schema

The evaluation data has records with values for the features **glimepiride-pioglitazone** and **medical_speciality**  that were not included in the schema generated from the training data. You can fix this by adding the new values that exist in the evaluation dataset to the domain of these features.

To get the `domain` of a particular feature you can use [`tfdv.get_domain()`](https://www.tensorflow.org/tfx/data_validation/api_docs/python/tfdv/get_domain).

You can use the `append()` method to the `value` property of the returned `domain` to add strings to the valid list of values. To be more explicit, given a domain you can do something like:

```python
domain.value.append("feature_value")

```


```python
# grader-required-cell

### START CODE HERE

# Get the domain associated with the input feature, glimepiride-pioglitazone, from the schema
glimepiride_pioglitazone_domain = tfdv.get_domain(schema, 'glimepiride-pioglitazone')

# HINT: Append the missing value 'Steady' to the domain
glimepiride_pioglitazone_domain.value.append('Steady')

# Get the domain associated with the input feature, medical_specialty, from the schema
medical_specialty_domain = tfdv.get_domain(schema, 'medical_specialty')

# HINT: Append the missing value 'Neurophysiology' to the domain
medical_specialty_domain.value.append('Neurophysiology')

# HINT: Re-calculate and re-display anomalies with the new schema
calculate_and_display_anomalies(eval_stats, schema=schema)
### END CODE HERE
```


<h4 style="color:green;">No anomalies found.</h4>


If you did the exercise correctly, you should see _"No anomalies found."_ after running the cell above.

<a name='6'></a>
## 6 - Schema Environments

By default, all datasets in a pipeline should use the same schema. However, there are some exceptions. 

For example, the **label column is dropped in the serving set** so this will be flagged when comparing with the training set schema. 

**In this case, introducing slight schema variations is necessary.**

<a name='ex-7'></a>
### Exercise 7: Check anomalies in the serving set

Now you are going to check for anomalies in the **serving data**. The process is very similar to the one you previously did for the evaluation data with a little change. 

Let's create a new `StatsOptions` that is aware of the information provided by the schema and use it when generating statistics from the serving DataFrame.


```python
# grader-required-cell

# Define a new statistics options by the tfdv.StatsOptions class for the serving data by passing the previously inferred schema
options = tfdv.StatsOptions(schema=schema, 
                            infer_type_from_schema=True, 
                            feature_allowlist=approved_cols)
```


```python
# grader-required-cell

### START CODE HERE
# Generate serving dataset statistics
# HINT: Remember to use the serving dataframe and to pass the newly defined statistics options
serving_stats = tfdv.generate_statistics_from_dataframe(serving_df, stats_options=options)

# HINT: Calculate and display anomalies using the generated serving statistics
calculate_and_display_anomalies(serving_stats, schema=schema)

### END CODE HERE
```


<div>
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
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Anomaly short description</th>
      <th>Anomaly long description</th>
    </tr>
    <tr>
      <th>Feature name</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>'medical_specialty'</th>
      <td>Unexpected string values</td>
      <td>Examples contain values missing from the schema: DCPTEAM (&lt;1%), Endocrinology-Metabolism (&lt;1%), Resident (&lt;1%).</td>
    </tr>
    <tr>
      <th>'payer_code'</th>
      <td>Unexpected string values</td>
      <td>Examples contain values missing from the schema: FR (&lt;1%).</td>
    </tr>
    <tr>
      <th>'readmitted'</th>
      <td>Column dropped</td>
      <td>Column is completely missing</td>
    </tr>
    <tr>
      <th>'metformin-rosiglitazone'</th>
      <td>Unexpected string values</td>
      <td>Examples contain values missing from the schema: Steady (&lt;1%).</td>
    </tr>
    <tr>
      <th>'metformin-pioglitazone'</th>
      <td>Unexpected string values</td>
      <td>Examples contain values missing from the schema: Steady (&lt;1%).</td>
    </tr>
  </tbody>
</table>
</div>


You should see that `metformin-rosiglitazone`, `metformin-pioglitazone`, `payer_code` and `medical_specialty` features have an anomaly (i.e. Unexpected string values) which is less than 1%. 

Let's **relax the anomaly detection constraints** for the last two of these features by defining the `min_domain_mass` of the feature's distribution constraints.


```python
# grader-required-cell

# This relaxes the minimum fraction of values that must come from the domain for the feature.

# Get the feature and relax to match 90% of the domain
payer_code = tfdv.get_feature(schema, 'payer_code')
payer_code.distribution_constraints.min_domain_mass = 0.9 

# Get the feature and relax to match 90% of the domain
medical_specialty = tfdv.get_feature(schema, 'medical_specialty')
medical_specialty.distribution_constraints.min_domain_mass = 0.9 

# Detect anomalies with the updated constraints
calculate_and_display_anomalies(serving_stats, schema=schema)
```


<div>
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
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Anomaly short description</th>
      <th>Anomaly long description</th>
    </tr>
    <tr>
      <th>Feature name</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>'metformin-pioglitazone'</th>
      <td>Unexpected string values</td>
      <td>Examples contain values missing from the schema: Steady (&lt;1%).</td>
    </tr>
    <tr>
      <th>'metformin-rosiglitazone'</th>
      <td>Unexpected string values</td>
      <td>Examples contain values missing from the schema: Steady (&lt;1%).</td>
    </tr>
    <tr>
      <th>'readmitted'</th>
      <td>Column dropped</td>
      <td>Column is completely missing</td>
    </tr>
  </tbody>
</table>
</div>


If the `payer_code` and `medical_specialty` are no longer part of the output cell, then the relaxation worked!

<a name='ex-8'></a>
### Exercise 8: Modifying the Domain

Let's investigate the possible cause of the anomalies for the other features, namely `metformin-pioglitazone` and `metformin-rosiglitazone`. From the output of the previous exercise, you'll see that the `anomaly long description` says: "Examples contain values missing from the schema: Steady (<1%)". You can redisplay the schema and look at the domain of these features to verify this statement.

When you inferred the schema at the start of this lab, it's possible that some  values were not detected in the training data so it was not included in the expected domain values of the feature's schema. In the case of `metformin-rosiglitazone` and `metformin-pioglitazone`, the value "Steady" is indeed missing. You will just see "No" in the domain of these two features after running the code cell below.


```python
# grader-required-cell

tfdv.display_schema(schema)
```


<div>
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
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Type</th>
      <th>Presence</th>
      <th>Valency</th>
      <th>Domain</th>
    </tr>
    <tr>
      <th>Feature name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>'race'</th>
      <td>STRING</td>
      <td>optional</td>
      <td>single</td>
      <td>'race'</td>
    </tr>
    <tr>
      <th>'gender'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'gender'</td>
    </tr>
    <tr>
      <th>'age'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'age'</td>
    </tr>
    <tr>
      <th>'weight'</th>
      <td>STRING</td>
      <td>optional</td>
      <td>single</td>
      <td>'weight'</td>
    </tr>
    <tr>
      <th>'admission_type_id'</th>
      <td>INT</td>
      <td>required</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <th>'discharge_disposition_id'</th>
      <td>INT</td>
      <td>required</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <th>'admission_source_id'</th>
      <td>INT</td>
      <td>required</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <th>'time_in_hospital'</th>
      <td>INT</td>
      <td>required</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <th>'payer_code'</th>
      <td>STRING</td>
      <td>optional</td>
      <td>single</td>
      <td>'payer_code'</td>
    </tr>
    <tr>
      <th>'medical_specialty'</th>
      <td>STRING</td>
      <td>optional</td>
      <td>single</td>
      <td>'medical_specialty'</td>
    </tr>
    <tr>
      <th>'num_lab_procedures'</th>
      <td>INT</td>
      <td>required</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <th>'num_procedures'</th>
      <td>INT</td>
      <td>required</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <th>'num_medications'</th>
      <td>INT</td>
      <td>required</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <th>'number_outpatient'</th>
      <td>INT</td>
      <td>required</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <th>'number_emergency'</th>
      <td>INT</td>
      <td>required</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <th>'number_inpatient'</th>
      <td>INT</td>
      <td>required</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <th>'diag_1'</th>
      <td>BYTES</td>
      <td>optional</td>
      <td>single</td>
      <td>-</td>
    </tr>
    <tr>
      <th>'diag_2'</th>
      <td>BYTES</td>
      <td>optional</td>
      <td>single</td>
      <td>-</td>
    </tr>
    <tr>
      <th>'diag_3'</th>
      <td>BYTES</td>
      <td>optional</td>
      <td>single</td>
      <td>-</td>
    </tr>
    <tr>
      <th>'number_diagnoses'</th>
      <td>INT</td>
      <td>required</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <th>'max_glu_serum'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'max_glu_serum'</td>
    </tr>
    <tr>
      <th>'A1Cresult'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'A1Cresult'</td>
    </tr>
    <tr>
      <th>'metformin'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'metformin'</td>
    </tr>
    <tr>
      <th>'repaglinide'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'repaglinide'</td>
    </tr>
    <tr>
      <th>'nateglinide'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'nateglinide'</td>
    </tr>
    <tr>
      <th>'chlorpropamide'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'chlorpropamide'</td>
    </tr>
    <tr>
      <th>'glimepiride'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'glimepiride'</td>
    </tr>
    <tr>
      <th>'acetohexamide'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'acetohexamide'</td>
    </tr>
    <tr>
      <th>'glipizide'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'glipizide'</td>
    </tr>
    <tr>
      <th>'glyburide'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'glyburide'</td>
    </tr>
    <tr>
      <th>'tolbutamide'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'tolbutamide'</td>
    </tr>
    <tr>
      <th>'pioglitazone'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'pioglitazone'</td>
    </tr>
    <tr>
      <th>'rosiglitazone'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'rosiglitazone'</td>
    </tr>
    <tr>
      <th>'acarbose'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'acarbose'</td>
    </tr>
    <tr>
      <th>'miglitol'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'miglitol'</td>
    </tr>
    <tr>
      <th>'troglitazone'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'troglitazone'</td>
    </tr>
    <tr>
      <th>'tolazamide'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'tolazamide'</td>
    </tr>
    <tr>
      <th>'examide'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'examide'</td>
    </tr>
    <tr>
      <th>'citoglipton'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'citoglipton'</td>
    </tr>
    <tr>
      <th>'insulin'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'insulin'</td>
    </tr>
    <tr>
      <th>'glyburide-metformin'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'glyburide-metformin'</td>
    </tr>
    <tr>
      <th>'glipizide-metformin'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'glipizide-metformin'</td>
    </tr>
    <tr>
      <th>'glimepiride-pioglitazone'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'glimepiride-pioglitazone'</td>
    </tr>
    <tr>
      <th>'metformin-rosiglitazone'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'metformin-rosiglitazone'</td>
    </tr>
    <tr>
      <th>'metformin-pioglitazone'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'metformin-pioglitazone'</td>
    </tr>
    <tr>
      <th>'change'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'change'</td>
    </tr>
    <tr>
      <th>'diabetesMed'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'diabetesMed'</td>
    </tr>
    <tr>
      <th>'readmitted'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'readmitted'</td>
    </tr>
  </tbody>
</table>
</div>



<div>
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
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Values</th>
    </tr>
    <tr>
      <th>Domain</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>'race'</th>
      <td>'AfricanAmerican', 'Asian', 'Caucasian', 'Hispanic', 'Other'</td>
    </tr>
    <tr>
      <th>'gender'</th>
      <td>'Female', 'Male', 'Unknown/Invalid'</td>
    </tr>
    <tr>
      <th>'age'</th>
      <td>'[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'</td>
    </tr>
    <tr>
      <th>'weight'</th>
      <td>'&gt;200', '[0-25)', '[100-125)', '[125-150)', '[150-175)', '[175-200)', '[25-50)', '[50-75)', '[75-100)'</td>
    </tr>
    <tr>
      <th>'payer_code'</th>
      <td>'BC', 'CH', 'CM', 'CP', 'DM', 'HM', 'MC', 'MD', 'MP', 'OG', 'OT', 'PO', 'SI', 'SP', 'UN', 'WC'</td>
    </tr>
    <tr>
      <th>'medical_specialty'</th>
      <td>'AllergyandImmunology', 'Anesthesiology', 'Anesthesiology-Pediatric', 'Cardiology', 'Cardiology-Pediatric', 'Dentistry', 'Dermatology', 'Emergency/Trauma', 'Endocrinology', 'Family/GeneralPractice', 'Gastroenterology', 'Gynecology', 'Hematology', 'Hematology/Oncology', 'Hospitalist', 'InfectiousDiseases', 'InternalMedicine', 'Nephrology', 'Neurology', 'Obsterics&amp;Gynecology-GynecologicOnco', 'Obstetrics', 'ObstetricsandGynecology', 'Oncology', 'Ophthalmology', 'Orthopedics', 'Orthopedics-Reconstructive', 'Osteopath', 'Otolaryngology', 'OutreachServices', 'Pathology', 'Pediatrics', 'Pediatrics-AllergyandImmunology', 'Pediatrics-CriticalCare', 'Pediatrics-EmergencyMedicine', 'Pediatrics-Endocrinology', 'Pediatrics-Hematology-Oncology', 'Pediatrics-InfectiousDiseases', 'Pediatrics-Neurology', 'Pediatrics-Pulmonology', 'Perinatology', 'PhysicalMedicineandRehabilitation', 'PhysicianNotFound', 'Podiatry', 'Proctology', 'Psychiatry', 'Psychiatry-Addictive', 'Psychiatry-Child/Adolescent', 'Psychology', 'Pulmonology', 'Radiologist', 'Radiology', 'Rheumatology', 'Speech', 'SportsMedicine', 'Surgeon', 'Surgery-Cardiovascular', 'Surgery-Cardiovascular/Thoracic', 'Surgery-Colon&amp;Rectal', 'Surgery-General', 'Surgery-Maxillofacial', 'Surgery-Neuro', 'Surgery-Pediatric', 'Surgery-Plastic', 'Surgery-PlasticwithinHeadandNeck', 'Surgery-Thoracic', 'Surgery-Vascular', 'SurgicalSpecialty', 'Urology', 'Neurophysiology'</td>
    </tr>
    <tr>
      <th>'max_glu_serum'</th>
      <td>'&gt;200', '&gt;300', 'None', 'Norm'</td>
    </tr>
    <tr>
      <th>'A1Cresult'</th>
      <td>'&gt;7', '&gt;8', 'None', 'Norm'</td>
    </tr>
    <tr>
      <th>'metformin'</th>
      <td>'Down', 'No', 'Steady', 'Up'</td>
    </tr>
    <tr>
      <th>'repaglinide'</th>
      <td>'Down', 'No', 'Steady', 'Up'</td>
    </tr>
    <tr>
      <th>'nateglinide'</th>
      <td>'Down', 'No', 'Steady', 'Up'</td>
    </tr>
    <tr>
      <th>'chlorpropamide'</th>
      <td>'Down', 'No', 'Steady', 'Up'</td>
    </tr>
    <tr>
      <th>'glimepiride'</th>
      <td>'Down', 'No', 'Steady', 'Up'</td>
    </tr>
    <tr>
      <th>'acetohexamide'</th>
      <td>'No', 'Steady'</td>
    </tr>
    <tr>
      <th>'glipizide'</th>
      <td>'Down', 'No', 'Steady', 'Up'</td>
    </tr>
    <tr>
      <th>'glyburide'</th>
      <td>'Down', 'No', 'Steady', 'Up'</td>
    </tr>
    <tr>
      <th>'tolbutamide'</th>
      <td>'No', 'Steady'</td>
    </tr>
    <tr>
      <th>'pioglitazone'</th>
      <td>'Down', 'No', 'Steady', 'Up'</td>
    </tr>
    <tr>
      <th>'rosiglitazone'</th>
      <td>'Down', 'No', 'Steady', 'Up'</td>
    </tr>
    <tr>
      <th>'acarbose'</th>
      <td>'Down', 'No', 'Steady', 'Up'</td>
    </tr>
    <tr>
      <th>'miglitol'</th>
      <td>'Down', 'No', 'Steady', 'Up'</td>
    </tr>
    <tr>
      <th>'troglitazone'</th>
      <td>'No', 'Steady'</td>
    </tr>
    <tr>
      <th>'tolazamide'</th>
      <td>'No', 'Steady', 'Up'</td>
    </tr>
    <tr>
      <th>'examide'</th>
      <td>'No'</td>
    </tr>
    <tr>
      <th>'citoglipton'</th>
      <td>'No'</td>
    </tr>
    <tr>
      <th>'insulin'</th>
      <td>'Down', 'No', 'Steady', 'Up'</td>
    </tr>
    <tr>
      <th>'glyburide-metformin'</th>
      <td>'Down', 'No', 'Steady', 'Up'</td>
    </tr>
    <tr>
      <th>'glipizide-metformin'</th>
      <td>'No', 'Steady'</td>
    </tr>
    <tr>
      <th>'glimepiride-pioglitazone'</th>
      <td>'No', 'Steady'</td>
    </tr>
    <tr>
      <th>'metformin-rosiglitazone'</th>
      <td>'No'</td>
    </tr>
    <tr>
      <th>'metformin-pioglitazone'</th>
      <td>'No'</td>
    </tr>
    <tr>
      <th>'change'</th>
      <td>'Ch', 'No'</td>
    </tr>
    <tr>
      <th>'diabetesMed'</th>
      <td>'No', 'Yes'</td>
    </tr>
    <tr>
      <th>'readmitted'</th>
      <td>'&lt;30', '&gt;30', 'NO'</td>
    </tr>
  </tbody>
</table>
</div>


Towards the bottom of the Domain-Values pairs of the cell above, you can see that many features (including **'metformin'**) have the same values: `['Down', 'No', 'Steady', 'Up']`. These values are common to many features including the ones with missing values during schema inference. 

TFDV allows you to modify the domains of some features to match an existing domain. To address the detected anomaly, you can **set the domain** of these features to the domain of the `metformin` feature.

Complete the function below to set the domain of a feature list to an existing feature domain. 

For this, use the [`tfdv.set_domain()`](https://www.tensorflow.org/tfx/data_validation/api_docs/python/tfdv/set_domain) function, which has the following parameters:

- `schema`: The schema


- `feature_path`: The name of the feature whose domain needs to be set.


- `domain`: A domain protocol buffer or the name of a global string domain present in the input schema.


```python
# grader-required-cell

def modify_domain_of_features(features_list, schema, to_domain_name):
    '''
    Modify a list of features' domains.

            Parameters:
                    features_list : Features that need to be modified
                    schema: Inferred schema
                    to_domain_name : Target domain to be transferred to the features list

            Returns:
                    schema: new schema
    '''
    ### START CODE HERE
    # HINT: Loop over the feature list and use set_domain with the inferred schema, feature name and target domain name
    for feature in features_list:
        tfdv.set_domain(schema, feature, to_domain_name)
    
    ### END CODE HERE
    return schema
```

Using this function, set the domain of the features defined in the `domain_change_features` list below to be equal to **metformin's domain** to address the anomalies found.

**Since you are overriding the existing domain of the features, it is normal to get a warning so you don't do this by accident.**


```python
# grader-required-cell

domain_change_features = ['repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 
                          'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 
                          'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 
                          'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin', 
                          'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone']


# Infer new schema by using your modify_domain_of_features function 
# and the defined domain_change_features feature list
schema = modify_domain_of_features(domain_change_features, schema, 'metformin')

# Display new schema
tfdv.display_schema(schema)
```

    WARNING:root:Replacing existing domain of feature "repaglinide".
    WARNING:root:Replacing existing domain of feature "nateglinide".
    WARNING:root:Replacing existing domain of feature "chlorpropamide".
    WARNING:root:Replacing existing domain of feature "glimepiride".
    WARNING:root:Replacing existing domain of feature "acetohexamide".
    WARNING:root:Replacing existing domain of feature "glipizide".
    WARNING:root:Replacing existing domain of feature "glyburide".
    WARNING:root:Replacing existing domain of feature "tolbutamide".
    WARNING:root:Replacing existing domain of feature "pioglitazone".
    WARNING:root:Replacing existing domain of feature "rosiglitazone".
    WARNING:root:Replacing existing domain of feature "acarbose".
    WARNING:root:Replacing existing domain of feature "miglitol".
    WARNING:root:Replacing existing domain of feature "troglitazone".
    WARNING:root:Replacing existing domain of feature "tolazamide".
    WARNING:root:Replacing existing domain of feature "examide".
    WARNING:root:Replacing existing domain of feature "citoglipton".
    WARNING:root:Replacing existing domain of feature "insulin".
    WARNING:root:Replacing existing domain of feature "glyburide-metformin".
    WARNING:root:Replacing existing domain of feature "glipizide-metformin".
    WARNING:root:Replacing existing domain of feature "glimepiride-pioglitazone".
    WARNING:root:Replacing existing domain of feature "metformin-rosiglitazone".
    WARNING:root:Replacing existing domain of feature "metformin-pioglitazone".



<div>
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
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Type</th>
      <th>Presence</th>
      <th>Valency</th>
      <th>Domain</th>
    </tr>
    <tr>
      <th>Feature name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>'race'</th>
      <td>STRING</td>
      <td>optional</td>
      <td>single</td>
      <td>'race'</td>
    </tr>
    <tr>
      <th>'gender'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'gender'</td>
    </tr>
    <tr>
      <th>'age'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'age'</td>
    </tr>
    <tr>
      <th>'weight'</th>
      <td>STRING</td>
      <td>optional</td>
      <td>single</td>
      <td>'weight'</td>
    </tr>
    <tr>
      <th>'admission_type_id'</th>
      <td>INT</td>
      <td>required</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <th>'discharge_disposition_id'</th>
      <td>INT</td>
      <td>required</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <th>'admission_source_id'</th>
      <td>INT</td>
      <td>required</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <th>'time_in_hospital'</th>
      <td>INT</td>
      <td>required</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <th>'payer_code'</th>
      <td>STRING</td>
      <td>optional</td>
      <td>single</td>
      <td>'payer_code'</td>
    </tr>
    <tr>
      <th>'medical_specialty'</th>
      <td>STRING</td>
      <td>optional</td>
      <td>single</td>
      <td>'medical_specialty'</td>
    </tr>
    <tr>
      <th>'num_lab_procedures'</th>
      <td>INT</td>
      <td>required</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <th>'num_procedures'</th>
      <td>INT</td>
      <td>required</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <th>'num_medications'</th>
      <td>INT</td>
      <td>required</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <th>'number_outpatient'</th>
      <td>INT</td>
      <td>required</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <th>'number_emergency'</th>
      <td>INT</td>
      <td>required</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <th>'number_inpatient'</th>
      <td>INT</td>
      <td>required</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <th>'diag_1'</th>
      <td>BYTES</td>
      <td>optional</td>
      <td>single</td>
      <td>-</td>
    </tr>
    <tr>
      <th>'diag_2'</th>
      <td>BYTES</td>
      <td>optional</td>
      <td>single</td>
      <td>-</td>
    </tr>
    <tr>
      <th>'diag_3'</th>
      <td>BYTES</td>
      <td>optional</td>
      <td>single</td>
      <td>-</td>
    </tr>
    <tr>
      <th>'number_diagnoses'</th>
      <td>INT</td>
      <td>required</td>
      <td></td>
      <td>-</td>
    </tr>
    <tr>
      <th>'max_glu_serum'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'max_glu_serum'</td>
    </tr>
    <tr>
      <th>'A1Cresult'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'A1Cresult'</td>
    </tr>
    <tr>
      <th>'metformin'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'metformin'</td>
    </tr>
    <tr>
      <th>'repaglinide'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'metformin'</td>
    </tr>
    <tr>
      <th>'nateglinide'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'metformin'</td>
    </tr>
    <tr>
      <th>'chlorpropamide'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'metformin'</td>
    </tr>
    <tr>
      <th>'glimepiride'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'metformin'</td>
    </tr>
    <tr>
      <th>'acetohexamide'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'metformin'</td>
    </tr>
    <tr>
      <th>'glipizide'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'metformin'</td>
    </tr>
    <tr>
      <th>'glyburide'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'metformin'</td>
    </tr>
    <tr>
      <th>'tolbutamide'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'metformin'</td>
    </tr>
    <tr>
      <th>'pioglitazone'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'metformin'</td>
    </tr>
    <tr>
      <th>'rosiglitazone'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'metformin'</td>
    </tr>
    <tr>
      <th>'acarbose'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'metformin'</td>
    </tr>
    <tr>
      <th>'miglitol'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'metformin'</td>
    </tr>
    <tr>
      <th>'troglitazone'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'metformin'</td>
    </tr>
    <tr>
      <th>'tolazamide'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'metformin'</td>
    </tr>
    <tr>
      <th>'examide'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'metformin'</td>
    </tr>
    <tr>
      <th>'citoglipton'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'metformin'</td>
    </tr>
    <tr>
      <th>'insulin'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'metformin'</td>
    </tr>
    <tr>
      <th>'glyburide-metformin'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'metformin'</td>
    </tr>
    <tr>
      <th>'glipizide-metformin'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'metformin'</td>
    </tr>
    <tr>
      <th>'glimepiride-pioglitazone'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'metformin'</td>
    </tr>
    <tr>
      <th>'metformin-rosiglitazone'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'metformin'</td>
    </tr>
    <tr>
      <th>'metformin-pioglitazone'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'metformin'</td>
    </tr>
    <tr>
      <th>'change'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'change'</td>
    </tr>
    <tr>
      <th>'diabetesMed'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'diabetesMed'</td>
    </tr>
    <tr>
      <th>'readmitted'</th>
      <td>STRING</td>
      <td>required</td>
      <td></td>
      <td>'readmitted'</td>
    </tr>
  </tbody>
</table>
</div>



<div>
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
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Values</th>
    </tr>
    <tr>
      <th>Domain</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>'race'</th>
      <td>'AfricanAmerican', 'Asian', 'Caucasian', 'Hispanic', 'Other'</td>
    </tr>
    <tr>
      <th>'gender'</th>
      <td>'Female', 'Male', 'Unknown/Invalid'</td>
    </tr>
    <tr>
      <th>'age'</th>
      <td>'[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'</td>
    </tr>
    <tr>
      <th>'weight'</th>
      <td>'&gt;200', '[0-25)', '[100-125)', '[125-150)', '[150-175)', '[175-200)', '[25-50)', '[50-75)', '[75-100)'</td>
    </tr>
    <tr>
      <th>'payer_code'</th>
      <td>'BC', 'CH', 'CM', 'CP', 'DM', 'HM', 'MC', 'MD', 'MP', 'OG', 'OT', 'PO', 'SI', 'SP', 'UN', 'WC'</td>
    </tr>
    <tr>
      <th>'medical_specialty'</th>
      <td>'AllergyandImmunology', 'Anesthesiology', 'Anesthesiology-Pediatric', 'Cardiology', 'Cardiology-Pediatric', 'Dentistry', 'Dermatology', 'Emergency/Trauma', 'Endocrinology', 'Family/GeneralPractice', 'Gastroenterology', 'Gynecology', 'Hematology', 'Hematology/Oncology', 'Hospitalist', 'InfectiousDiseases', 'InternalMedicine', 'Nephrology', 'Neurology', 'Obsterics&amp;Gynecology-GynecologicOnco', 'Obstetrics', 'ObstetricsandGynecology', 'Oncology', 'Ophthalmology', 'Orthopedics', 'Orthopedics-Reconstructive', 'Osteopath', 'Otolaryngology', 'OutreachServices', 'Pathology', 'Pediatrics', 'Pediatrics-AllergyandImmunology', 'Pediatrics-CriticalCare', 'Pediatrics-EmergencyMedicine', 'Pediatrics-Endocrinology', 'Pediatrics-Hematology-Oncology', 'Pediatrics-InfectiousDiseases', 'Pediatrics-Neurology', 'Pediatrics-Pulmonology', 'Perinatology', 'PhysicalMedicineandRehabilitation', 'PhysicianNotFound', 'Podiatry', 'Proctology', 'Psychiatry', 'Psychiatry-Addictive', 'Psychiatry-Child/Adolescent', 'Psychology', 'Pulmonology', 'Radiologist', 'Radiology', 'Rheumatology', 'Speech', 'SportsMedicine', 'Surgeon', 'Surgery-Cardiovascular', 'Surgery-Cardiovascular/Thoracic', 'Surgery-Colon&amp;Rectal', 'Surgery-General', 'Surgery-Maxillofacial', 'Surgery-Neuro', 'Surgery-Pediatric', 'Surgery-Plastic', 'Surgery-PlasticwithinHeadandNeck', 'Surgery-Thoracic', 'Surgery-Vascular', 'SurgicalSpecialty', 'Urology', 'Neurophysiology'</td>
    </tr>
    <tr>
      <th>'max_glu_serum'</th>
      <td>'&gt;200', '&gt;300', 'None', 'Norm'</td>
    </tr>
    <tr>
      <th>'A1Cresult'</th>
      <td>'&gt;7', '&gt;8', 'None', 'Norm'</td>
    </tr>
    <tr>
      <th>'metformin'</th>
      <td>'Down', 'No', 'Steady', 'Up'</td>
    </tr>
    <tr>
      <th>'repaglinide'</th>
      <td>'Down', 'No', 'Steady', 'Up'</td>
    </tr>
    <tr>
      <th>'nateglinide'</th>
      <td>'Down', 'No', 'Steady', 'Up'</td>
    </tr>
    <tr>
      <th>'chlorpropamide'</th>
      <td>'Down', 'No', 'Steady', 'Up'</td>
    </tr>
    <tr>
      <th>'glimepiride'</th>
      <td>'Down', 'No', 'Steady', 'Up'</td>
    </tr>
    <tr>
      <th>'acetohexamide'</th>
      <td>'No', 'Steady'</td>
    </tr>
    <tr>
      <th>'glipizide'</th>
      <td>'Down', 'No', 'Steady', 'Up'</td>
    </tr>
    <tr>
      <th>'glyburide'</th>
      <td>'Down', 'No', 'Steady', 'Up'</td>
    </tr>
    <tr>
      <th>'tolbutamide'</th>
      <td>'No', 'Steady'</td>
    </tr>
    <tr>
      <th>'pioglitazone'</th>
      <td>'Down', 'No', 'Steady', 'Up'</td>
    </tr>
    <tr>
      <th>'rosiglitazone'</th>
      <td>'Down', 'No', 'Steady', 'Up'</td>
    </tr>
    <tr>
      <th>'acarbose'</th>
      <td>'Down', 'No', 'Steady', 'Up'</td>
    </tr>
    <tr>
      <th>'miglitol'</th>
      <td>'Down', 'No', 'Steady', 'Up'</td>
    </tr>
    <tr>
      <th>'troglitazone'</th>
      <td>'No', 'Steady'</td>
    </tr>
    <tr>
      <th>'tolazamide'</th>
      <td>'No', 'Steady', 'Up'</td>
    </tr>
    <tr>
      <th>'examide'</th>
      <td>'No'</td>
    </tr>
    <tr>
      <th>'citoglipton'</th>
      <td>'No'</td>
    </tr>
    <tr>
      <th>'insulin'</th>
      <td>'Down', 'No', 'Steady', 'Up'</td>
    </tr>
    <tr>
      <th>'glyburide-metformin'</th>
      <td>'Down', 'No', 'Steady', 'Up'</td>
    </tr>
    <tr>
      <th>'glipizide-metformin'</th>
      <td>'No', 'Steady'</td>
    </tr>
    <tr>
      <th>'glimepiride-pioglitazone'</th>
      <td>'No', 'Steady'</td>
    </tr>
    <tr>
      <th>'metformin-rosiglitazone'</th>
      <td>'No'</td>
    </tr>
    <tr>
      <th>'metformin-pioglitazone'</th>
      <td>'No'</td>
    </tr>
    <tr>
      <th>'change'</th>
      <td>'Ch', 'No'</td>
    </tr>
    <tr>
      <th>'diabetesMed'</th>
      <td>'No', 'Yes'</td>
    </tr>
    <tr>
      <th>'readmitted'</th>
      <td>'&lt;30', '&gt;30', 'NO'</td>
    </tr>
  </tbody>
</table>
</div>



```python
# grader-required-cell

# TEST CODE

# check that the domain of some features are now switched to `metformin`
print(f"Domain name of 'chlorpropamide': {tfdv.get_feature(schema, 'chlorpropamide').domain}")
print(f"Domain values of 'chlorpropamide': {tfdv.get_domain(schema, 'chlorpropamide').value}")
print(f"Domain name of 'repaglinide': {tfdv.get_feature(schema, 'repaglinide').domain}")
print(f"Domain values of 'repaglinide': {tfdv.get_domain(schema, 'repaglinide').value}")
print(f"Domain name of 'nateglinide': {tfdv.get_feature(schema, 'nateglinide').domain}")
print(f"Domain values of 'nateglinide': {tfdv.get_domain(schema, 'nateglinide').value}")
```

    Domain name of 'chlorpropamide': metformin
    Domain values of 'chlorpropamide': ['Down', 'No', 'Steady', 'Up']
    Domain name of 'repaglinide': metformin
    Domain values of 'repaglinide': ['Down', 'No', 'Steady', 'Up']
    Domain name of 'nateglinide': metformin
    Domain values of 'nateglinide': ['Down', 'No', 'Steady', 'Up']


**Expected Output:**

```
Domain name of 'chlorpropamide': metformin
Domain values of 'chlorpropamide': ['Down', 'No', 'Steady', 'Up']
Domain name of 'repaglinide': metformin
Domain values of 'repaglinide': ['Down', 'No', 'Steady', 'Up']
Domain name of 'nateglinide': metformin
Domain values of 'nateglinide': ['Down', 'No', 'Steady', 'Up']
```

Let's do a final check of anomalies to see if this solved the issue.


```python
# grader-required-cell

calculate_and_display_anomalies(serving_stats, schema=schema)
```


<div>
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
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Anomaly short description</th>
      <th>Anomaly long description</th>
    </tr>
    <tr>
      <th>Feature name</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>'readmitted'</th>
      <td>Column dropped</td>
      <td>Column is completely missing</td>
    </tr>
  </tbody>
</table>
</div>


You should now see the `metformin-pioglitazone` and `metformin-rosiglitazone` features dropped from the output anomalies.

<a name='ex-9'></a>
### Exercise 9: Detecting anomalies with environments

There is still one thing to address. The `readmitted` feature (which is the label column) showed up as an anomaly ('Column dropped'). Since labels are not expected in the serving data, let's tell TFDV to ignore this detected anomaly.

This requirement of introducing slight schema variations can be expressed by using [environments](https://www.tensorflow.org/tfx/data_validation/get_started#schema_environments). In particular, features in the schema can be associated with a set of environments using `default_environment`, `in_environment` and `not_in_environment`.


```python
# grader-required-cell

# All features are by default in both TRAINING and SERVING environments.
schema.default_environment.append('TRAINING')
schema.default_environment.append('SERVING')
```

Complete the code below to exclude the `readmitted` feature from the `SERVING` environment.

To achieve this, you can use the [`tfdv.get_feature()`](https://www.tensorflow.org/tfx/data_validation/api_docs/python/tfdv/get_feature) function to get the `readmitted` feature from the inferred schema and use its `not_in_environment` attribute to specify that `readmitted` should be removed from the `SERVING` environment's schema. This **attribute is a list** so you will have to **append** the name of the environment that you wish to omit this feature for.

To be more explicit, given a feature you can do something like:

```python
feature.not_in_environment.append('NAME_OF_ENVIRONMENT')
```

The function `tfdv.get_feature` receives the following parameters:

- `schema`: The schema.
- `feature_path`: The path of the feature to obtain from the schema. In this case this is equal to the name of the feature.


```python
# grader-required-cell

### START CODE HERE
# Specify that 'readmitted' feature is not in SERVING environment.
# HINT: Append the 'SERVING' environmnet to the not_in_environment attribute of the feature
tfdv.get_feature(schema, 'readmitted').not_in_environment.append('SERVING')

# HINT: Calculate anomalies with the validate_statistics function by using the serving statistics, 
# inferred schema and the SERVING environment parameter.
serving_anomalies_with_env = tfdv.validate_statistics(serving_stats, schema, environment='SERVING')
### END CODE HERE
```

You should see "No anomalies found" by running the cell below.


```python
# grader-required-cell

# Display anomalies
tfdv.display_anomalies(serving_anomalies_with_env)
```


<h4 style="color:green;">No anomalies found.</h4>


Now you have succesfully addressed all anomaly-related issues!

<a name='7'></a>
## 7 - Check for Data Drift and Skew

During data validation, you also need to check for data drift and data skew between the training and serving data. You can do this by specifying the [skew_comparator and drift_comparator](https://www.tensorflow.org/tfx/data_validation/get_started#checking_data_skew_and_drift) in the schema. 

Drift and skew is expressed in terms of [L-infinity distance](https://en.wikipedia.org/wiki/Chebyshev_distance) which evaluates the difference between vectors as the greatest of the differences along any coordinate dimension.

You can set the threshold distance so that you receive warnings when the drift is higher than is acceptable.  Setting the correct distance is typically an iterative process requiring domain knowledge and experimentation.

Let's check for the skew in the **diabetesMed** feature and drift in the **payer_code** feature.


```python
# grader-required-cell

# Calculate skew for the diabetesMed feature
diabetes_med = tfdv.get_feature(schema, 'diabetesMed')
diabetes_med.skew_comparator.infinity_norm.threshold = 0.03 # domain knowledge helps to determine this threshold

# Calculate drift for the payer_code feature
payer_code = tfdv.get_feature(schema, 'payer_code')
payer_code.drift_comparator.infinity_norm.threshold = 0.03 # domain knowledge helps to determine this threshold

# Calculate anomalies
skew_drift_anomalies = tfdv.validate_statistics(train_stats, schema,
                                          previous_statistics=eval_stats,
                                          serving_statistics=serving_stats)

# Display anomalies
tfdv.display_anomalies(skew_drift_anomalies)
```


<div>
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
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Anomaly short description</th>
      <th>Anomaly long description</th>
    </tr>
    <tr>
      <th>Feature name</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>'diabetesMed'</th>
      <td>High Linfty distance between training and serving</td>
      <td>The Linfty distance between training and serving is 0.0325464 (up to six significant digits), above the threshold 0.03. The feature value with maximum difference is: No</td>
    </tr>
    <tr>
      <th>'payer_code'</th>
      <td>High Linfty distance between current and previous</td>
      <td>The Linfty distance between current and previous is 0.0342144 (up to six significant digits), above the threshold 0.03. The feature value with maximum difference is: MC</td>
    </tr>
  </tbody>
</table>
</div>


In both of these cases, the detected anomaly distance is not too far from the threshold value of `0.03`. For this exercise, let's accept this as within bounds (i.e. you can set the distance to something like `0.035` instead).

**However, if the anomaly truly indicates a skew and drift, then further investigation is necessary as this could have a direct impact on model performance.**

<a name='8'></a>
## 8 - Display Stats for Data Slices <a class="anchor" id="fourth-objective"></a>

Finally, you can [slice the dataset and calculate the statistics](https://www.tensorflow.org/tfx/data_validation/get_started#computing_statistics_over_slices_of_data) for each unique value of a feature. By default, TFDV computes statistics for the overall dataset in addition to the configured slices. Each slice is identified by a unique name which is set as the dataset name in the [DatasetFeatureStatistics](https://github.com/tensorflow/metadata/blob/master/tensorflow_metadata/proto/v0/statistics.proto#L43) protocol buffer. Generating and displaying statistics over different slices of data can help track model and anomaly metrics. 

Let's first define a few helper functions to make our code in the exercise more neat.


```python
# grader-required-cell

def split_datasets(dataset_list):
    '''
    split datasets.

            Parameters:
                    dataset_list: List of datasets to split

            Returns:
                    datasets: sliced data
    '''
    datasets = []
    for dataset in dataset_list.datasets:
        proto_list = DatasetFeatureStatisticsList()
        proto_list.datasets.extend([dataset])
        datasets.append(proto_list)
    return datasets


def display_stats_at_index(index, datasets):
    '''
    display statistics at the specified data index

            Parameters:
                    index : index to show the anomalies
                    datasets: split data

            Returns:
                    display of generated sliced data statistics at the specified index
    '''
    if index < len(datasets):
        print(datasets[index].datasets[0].name)
        tfdv.visualize_statistics(datasets[index])
```

The function below returns a list of `DatasetFeatureStatisticsList` protocol buffers. As shown in the ungraded lab, the first one will be for `All Examples` followed by individual slices through the feature you specified.

To configure TFDV to generate statistics for dataset slices, you will use the function `tfdv.StatsOptions()` with the following 4 arguments: 

- `schema`


- `slice_functions` passed as a list.


- `infer_type_from_schema` set to True. 


- `feature_allowlist` set to the approved features.


Remember that `slice_functions` only work with [`generate_statistics_from_csv()`](https://www.tensorflow.org/tfx/data_validation/api_docs/python/tfdv/generate_statistics_from_csv) so you will need to convert the dataframe to CSV.


```python
# grader-required-cell

def sliced_stats_for_slice_fn(slice_fn, approved_cols, dataframe, schema):
    '''
    generate statistics for the sliced data.

            Parameters:
                    slice_fn : slicing definition
                    approved_cols: list of features to pass to the statistics options
                    dataframe: pandas dataframe to slice
                    schema: the schema

            Returns:
                    slice_info_datasets: statistics for the sliced dataset
    '''
    # Set the StatsOptions
    slice_stats_options = tfdv.StatsOptions(schema=schema,
                                            slice_functions=[slice_fn],
                                            infer_type_from_schema=True,
                                            feature_allowlist=approved_cols)
    
    # Convert Dataframe to CSV since `slice_functions` works only with `tfdv.generate_statistics_from_csv`
    CSV_PATH = 'slice_sample.csv'
    dataframe.to_csv(CSV_PATH)
    
    # Calculate statistics for the sliced dataset
    sliced_stats = tfdv.generate_statistics_from_csv(CSV_PATH, stats_options=slice_stats_options)
    
    # Split the dataset using the previously defined split_datasets function
    slice_info_datasets = split_datasets(sliced_stats)
    
    return slice_info_datasets
```

With that, you can now use the helper functions to generate and visualize statistics for the sliced datasets.


```python
# grader-required-cell

# Generate slice function for the `medical_speciality` feature
slice_fn = slicing_util.get_feature_value_slicer(features={'medical_specialty': None})

# Generate stats for the sliced dataset
slice_datasets = sliced_stats_for_slice_fn(slice_fn, approved_cols, dataframe=train_df, schema=schema)

# Print name of slices for reference
print(f'Statistics generated for:\n')
print('\n'.join([sliced.datasets[0].name for sliced in slice_datasets]))

# Display at index 10, which corresponds to the slice named `medical_specialty_Gastroenterology`
display_stats_at_index(10, slice_datasets) 
```



    WARNING:root:Make sure that locally built Python SDK docker image has Python 3.8 interpreter.


    Statistics generated for:
    
    All Examples
    medical_specialty_Pediatrics-Endocrinology
    medical_specialty_InternalMedicine
    medical_specialty_Family/GeneralPractice
    medical_specialty_Cardiology
    medical_specialty_Surgery-General
    medical_specialty_Orthopedics
    medical_specialty_Gastroenterology
    medical_specialty_Surgery-Cardiovascular/Thoracic
    medical_specialty_Nephrology
    medical_specialty_Orthopedics-Reconstructive
    medical_specialty_Psychiatry
    medical_specialty_Emergency/Trauma
    medical_specialty_Pulmonology
    medical_specialty_Surgery-Neuro
    medical_specialty_Obsterics&Gynecology-GynecologicOnco
    medical_specialty_ObstetricsandGynecology
    medical_specialty_Pediatrics
    medical_specialty_Hematology/Oncology
    medical_specialty_Otolaryngology
    medical_specialty_Surgery-Colon&Rectal
    medical_specialty_Pediatrics-CriticalCare
    medical_specialty_Endocrinology
    medical_specialty_Urology
    medical_specialty_Psychiatry-Child/Adolescent
    medical_specialty_Pediatrics-Pulmonology
    medical_specialty_Neurology
    medical_specialty_Anesthesiology-Pediatric
    medical_specialty_Radiology
    medical_specialty_Pediatrics-Hematology-Oncology
    medical_specialty_Psychology
    medical_specialty_Podiatry
    medical_specialty_Gynecology
    medical_specialty_Oncology
    medical_specialty_Pediatrics-Neurology
    medical_specialty_Surgery-Plastic
    medical_specialty_Surgery-Thoracic
    medical_specialty_Surgery-PlasticwithinHeadandNeck
    medical_specialty_Ophthalmology
    medical_specialty_Surgery-Pediatric
    medical_specialty_Pediatrics-EmergencyMedicine
    medical_specialty_PhysicalMedicineandRehabilitation
    medical_specialty_InfectiousDiseases
    medical_specialty_Anesthesiology
    medical_specialty_Rheumatology
    medical_specialty_AllergyandImmunology
    medical_specialty_Surgery-Maxillofacial
    medical_specialty_Pediatrics-InfectiousDiseases
    medical_specialty_Pediatrics-AllergyandImmunology
    medical_specialty_Dentistry
    medical_specialty_Surgeon
    medical_specialty_Surgery-Vascular
    medical_specialty_Osteopath
    medical_specialty_Psychiatry-Addictive
    medical_specialty_Surgery-Cardiovascular
    medical_specialty_PhysicianNotFound
    medical_specialty_Hematology
    medical_specialty_Proctology
    medical_specialty_Obstetrics
    medical_specialty_SurgicalSpecialty
    medical_specialty_Radiologist
    medical_specialty_Pathology
    medical_specialty_Dermatology
    medical_specialty_SportsMedicine
    medical_specialty_Speech
    medical_specialty_Hospitalist
    medical_specialty_OutreachServices
    medical_specialty_Cardiology-Pediatric
    medical_specialty_Perinatology
    medical_specialty_Orthopedics-Reconstructive



<iframe id='facets-iframe' width="100%" height="500px"></iframe>
        <script>
        facets_iframe = document.getElementById('facets-iframe');
        facets_html = '<script src="https://cdnjs.cloudflare.com/ajax/libs/webcomponentsjs/1.3.3/webcomponents-lite.js"><\/script><link rel="import" href="https://raw.githubusercontent.com/PAIR-code/facets/master/facets-dist/facets-jupyter.html"><facets-overview proto-input="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"></facets-overview>';
        facets_iframe.srcdoc = facets_html;
         facets_iframe.id = "";
         setTimeout(() => {
           facets_iframe.setAttribute('height', facets_iframe.contentWindow.document.body.offsetHeight + 'px')
         }, 1500)
         </script>


If you are curious, try different slice indices to extract the group statistics. For instance, `index=5` corresponds to all `medical_specialty_Surgery-General` records. You can also try slicing through multiple features as shown in the ungraded lab. 

Another challenge is to implement your own helper functions. For instance, you can make a `display_stats_for_slice_name()` function so you don't have to determine the index of a slice. If done correctly, you can just do `display_stats_for_slice_name('medical_specialty_Gastroenterology', slice_datasets)` and it will generate the same result as `display_stats_at_index(10, slice_datasets)`.

<a name='9'></a>
## 9 - Freeze the schema

Now that the schema has been reviewed, you will store the schema in a file in its "frozen" state. This can be used to validate incoming data once your application goes live to your users.

This is pretty straightforward using Tensorflow's `io` utils and TFDV's [`write_schema_text()`](https://www.tensorflow.org/tfx/data_validation/api_docs/python/tfdv/write_schema_text) function.


```python
# grader-required-cell

# Create output directory
OUTPUT_DIR = "output"
file_io.recursive_create_dir(OUTPUT_DIR)

# Use TensorFlow text output format pbtxt to store the schema
schema_file = os.path.join(OUTPUT_DIR, 'schema.pbtxt')

# write_schema_text function expect the defined schema and output path as parameters
tfdv.write_schema_text(schema, schema_file) 
```

After submitting this assignment, you can click the Jupyter logo in the left upper corner of the screen to check the Jupyter filesystem. The `schema.pbtxt` file should be inside the `output` directory. 

**Congratulations on finishing this week's assignment!** A lot of concepts where introduced and now you should feel more familiar with using TFDV for inferring schemas, anomaly detection and other data-related tasks.

**Keep it up!**

<details>
  <summary><font size="2" color="darkgreen"><b>Please click here if you want to experiment with any of the non-graded code.</b></font></summary>
    <p><i><b>Important Note: Please only do this when you've already passed the assignment to avoid problems with the autograder.</b></i>
    <ol>
        <li> On the notebook’s menu, click “View” > “Cell Toolbar” > “Edit Metadata”</li>
        <li> Hit the “Edit Metadata” button next to the code cell which you want to lock/unlock</li>
        <li> Set the attribute value for “editable” to:
            <ul>
                <li> “true” if you want to unlock it </li>
                <li> “false” if you want to lock it </li>
            </ul>
        </li>
        <li> On the notebook’s menu, click “View” > “Cell Toolbar” > “None” </li>
    </ol>
    <p> Here's a short demo of how to do the steps above: 
        <br>
        <img src="https://drive.google.com/uc?export=view&id=14Xy_Mb17CZVgzVAgq7NCjMVBvSae3xO1" align="center">
</details>
