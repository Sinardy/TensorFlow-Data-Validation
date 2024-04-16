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

<a name='1'></a>
## 1 - Setup and Imports

```python
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

<a name='2-1'></a>
### 2.1 Read and Split the Dataset

```python
# Read CSV data into a dataframe and recognize the missing data that is encoded with '?' string as NaN
df = pd.read_csv('data/diabetic_data.csv', header=0, na_values = '?')

# Preview the dataset
df.head()
```

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
# Split the datasets
train_df, eval_df, serving_df = prepare_data_splits_from_dataframe(df)
print('Training dataset has {} records\nValidation dataset has {} records\nServing dataset has {} records'.format(len(train_df),len(eval_df),len(serving_df)))
```

**Note:** Depending on your use case, you might need to shuffle the dataset first before calling the function above. For example, some datasets are arranged by geographical location so directly splitting them might result in training sets that only come from locations that are not found in the evaluation and serving sets. That will likely result in a model that will only do well on the locations found in the training set. One way to avoid that kind of problem is to shuffle the dataset. For instance, you can do something like this:

```python
# Shuffle the dataset
shuffled_df = df.sample(frac=1, random_state=48)

# Split the dataset
train_df, eval_df, serving_df = prepare_data_splits_from_dataframe(shuffled_df)
```


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

<a name='ex-1'></a>
### Exercise 1: Generate Training Statistics 

TFDV allows you to generate statistics from different data formats such as CSV or a Pandas DataFrame. 

Since you already have the data stored in a DataFrame you can use the function [`tfdv.generate_statistics_from_dataframe()`](https://www.tensorflow.org/tfx/data_validation/api_docs/python/tfdv/generate_statistics_from_dataframe) which, given a DataFrame and `stats_options`, generates an object of type `DatasetFeatureStatisticsList`. This object includes the computed statistics of the given dataset.

Complete the cell below to generate the statistics of the training set. Remember to pass the training dataframe and the `stats_options` that you defined above as arguments.


```python
train_stats = tfdv.generate_statistics_from_dataframe(train_df, stats_options=stats_options)
```


```python
# get the number of features used to compute statistics
print(f"Number of features used: {len(train_stats.datasets[0].features)}")

# check the number of examples used
print(f"Number of examples used: {train_stats.datasets[0].num_examples}")

# check the column names of the first and last feature
print(f"First feature: {train_stats.datasets[0].features[0].path.step[0]}")
print(f"Last feature: {train_stats.datasets[0].features[-1].path.step[0]}")
```

<a name='ex-2'></a>
### Exercise 2: Visualize Training Statistics

Now that you have the computed statistics in the `DatasetFeatureStatisticsList` instance, you will need a way to **visualize** these to get actual insights. TFDV provides this functionality through the method [`tfdv.visualize_statistics()`](https://www.tensorflow.org/tfx/data_validation/api_docs/python/tfdv/visualize_statistics).

Using this function in an interactive Python environment such as this one will output a very nice and convenient way to interact with the descriptive statistics you generated earlier. 

**Try it out yourself!** Remember to pass in the generated training statistics in the previous exercise as an argument.

```python
tfdv.visualize_statistics(train_stats)
```

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
# Infer the data schema by using the training statistics that you generated
schema = tfdv.infer_schema(statistics=train_stats)

# Display the data schema
tfdv.display_schema(schema)
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
```



<a name='ex-5'></a>
### Exercise 5: Detecting Anomalies ###

At this point, you should ask if your evaluation dataset matches the schema from your training dataset. For instance, if you scroll through the output cell in the previous exercise, you can see that the categorical feature **glimepiride-pioglitazone** has 1 unique value in the training set while the evaluation dataset has 2. You can verify with the built-in Pandas `describe()` method as well.


```python
train_df["glimepiride-pioglitazone"].describe()
eval_df["glimepiride-pioglitazone"].describe()
```

It is possible but highly inefficient to visually inspect and determine all the anomalies. So, let's instead use TFDV functions to detect and display these.

You can use the function [`tfdv.validate_statistics()`](https://www.tensorflow.org/tfx/data_validation/api_docs/python/tfdv/validate_statistics) for detecting anomalies and [`tfdv.display_anomalies()`](https://www.tensorflow.org/tfx/data_validation/api_docs/python/tfdv/display_anomalies) for displaying them.

The `validate_statistics()` method has two required arguments:
- an instance of `DatasetFeatureStatisticsList`
- an instance of `Schema`

Fill in the following graded function which, given the statistics and schema, displays the anomalies found.


```python
def calculate_and_display_anomalies(statistics, schema):
    '''
    Calculate and display anomalies.

            Parameters:
                    statistics : Data statistics in statistics_pb2.DatasetFeatureStatisticsList format
                    schema : Data schema in schema_pb2.Schema format

            Returns:
                    display of calculated anomalies
    '''
    # HINTS: Pass the statistics and schema parameters into the validation function 
    anomalies = tfdv.validate_statistics(statistics=statistics, schema=schema)
    
    # HINTS: Display input anomalies by using the calculated anomalies
    tfdv.display_anomalies(anomalies)
```

You should see detected anomalies in the `medical_specialty` and `glimepiride-pioglitazone` features by running the cell below.


```python
# Check evaluation data for errors by validating the evaluation data staticss using the previously inferred schema
calculate_and_display_anomalies(eval_stats, schema=schema)
```

<a name='ex-6'></a>
### Exercise 6: Fix evaluation anomalies in the schema

The evaluation data has records with values for the features **glimepiride-pioglitazone** and **medical_speciality**  that were not included in the schema generated from the training data. You can fix this by adding the new values that exist in the evaluation dataset to the domain of these features.

To get the `domain` of a particular feature you can use [`tfdv.get_domain()`](https://www.tensorflow.org/tfx/data_validation/api_docs/python/tfdv/get_domain).

You can use the `append()` method to the `value` property of the returned `domain` to add strings to the valid list of values. To be more explicit, given a domain you can do something like:


```python
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
```


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
# Define a new statistics options by the tfdv.StatsOptions class for the serving data by passing the previously inferred schema
options = tfdv.StatsOptions(schema=schema, 
                            infer_type_from_schema=True, 
                            feature_allowlist=approved_cols)
```


```python
# Generate serving dataset statistics
# HINT: Remember to use the serving dataframe and to pass the newly defined statistics options
serving_stats = tfdv.generate_statistics_from_dataframe(serving_df, stats_options=options)

# HINT: Calculate and display anomalies using the generated serving statistics
calculate_and_display_anomalies(serving_stats, schema=schema)
```

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
        facets_html = '<script src="https://cdnjs.cloudflare.com/ajax/libs/webcomponentsjs/1.3.3/webcomponents-lite.js"><\/script><link rel="import" href="https://raw.githubusercontent.com/PAIR-code/facets/master/facets-dist/facets-jupyter.html"><facets-overview proto-input="CpaYAgosbWVkaWNhbF9zcGVjaWFsdHlfT3J0aG9wZWRpY3MtUmVjb25zdHJ1Y3RpdmUQkwkatwQQAiKqBAq4AgjoCBArGAEgAS0AAIA/MqQCGhsJAAAAAAAA8D8RAAAAAAAA8D8hMzMzMzMzXEAaGwkAAAAAAADwPxEAAAAAAADwPyEzMzMzMzNcQBobCQAAAAAAAPA/EQAAAAAAAPA/ITMzMzMzM1xAGhsJAAAAAAAA8D8RAAAAAAAA8D8hMzMzMzMzXEAaGwkAAAAAAADwPxEAAAAAAADwPyEzMzMzMzNcQBobCQAAAAAAAPA/EQAAAAAAAPA/ITMzMzMzM1xAGhsJAAAAAAAA8D8RAAAAAAAA8D8hMzMzMzMzXEAaGwkAAAAAAADwPxEAAAAAAADwPyEzMzMzMzNcQBobCQAAAAAAAPA/EQAAAAAAAPA/ITMzMzMzM1xAGhsJAAAAAAAA8D8RAAAAAAAA8D8hMzMzMzMzXEAgAUDoCBAFGhQSCUNhdWNhc2lhbhkAAAAAALCMQBoaEg9BZnJpY2FuQW1lcmljYW4ZAAAAAADgZkAaExIISGlzcGFuaWMZAAAAAAAAMEAaEBIFT3RoZXIZAAAAAAAAIEAaEBIFQXNpYW4ZAAAAAAAACEAlMbkeQSp7ChQiCUNhdWNhc2lhbikAAAAAALCMQAoeCAEQASIPQWZyaWNhbkFtZXJpY2FuKQAAAAAA4GZAChcIAhACIghIaXNwYW5pYykAAAAAAAAwQAoUCAMQAyIFT3RoZXIpAAAAAAAAIEAKFAgEEAQiBUFzaWFuKQAAAAAAAAhAQgYKBHJhY2UanQMQAiKOAwq2AgiTCRgBIAEtAACAPzKkAhobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AIAFAkwkQAhoREgZGZW1hbGUZAAAAAADghUAaDxIETWFsZRkAAAAAAHB9QCUGQqZAKigKESIGRmVtYWxlKQAAAAAA4IVAChMIARABIgRNYWxlKQAAAAAAcH1AQggKBmdlbmRlchrZBRACIs0FCrYCCJMJGAEgAS0AAIA/MqQCGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAgAUCTCRAJGhISB1s3MC04MCkZAAAAAACAdEAaEhIHWzYwLTcwKRkAAAAAALByQBoSEgdbNTAtNjApGQAAAAAAoGtAGhISB1s4MC05MCkZAAAAAACgZEAaEhIHWzQwLTUwKRkAAAAAAABaQBoTEghbOTAtMTAwKRkAAAAAAAA5QBoSEgdbMzAtNDApGQAAAAAAADdAGhISB1syMC0zMCkZAAAAAAAAFEAaEhIHWzEwLTIwKRkAAAAAAADwPyXlruBAKtUBChIiB1s3MC04MCkpAAAAAACAdEAKFggBEAEiB1s2MC03MCkpAAAAAACwckAKFggCEAIiB1s1MC02MCkpAAAAAACga0AKFggDEAMiB1s4MC05MCkpAAAAAACgZEAKFggEEAQiB1s0MC01MCkpAAAAAAAAWkAKFwgFEAUiCFs5MC0xMDApKQAAAAAAADlAChYIBhAGIgdbMzAtNDApKQAAAAAAADdAChYIBxAHIgdbMjAtMzApKQAAAAAAABRAChYICBAIIgdbMTAtMjApKQAAAAAAAPA/QgUKA2FnZRoTEAIiBQoDEJMJQggKBndlaWdodBrKBxqyBwq2AgiTCRgBIAEtAACAPzKkAhobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AIAFAkwkRvaohwlguDEAZoU4n0GZV/D8pAAAAAAAA8D8xAAAAAAAACEA5AAAAAAAAIEBCogIaGwkAAAAAAADwPxEzMzMzMzP7PyEjbHh6pdZiQBobCTMzMzMzM/s/ETMzMzMzMwNAIRvAWyBB42RAGhsJMzMzMzMzA0ARzMzMzMzMCEAhT/OOU/SpfEAaGwnMzMzMzMwIQBFmZmZmZmYOQCGZ/5B++zraPxobCWZmZmZmZg5AEQAAAAAAABJAIZn/kH77Oto/GhsJAAAAAAAAEkARzMzMzMzMFEAhnoAmwoYeZEAaGwnMzMzMzMwUQBGZmZmZmZkXQCGZ/5B++zrqPxobCZmZmZmZmRdAEWZmZmZmZhpAIcDsnjwsPGlAGhsJZmZmZmZmGkARMzMzMzMzHUAhmf+Qfvs62j8aGwkzMzMzMzMdQBEAAAAAAAAgQCFlqmBUUq89QEKkAhobCQAAAAAAAPA/EQAAAAAAAPA/IWdmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAAAEAhZ2ZmZmZGXUAaGwkAAAAAAAAAQBEAAAAAAAAIQCFnZmZmZkZdQBobCQAAAAAAAAhAEQAAAAAAAAhAIWdmZmZmRl1AGhsJAAAAAAAACEARAAAAAAAACEAhZ2ZmZmZGXUAaGwkAAAAAAAAIQBEAAAAAAAAIQCFnZmZmZkZdQBobCQAAAAAAAAhAEQAAAAAAABRAIWdmZmZmRl1AGhsJAAAAAAAAFEARAAAAAAAAFEAhZ2ZmZmZGXUAaGwkAAAAAAAAUQBEAAAAAAAAYQCFnZmZmZkZdQBobCQAAAAAAABhAEQAAAAAAACBAIWdmZmZmRl1AIAFCEwoRYWRtaXNzaW9uX3R5cGVfaWQa0QcasgcKtgIIkwkYASABLQAAgD8ypAIaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQCABQJMJEcDLtEdoMRZAGSKQtnv3khxAKQAAAAAAAPA/MQAAAAAAAAhAOQAAAAAAADlAQqICGhsJAAAAAAAA8D8RMzMzMzMzC0AhJLn8h/Qah0AaGwkzMzMzMzMLQBEzMzMzMzMXQCGl374OnPNWQBobCTMzMzMzMxdAEWZmZmZmZiBAIV/fdLhDJGZAGhsJZmZmZmZmIEARMzMzMzMzJUAh05Fc/kP67T8aGwkzMzMzMzMlQBEAAAAAAAAqQCGMtj1ULfwDQBobCQAAAAAAACpAEczMzMzMzC5AIYi2PVQt/NM/GhsJzMzMzMzMLkARzczMzMzMMUAhkbY9VC380z8aGwnNzMzMzMwxQBEzMzMzMzM0QCGItj1ULfzTPxobCTMzMzMzMzRAEZmZmZmZmTZAIXRGlPYG/1lAGhsJmZmZmZmZNkARAAAAAAAAOUAhJuSDns0qS0BCpAIaGwkAAAAAAADwPxEAAAAAAADwPyFnZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWdmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZ2ZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAAAAQCFnZmZmZkZdQBobCQAAAAAAAABAEQAAAAAAAAhAIWdmZmZmRl1AGhsJAAAAAAAACEARAAAAAAAACEAhZ2ZmZmZGXUAaGwkAAAAAAAAIQBEAAAAAAAAUQCFnZmZmZkZdQBobCQAAAAAAABRAEQAAAAAAABhAIWdmZmZmRl1AGhsJAAAAAAAAGEARAAAAAAAANkAhZ2ZmZmZGXUAaGwkAAAAAAAA2QBEAAAAAAAA5QCFnZmZmZkZdQCABQhoKGGRpc2NoYXJnZV9kaXNwb3NpdGlvbl9pZBrMBxqyBwq2AgiTCRgBIAEtAACAPzKkAhobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AIAFAkwkRor4IyP+aGUAZIl9iUwMIHEApAAAAAAAA8D8xAAAAAAAA8D85AAAAAAAAMUBCogIaGwkAAAAAAADwPxHNzMzMzMwEQCF/+zpwTvGEQBobCc3MzMzMzARAEc3MzMzMzBBAISzUmuYdhylAGhsJzczMzMzMEEARNDMzMzMzF0AhmP+Qfvs6GkAaGwk0MzMzMzMXQBGamZmZmZkdQCEmUwWjkuphQBobCZqZmZmZmR1AEQAAAAAAACJAIdCRXP5D+u0/GhsJAAAAAAAAIkARNDMzMzMzJUAh2ZFc/kP6zT8aGwk0MzMzMzMlQBFnZmZmZmYoQCHQkVz+Q/rNPxobCWdmZmZmZihAEZqZmZmZmStAIdCRXP5D+s0/GhsJmpmZmZmZK0ARzczMzMzMLkAh0JFc/kP6zT8aGwnNzMzMzMwuQBEAAAAAAAAxQCE+6Nms+gR1QEKkAhobCQAAAAAAAPA/EQAAAAAAAPA/IWdmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZ2ZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFnZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWdmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZ2ZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAAAcQCFnZmZmZkZdQBobCQAAAAAAABxAEQAAAAAAABxAIWdmZmZmRl1AGhsJAAAAAAAAHEARAAAAAAAAMUAhZ2ZmZmZGXUAaGwkAAAAAAAAxQBEAAAAAAAAxQCFnZmZmZkZdQBobCQAAAAAAADFAEQAAAAAAADFAIWdmZmZmRl1AIAFCFQoTYWRtaXNzaW9uX3NvdXJjZV9pZBrJBxqyBwq2AgiTCRgBIAEtAACAPzKkAhobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AIAFAkwkRT21EfRGQD0AZ4PyYJ/C/AUApAAAAAAAA8D8xAAAAAAAACEA5AAAAAAAALEBCogIaGwkAAAAAAADwPxFmZmZmZmYCQCGx4emVsgZtQBobCWZmZmZmZgJAEc3MzMzMzAxAIWvecYqO4XdAGhsJzczMzMzMDEARmpmZmZmZE0AhZ9XnaissbUAaGwmamZmZmZkTQBHNzMzMzMwYQCGitDf4wnZnQBobCc3MzMzMzBhAEQAAAAAAAB5AIe4NvjCZAkdAGhsJAAAAAAAAHkARmpmZmZmZIUAhzxlR2hvMPkAaGwmamZmZmZkhQBEzMzMzMzMkQCFz+Q/ptydBQBobCTMzMzMzMyRAEc3MzMzMzCZAIfNBz2bVxyVAGhsJzczMzMzMJkARZ2ZmZmZmKUAhwqikTkDTGEAaGwlnZmZmZmYpQBEAAAAAAAAsQCGJH2PuWoIdQEKkAhobCQAAAAAAAPA/EQAAAAAAAABAIWdmZmZmRl1AGhsJAAAAAAAAAEARAAAAAAAACEAhZ2ZmZmZGXUAaGwkAAAAAAAAIQBEAAAAAAAAIQCFnZmZmZkZdQBobCQAAAAAAAAhAEQAAAAAAAAhAIWdmZmZmRl1AGhsJAAAAAAAACEARAAAAAAAACEAhZ2ZmZmZGXUAaGwkAAAAAAAAIQBEAAAAAAAAQQCFnZmZmZkZdQBobCQAAAAAAABBAEQAAAAAAABBAIWdmZmZmRl1AGhsJAAAAAAAAEEARAAAAAAAAFEAhZ2ZmZmZGXUAaGwkAAAAAAAAUQBEAAAAAAAAcQCFnZmZmZkZdQBobCQAAAAAAABxAEQAAAAAAACxAIWdmZmZmRl1AIAFCEgoQdGltZV9pbl9ob3NwaXRhbBrLBRACIrgFCrkCCNACEMMGGAEgAS0AAIA/MqQCGhsJAAAAAAAA8D8RAAAAAAAA8D8hzczMzMzMQEAaGwkAAAAAAADwPxEAAAAAAADwPyHNzMzMzMxAQBobCQAAAAAAAPA/EQAAAAAAAPA/Ic3MzMzMzEBAGhsJAAAAAAAA8D8RAAAAAAAA8D8hzczMzMzMQEAaGwkAAAAAAADwPxEAAAAAAADwPyHNzMzMzMxAQBobCQAAAAAAAPA/EQAAAAAAAPA/Ic3MzMzMzEBAGhsJAAAAAAAA8D8RAAAAAAAA8D8hzczMzMzMQEAaGwkAAAAAAADwPxEAAAAAAADwPyHNzMzMzMxAQBobCQAAAAAAAPA/EQAAAAAAAPA/Ic3MzMzMzEBAGhsJAAAAAAAA8D8RAAAAAAAA8D8hzczMzMzMQEAgAUDQAhALGg0SAk1DGQAAAAAAwGhAGg0SAkJDGQAAAAAAgElAGg0SAkhNGQAAAAAAAEBAGg0SAkNQGQAAAAAAADNAGg0SAlVOGQAAAAAAAChAGg0SAldDGQAAAAAAACRAGg0SAlNQGQAAAAAAABRAGg0SAk1EGQAAAAAAABRAGg0SAkRNGQAAAAAAAABAGg0SAlNJGQAAAAAAAPA/Gg0SAlBPGQAAAAAAAPA/JQAAAEAqzQEKDSICTUMpAAAAAADAaEAKEQgBEAEiAkJDKQAAAAAAgElAChEIAhACIgJITSkAAAAAAABAQAoRCAMQAyICQ1ApAAAAAAAAM0AKEQgEEAQiAlVOKQAAAAAAAChAChEIBRAFIgJXQykAAAAAAAAkQAoRCAYQBiICU1ApAAAAAAAAFEAKEQgHEAciAk1EKQAAAAAAABRAChEICBAIIgJETSkAAAAAAAAAQAoRCAkQCSICU0kpAAAAAAAA8D8KEQgKEAoiAlBPKQAAAAAAAPA/QgwKCnBheWVyX2NvZGUaqgMQAiKQAwq2AgiTCRgBIAEtAACAPzKkAhobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AIAFAkwkQARolEhpPcnRob3BlZGljcy1SZWNvbnN0cnVjdGl2ZRkAAAAAAEySQCUAANBBKicKJSIaT3J0aG9wZWRpY3MtUmVjb25zdHJ1Y3RpdmUpAAAAAABMkkBCEwoRbWVkaWNhbF9zcGVjaWFsdHkaywcasgcKtgIIkwkYASABLQAAgD8ypAIaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQCABQJMJEcUUex/QmEBAGcebazWd0DFAKQAAAAAAAPA/MQAAAAAAgEBAOQAAAAAAgFRAQqICGhsJAAAAAAAA8D8RMzMzMzMzIkAh5IOezaodYEAaGwkzMzMzMzMiQBEzMzMzMzMxQCFv8IXJVN9hQBobCTMzMzMzMzFAEczMzMzMTDlAIWyad5yi5WJAGhsJzMzMzMxMOUARMzMzMzOzQEAhg3NGlPZoZkAaGwkzMzMzM7NAQBEAAAAAAMBEQCEZ4lgXtxdlQBobCQAAAAAAwERAEczMzMzMzEhAIWuad5yi5WJAGhsJzMzMzMzMSEARmZmZmZnZTEAhkst/SL9PYkAaGwmZmZmZmdlMQBEzMzMzM3NQQCEo7Q2+ME1RQBobCTMzMzMzc1BAEZmZmZmZeVJAIZOHhVrTDDtAGhsJmZmZmZl5UkARAAAAAACAVEAhPsSxLm7jF0BCpAIaGwkAAAAAAADwPxEAAAAAAAAiQCFnZmZmZkZdQBobCQAAAAAAACJAEQAAAAAAADBAIWdmZmZmRl1AGhsJAAAAAAAAMEARAAAAAAAAN0AhZ2ZmZmZGXUAaGwkAAAAAAAA3QBEAAAAAAAA7QCFnZmZmZkZdQBobCQAAAAAAADtAEQAAAAAAgEBAIWdmZmZmRl1AGhsJAAAAAACAQEARAAAAAAAAQ0AhZ2ZmZmZGXUAaGwkAAAAAAABDQBEAAAAAAABGQCFnZmZmZkZdQBobCQAAAAAAAEZAEQAAAAAAAElAIWdmZmZmRl1AGhsJAAAAAAAASUARAAAAAACATEAhZ2ZmZmZGXUAaGwkAAAAAAIBMQBEAAAAAAIBUQCFnZmZmZkZdQCABQhQKEm51bV9sYWJfcHJvY2VkdXJlcxquBxqZBwq2AgiTCRgBIAEtAACAPzKkAhobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AIAFAkwkRjRQW2cUN+z8ZNkCGD1Vo8T8gKzEAAAAAAADwPzkAAAAAAAAYQEKZAhoSETMzMzMzM+M/ITws1JrmbUVAGhsJMzMzMzMz4z8RMzMzMzMz8z8h2PD0SlkWg0AaGwkzMzMzMzPzPxHMzMzMzMz8PyFcbcX+snvmPxobCczMzMzMzPw/ETMzMzMzMwNAISEf9GxWJXNAGhsJMzMzMzMzA0ARAAAAAAAACEAhXm3F/rJ75j8aGwkAAAAAAAAIQBHMzMzMzMwMQCFOQBNhw+tgQBobCczMzMzMzAxAEc3MzMzMzBBAIQAi/fZ1gEJAGhsJzczMzMzMEEARMzMzMzMzE0AhWW3F/rJ75j8aGwkzMzMzMzMTQBGZmZmZmZkVQCGl374OnPM2QBobCZmZmZmZmRVAEQAAAAAAABhAISbkg57NKitAQpsCGhIRAAAAAAAA8D8hZ2ZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFnZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWdmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZ2ZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFnZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAABAIWdmZmZmRl1AGhsJAAAAAAAAAEARAAAAAAAAAEAhZ2ZmZmZGXUAaGwkAAAAAAAAAQBEAAAAAAAAAQCFnZmZmZkZdQBobCQAAAAAAAABAEQAAAAAAAAhAIWdmZmZmRl1AGhsJAAAAAAAACEARAAAAAAAAGEAhZ2ZmZmZGXUAgAUIQCg5udW1fcHJvY2VkdXJlcxrIBxqyBwq2AgiTCRgBIAEtAACAPzKkAhobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AIAFAkwkR6GNNqlc1NEAZ/SipiTYaHkApAAAAAAAAAEAxAAAAAAAANEA5AAAAAAAATUBCogIaGwkAAAAAAAAAQBFmZmZmZmYeQCEf9GxWfc48QBobCWZmZmZmZh5AEWZmZmZmZipAIejZrPpcPWhAGhsJZmZmZmZmKkARzMzMzMzMMkAh3pOHhVrBcUAaGwnMzMzMzMwyQBFmZmZmZmY4QCF/2T15WAZ3QBobCWZmZmZmZjhAEQAAAAAAAD5AIWHD0ytldWVAGhsJAAAAAAAAPkARzMzMzMzMQUAh9bnaiv31VEAaGwnMzMzMzMxBQBGZmZmZmZlEQCHSkVz+Q/o9QBobCZmZmZmZmURAEWZmZmZmZkdAId/RJQmvyhtAGhsJZmZmZmZmR0ARMzMzMzMzSkAhirY9VC388z8aGwkzMzMzMzNKQBEAAAAAAABNQCEIEhQ/xtwAQEKkAhobCQAAAAAAAABAEQAAAAAAACZAIWdmZmZmRl1AGhsJAAAAAAAAJkARAAAAAAAALEAhZ2ZmZmZGXUAaGwkAAAAAAAAsQBEAAAAAAAAwQCFnZmZmZkZdQBobCQAAAAAAADBAEQAAAAAAADJAIWdmZmZmRl1AGhsJAAAAAAAAMkARAAAAAAAANEAhZ2ZmZmZGXUAaGwkAAAAAAAA0QBEAAAAAAAA1QCFnZmZmZkZdQBobCQAAAAAAADVAEQAAAAAAADhAIWdmZmZmRl1AGhsJAAAAAAAAOEARAAAAAAAAOkAhZ2ZmZmZGXUAaGwkAAAAAAAA6QBEAAAAAAAA+QCFnZmZmZkZdQBobCQAAAAAAAD5AEQAAAAAAAE1AIWdmZmZmRl1AIAFCEQoPbnVtX21lZGljYXRpb25zGpkGGoEGCrYCCJMJGAEgAS0AAIA/MqQCGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAgAUCTCREcG+wY+ljTPxlzAoczA2T1PyD1BzkAAAAAAABCQEKZAhoSEc3MzMzMzAxAIWb35GEh8ZFAGhsJzczMzMzMDEARzczMzMzMHEAhhxbZzvcTNUAaGwnNzMzMzMwcQBGamZmZmZklQCEi2/l+arziPxobCZqZmZmZmSVAEc3MzMzMzCxAIZgUYNp0RcM/GhsJzczMzMzMLEARAAAAAAAAMkAhmBRg2nRFwz8aGwkAAAAAAAAyQBGamZmZmZk1QCGbFGDadEXDPxobCZqZmZmZmTVAETMzMzMzMzlAIZYUYNp0RcM/GhsJMzMzMzMzOUARzczMzMzMPEAhmxRg2nRFwz8aGwnNzMzMzMw8QBEzMzMzMzNAQCGWFGDadEXDPxobCTMzMzMzM0BAEQAAAAAAAEJAIZsUYNp0RcM/QosBGgkhZ2ZmZmZGXUAaCSFnZmZmZkZdQBoJIWdmZmZmRl1AGgkhZ2ZmZmZGXUAaCSFnZmZmZkZdQBoJIWdmZmZmRl1AGgkhZ2ZmZmZGXUAaCSFnZmZmZkZdQBoSEQAAAAAAAPA/IWdmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAAQkAhZ2ZmZmZGXUAgAUITChFudW1iZXJfb3V0cGF0aWVudBqFBhruBQq2AgiTCRgBIAEtAACAPzKkAhobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AIAFAkwkR8Px5JnBlsD8ZDjpAa4+40j8g1Qg5AAAAAAAACEBCmQIaEhEzMzMzMzPTPyHAfR04J1WRQBobCTMzMzMzM9M/ETMzMzMzM+M/IV1txf6ye9Y/GhsJMzMzMzMz4z8RzMzMzMzM7D8hXG3F/rJ71j8aGwnMzMzMzMzsPxEzMzMzMzPzPyF5eqUsQ8RIQBobCTMzMzMzM/M/EQAAAAAAAPg/IV5txf6ye9Y/GhsJAAAAAAAA+D8RzMzMzMzM/D8hWW3F/rJ71j8aGwnMzMzMzMz8PxHNzMzMzMwAQCEqyxDHuhghQBobCc3MzMzMzABAETMzMzMzMwNAIVltxf6ye9Y/GhsJMzMzMzMzA0ARmZmZmZmZBUAhWW3F/rJ71j8aGwmZmZmZmZkFQBEAAAAAAAAIQCF8Nqs+V1v4P0J5GgkhZ2ZmZmZGXUAaCSFnZmZmZkZdQBoJIWdmZmZmRl1AGgkhZ2ZmZmZGXUAaCSFnZmZmZkZdQBoJIWdmZmZmRl1AGgkhZ2ZmZmZGXUAaCSFnZmZmZkZdQBoJIWdmZmZmRl1AGhIRAAAAAAAACEAhZ2ZmZmZGXUAgAUISChBudW1iZXJfZW1lcmdlbmN5GqoGGpMGCrYCCJMJGAEgAS0AAIA/MqQCGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAgAUCTCREecvKJ84LTPxk+7503wyLoPyCmBzkAAAAAAAAcQEKZAhoSEWZmZmZmZuY/Id6Th4XaMI1AGhsJZmZmZmZm5j8RZmZmZmZm9j8hQvFjzF1NZEAaGwlmZmZmZmb2PxHMzMzMzMwAQCGcM6K0NwBJQBobCczMzMzMzABAEWZmZmZmZgZAIZn/kH77Ouo/GhsJZmZmZmZmBkARAAAAAAAADEAhSZ2AJsJmK0AaGwkAAAAAAAAMQBHMzMzMzMwQQCEFEhQ/xtwQQBobCczMzMzMzBBAEZmZmZmZmRNAIZn/kH77Oto/GhsJmZmZmZmZE0ARZmZmZmZmFkAhmf+Qfvs62j8aGwlmZmZmZmYWQBEzMzMzMzMZQCHrUbgehWsHQBobCTMzMzMzMxlAEQAAAAAAABxAIfBaQj7o2f8/Qp0BGgkhZ2ZmZmZGXUAaCSFnZmZmZkZdQBoJIWdmZmZmRl1AGgkhZ2ZmZmZGXUAaCSFnZmZmZkZdQBoJIWdmZmZmRl1AGgkhZ2ZmZmZGXUAaEhEAAAAAAADwPyFnZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWdmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAAHEAhZ2ZmZmZGXUAgAUISChBudW1iZXJfaW5wYXRpZW50GtAREAIiwREKtgIIkwkYASABLQAAgD8ypAIaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQCABQJMJEE8aDhIDNzE1GQAAAAAAgHxAGg4SAzgyMBkAAAAAAEBgQBoOEgM3MjIZAAAAAADAVUAaDhIDOTk2GQAAAAAAAFVAGg4SAzgyNBkAAAAAAABGQBoOEgM3MzMZAAAAAAAAQEAaDhIDODEyGQAAAAAAAD9AGg4SAzk5OBkAAAAAAAA9QBoOEgM4MjMZAAAAAAAAMUAaDhIDNzI2GQAAAAAAADFAGg4SAzk5NxkAAAAAAAAuQBoOEgNWNTcZAAAAAAAALEAaDhIDNzI0GQAAAAAAACpAGg4SAzY4MhkAAAAAAAAqQBoQEgUyNTAuOBkAAAAAAAAmQBoOEgM3MzAZAAAAAAAAIkAaDhIDNzExGQAAAAAAACJAGhASBTI1MC43GQAAAAAAACBAGhASBTI1MC42GQAAAAAAACBAGg4SAzgyMhkAAAAAAAAcQCVTUURAKrgMCg4iAzcxNSkAAAAAAIB8QAoSCAEQASIDODIwKQAAAAAAQGBAChIIAhACIgM3MjIpAAAAAADAVUAKEggDEAMiAzk5NikAAAAAAABVQAoSCAQQBCIDODI0KQAAAAAAAEZAChIIBRAFIgM3MzMpAAAAAAAAQEAKEggGEAYiAzgxMikAAAAAAAA/QAoSCAcQByIDOTk4KQAAAAAAAD1AChIICBAIIgM4MjMpAAAAAAAAMUAKEggJEAkiAzcyNikAAAAAAAAxQAoSCAoQCiIDOTk3KQAAAAAAAC5AChIICxALIgNWNTcpAAAAAAAALEAKEggMEAwiAzcyNCkAAAAAAAAqQAoSCA0QDSIDNjgyKQAAAAAAACpAChQIDhAOIgUyNTAuOCkAAAAAAAAmQAoSCA8QDyIDNzMwKQAAAAAAACJAChIIEBAQIgM3MTEpAAAAAAAAIkAKFAgREBEiBTI1MC43KQAAAAAAACBAChQIEhASIgUyNTAuNikAAAAAAAAgQAoSCBMQEyIDODIyKQAAAAAAABxAChIIFBAUIgM3MjEpAAAAAAAAHEAKEggVEBUiAzgyMSkAAAAAAAAYQAoSCBYQFiIDODEzKQAAAAAAABhAChIIFxAXIgM3MTYpAAAAAAAAGEAKEggYEBgiAzcxNCkAAAAAAAAYQAoSCBkQGSIDODQwKQAAAAAAABRAChIIGhAaIgM3MzgpAAAAAAAAFEAKEggbEBsiAzcyNykAAAAAAAAUQAoSCBwQHCIDNzE4KQAAAAAAABRAChUIHRAdIgYyNTAuODEpAAAAAAAAFEAKEggeEB4iAzg0NCkAAAAAAAAQQAoSCB8QHyIDODA1KQAAAAAAABBAChIIIBAgIgM3MTcpAAAAAAAAEEAKEgghECEiAzY4MSkAAAAAAAAQQAoSCCIQIiIDNTE4KQAAAAAAABBAChIIIxAjIgM5NTgpAAAAAAAACEAKEggkECQiAzgwOCkAAAAAAAAIQAoSCCUQJSIDNzM2KQAAAAAAAAhAChIIJhAmIgM3MzQpAAAAAAAACEAKEggnECciA1Y1NCkAAAAAAAAAQAoSCCgQKCIDODg2KQAAAAAAAABAChIIKRApIgM4MjUpAAAAAAAAAEAKEggqECoiAzcyOSkAAAAAAAAAQAoSCCsQKyIDNTkyKQAAAAAAAABAChIILBAsIgM1NjcpAAAAAAAAAEAKEggtEC0iAzQ0MCkAAAAAAAAAQAoVCC4QLiIGMjUwLjgyKQAAAAAAAABAChIILxAvIgMxOTgpAAAAAAAAAEAKEggwEDAiA1Y1OCkAAAAAAADwPwoSCDEQMSIDVjQzKQAAAAAAAPA/ChIIMhAyIgM5NTUpAAAAAAAA8D8KEQgzEDMiAjk0KQAAAAAAAPA/ChIINBA0IgM5MjgpAAAAAAAA8D8KEgg1EDUiAzkyNCkAAAAAAADwPwoSCDYQNiIDOTIyKQAAAAAAAPA/ChIINxA3IgM5MTcpAAAAAAAA8D8KEgg4EDgiAzg4MykAAAAAAADwPwoSCDkQOSIDODM2KQAAAAAAAPA/ChIIOhA6IgM4MzEpAAAAAAAA8D8KEgg7EDsiAzgxNikAAAAAAADwPwoSCDwQPCIDODE1KQAAAAAAAPA/ChIIPRA9IgM4MTApAAAAAAAA8D8KEgg+ED4iAzc4NCkAAAAAAADwPwoSCD8QPyIDNzU2KQAAAAAAAPA/ChIIQBBAIgM3MzcpAAAAAAAA8D8KEghBEEEiAzczNSkAAAAAAADwPwoSCEIQQiIDNzMyKQAAAAAAAPA/ChIIQxBDIgM3MjgpAAAAAAAA8D8KEghEEEQiAzcxOSkAAAAAAADwPwoSCEUQRSIDNTMxKQAAAAAAAPA/ChIIRhBGIgM0ODYpAAAAAAAA8D8KEghHEEciAzQ1OSkAAAAAAADwPwoSCEgQSCIDNDQzKQAAAAAAAPA/ChIISRBJIgMzNTUpAAAAAAAA8D8KEghKEEoiAzI4MCkAAAAAAADwPwoSCEsQSyIDMjc1KQAAAAAAAPA/ChUITBBMIgYyNTAuODMpAAAAAAAA8D8KFAhNEE0iBTI1MC4yKQAAAAAAAPA/ChIIThBOIgMyMTUpAAAAAAAA8D9CCAoGZGlhZ18xGswgEAIivSAKuAIIiwkQCBgBIAEtAACAPzKkAhobCQAAAAAAAPA/EQAAAAAAAPA/ITMzMzMzE11AGhsJAAAAAAAA8D8RAAAAAAAA8D8hMzMzMzMTXUAaGwkAAAAAAADwPxEAAAAAAADwPyEzMzMzMxNdQBobCQAAAAAAAPA/EQAAAAAAAPA/ITMzMzMzE11AGhsJAAAAAAAA8D8RAAAAAAAA8D8hMzMzMzMTXUAaGwkAAAAAAADwPxEAAAAAAADwPyEzMzMzMxNdQBobCQAAAAAAAPA/EQAAAAAAAPA/ITMzMzMzE11AGhsJAAAAAAAA8D8RAAAAAAAA8D8hMzMzMzMTXUAaGwkAAAAAAADwPxEAAAAAAADwPyEzMzMzMxNdQBobCQAAAAAAAPA/EQAAAAAAAPA/ITMzMzMzE11AIAFAiwkQqQEaDhIDMjUwGQAAAAAAwGZAGg4SAzI4NRkAAAAAAIBZQBoOEgM0MDEZAAAAAAAAWEAaDhIDNDI4GQAAAAAAAEpAGhESBjI1MC4wMRkAAAAAAIBGQBoOEgM0OTYZAAAAAACAQUAaDhIDNDI3GQAAAAAAgEBAGg4SAzI3NhkAAAAAAAA8QBoOEgM5OTgZAAAAAAAAOkAaDhIDNzA3GQAAAAAAADpAGhESBjI1MC4wMhkAAAAAAAA4QBoOEgM1OTkZAAAAAAAANUAaDhIDNzE1GQAAAAAAADJAGg4SAzQ5MxkAAAAAAAAxQBoOEgM5OTcZAAAAAAAAMEAaDhIDNDE0GQAAAAAAAC5AGg4SAzY4MhkAAAAAAAAsQBoOEgM0MjQZAAAAAAAAKkAaDhIDNDAzGQAAAAAAACpAGg8SBEU4ODgZAAAAAAAAJkAlk6xQQCqwGwoOIgMyNTApAAAAAADAZkAKEggBEAEiAzI4NSkAAAAAAIBZQAoSCAIQAiIDNDAxKQAAAAAAAFhAChIIAxADIgM0MjgpAAAAAAAASkAKFQgEEAQiBjI1MC4wMSkAAAAAAIBGQAoSCAUQBSIDNDk2KQAAAAAAgEFAChIIBhAGIgM0MjcpAAAAAACAQEAKEggHEAciAzI3NikAAAAAAAA8QAoSCAgQCCIDOTk4KQAAAAAAADpAChIICRAJIgM3MDcpAAAAAAAAOkAKFQgKEAoiBjI1MC4wMikAAAAAAAA4QAoSCAsQCyIDNTk5KQAAAAAAADVAChIIDBAMIgM3MTUpAAAAAAAAMkAKEggNEA0iAzQ5MykAAAAAAAAxQAoSCA4QDiIDOTk3KQAAAAAAADBAChIIDxAPIgM0MTQpAAAAAAAALkAKEggQEBAiAzY4MikAAAAAAAAsQAoSCBEQESIDNDI0KQAAAAAAACpAChIIEhASIgM0MDMpAAAAAAAAKkAKEwgTEBMiBEU4ODgpAAAAAAAAJkAKEggUEBQiAzc4MCkAAAAAAAAmQAoSCBUQFSIDNzMzKQAAAAAAACZAChIIFhAWIgMyODApAAAAAAAAJkAKEwgXEBciBEU4ODUpAAAAAAAAJEAKEggYEBgiAzUxOCkAAAAAAAAiQAoUCBkQGSIFMjUwLjYpAAAAAAAAIkAKEggaEBoiAzk5NikAAAAAAAAgQAoSCBsQGyIDMjg3KQAAAAAAACBAChIIHBAcIgM3ODgpAAAAAAAAHEAKEggdEB0iAzc4MSkAAAAAAAAcQAoSCB4QHiIDNzMwKQAAAAAAABxAChIIHxAfIgM1NjApAAAAAAAAHEAKEgggECAiA1Y0NSkAAAAAAAAYQAoSCCEQISIDNzI0KQAAAAAAABhAChIIIhAiIgM3MjIpAAAAAAAAGEAKEggjECMiAzU4NCkAAAAAAAAYQAoSCCQQJCIDNDQwKQAAAAAAABhAChIIJRAlIgMzMDUpAAAAAAAAGEAKEggmECYiAzI3OCkAAAAAAAAYQAoSCCcQJyIDVjQzKQAAAAAAABRAChMIKBAoIgRFODc4KQAAAAAAABRAChIIKRApIgM3MzYpAAAAAAAAFEAKEggqECoiAzcyNikAAAAAAAAUQAoSCCsQKyIDNzE4KQAAAAAAABRAChIILBAsIgM3MTEpAAAAAAAAFEAKEggtEC0iAzQxMykAAAAAAAAUQAoRCC4QLiICNDEpAAAAAAAAFEAKEggvEC8iAzgyNCkAAAAAAAAQQAoSCDAQMCIDNzMxKQAAAAAAABBAChIIMRAxIgM3MjcpAAAAAAAAEEAKEggyEDIiAzcxNykAAAAAAAAQQAoSCDMQMyIDNTcxKQAAAAAAABBAChIINBA0IgM0OTIpAAAAAAAAEEAKEgg1EDUiAzQwMikAAAAAAAAQQAoSCDYQNiIDMjcyKQAAAAAAABBAChIINxA3IgM5OTkpAAAAAAAACEAKEgg4EDgiAzg0MCkAAAAAAAAIQAoSCDkQOSIDODEyKQAAAAAAAAhAChIIOhA6IgM3ODUpAAAAAAAACEAKEgg7EDsiAzU4NSkAAAAAAAAIQAoSCDwQPCIDNDI2KQAAAAAAAAhAChIIPRA9IgM0MTUpAAAAAAAACEAKFQg+ED4iBjI1MC45MikAAAAAAAAIQAoVCD8QPyIGMjUwLjUxKQAAAAAAAAhAChIIQBBAIgMyNDIpAAAAAAAACEAKEghBEEEiAzEzNSkAAAAAAAAIQAoSCEIQQiIDVjU0KQAAAAAAAABAChIIQxBDIgNWMTUpAAAAAAAAAEAKEwhEEEQiBEU4ODApAAAAAAAAAEAKEwhFEEUiBEU4NDkpAAAAAAAAAEAKEghGEEYiAzgyMCkAAAAAAAAAQAoSCEcQRyIDODEzKQAAAAAAAABAChIISBBIIgM3MjMpAAAAAAAAAEAKEghJEEkiAzcyMSkAAAAAAAAAQAoSCEoQSiIDNzE5KQAAAAAAAABAChEISxBLIgI3MCkAAAAAAAAAQAoSCEwQTCIDNDg2KQAAAAAAAABAChEITRBNIgI0MikAAAAAAAAAQAoSCE4QTiIDNDEyKQAAAAAAAABAChIITxBPIgM0MTApAAAAAAAAAEAKEghQEFAiAzMwMykAAAAAAAAAQAoSCFEQUSIDMjkyKQAAAAAAAABAChUIUhBSIgYyNTAuODEpAAAAAAAAAEAKFQhTEFMiBjI1MC40MSkAAAAAAAAAQAoVCFQQVCIGMjUwLjAzKQAAAAAAAABAChIIVRBVIgMyNDQpAAAAAAAAAEAKEghWEFYiAzExMikAAAAAAAAAQAoSCFcQVyIDVjcyKQAAAAAAAPA/ChIIWBBYIgNWNjMpAAAAAAAA8D8KEghZEFkiA1Y1NykAAAAAAADwPwoSCFoQWiIDVjQyKQAAAAAAAPA/ChIIWxBbIgNWMTQpAAAAAAAA8D8KEghcEFwiA1YxMikAAAAAAADwPwoSCF0QXSIDVjEwKQAAAAAAAPA/ChIIXhBeIgNWMDkpAAAAAAAA8D8KEwhfEF8iBEU5MjgpAAAAAAAA8D8KEwhgEGAiBEU5MTkpAAAAAAAA8D8KEwhhEGEiBEU5MTcpAAAAAAAA8D8KEwhiEGIiBEU4ODQpAAAAAAAA8D8KEwhjEGMiBEU4MTcpAAAAAAAA8D8KEwhkEGQiBEU4MTQpAAAAAAAA8D8KEghlEGUiAzk1OSkAAAAAAADwPwoRCGYQZiICOTQpAAAAAAAA8D8KEghnEGciAzkyMikAAAAAAADwPwoSCGgQaCIDOTE4KQAAAAAAAPA/ChIIaRBpIgM5MDYpAAAAAAAA8D8KEghqEGoiAzkwNSkAAAAAAADwPwoSCGsQayIDODQ1KQAAAAAAAPA/ChIIbBBsIgM4NDQpAAAAAAAA8D8KEghtEG0iAzgzNykAAAAAAADwPwoSCG4QbiIDODI1KQAAAAAAAPA/ChIIbxBvIgM4MjMpAAAAAAAA8D8KEghwEHAiAzgyMSkAAAAAAADwPwoSCHEQcSIDODE2KQAAAAAAAPA/ChIIchByIgM4MDUpAAAAAAAA8D8KEghzEHMiAzgwMikAAAAAAADwPwoQCHQQdCIBOCkAAAAAAADwPwoSCHUQdSIDNzkwKQAAAAAAAPA/ChIIdhB2IgM3ODcpAAAAAAAA8D8KEgh3EHciAzc4NikAAAAAAADwPwoSCHgQeCIDNzg0KQAAAAAAAPA/ChIIeRB5IgM3NTYpAAAAAAAA8D8KEgh6EHoiAzczOCkAAAAAAADwPwoSCHsQeyIDNzM0KQAAAAAAAPA/ChIIfBB8IgM3MTQpAAAAAAAA8D8KEgh9EH0iAzcxMCkAAAAAAADwPwoSCH4QfiIDNjg0KQAAAAAAAPA/ChIIfxB/IgM1OTYpAAAAAAAA8D8KFAiAARCAASIDNTk1KQAAAAAAAPA/ChQIgQEQgQEiAzU5MykAAAAAAADwPwoUCIIBEIIBIgM1OTEpAAAAAAAA8D8KFAiDARCDASIDNTgzKQAAAAAAAPA/ChQIhAEQhAEiAzU4MSkAAAAAAADwPwoUCIUBEIUBIgM1NzgpAAAAAAAA8D8KFAiGARCGASIDNTMzKQAAAAAAAPA/ChQIhwEQhwEiAzUzMCkAAAAAAADwPwoUCIgBEIgBIgM0OTEpAAAAAAAA8D8KFAiJARCJASIDNDY1KQAAAAAAAPA/ChQIigEQigEiAzQ1OCkAAAAAAADwPwoUCIsBEIsBIgM0NTMpAAAAAAAA8D8KFAiMARCMASIDNDQ0KQAAAAAAAPA/ChQIjQEQjQEiAzQzMykAAAAAAADwPwoUCI4BEI4BIgM0MjUpAAAAAAAA8D8KFAiPARCPASIDMzk3KQAAAAAAAPA/ChMIkAEQkAEiAjM4KQAAAAAAAPA/ChQIkQEQkQEiAzM2NSkAAAAAAADwPwoUCJIBEJIBIgMzNTUpAAAAAAAA8D8KFAiTARCTASIDMzI0KQAAAAAAAPA/ChQIlAEQlAEiAzMxMSkAAAAAAADwPwoUCJUBEJUBIgMyOTMpAAAAAAAA8D8KFAiWARCWASIDMjkxKQAAAAAAAPA/ChQIlwEQlwEiAzI4NCkAAAAAAADwPwoUCJgBEJgBIgMyODIpAAAAAAAA8D8KFAiZARCZASIDMjc3KQAAAAAAAPA/ChQImgEQmgEiAzI1OCkAAAAAAADwPwoXCJsBEJsBIgYyNTAuODMpAAAAAAAA8D8KFwicARCcASIGMjUwLjgyKQAAAAAAAPA/ChYInQEQnQEiBTI1MC43KQAAAAAAAPA/ChcIngEQngEiBjI1MC41MykAAAAAAADwPwoXCJ8BEJ8BIgYyNTAuNTIpAAAAAAAA8D8KFwigARCgASIGMjUwLjQyKQAAAAAAAPA/ChYIoQEQoQEiBTI1MC40KQAAAAAAAPA/ChQIogEQogEiAzIwNCkAAAAAAADwPwoUCKMBEKMBIgMyMDIpAAAAAAAA8D8KFAikARCkASIDMTk4KQAAAAAAAPA/ChQIpQEQpQEiAzE5NikAAAAAAADwPwoUCKYBEKYBIgMxOTEpAAAAAAAA8D8KFAinARCnASIDMTg5KQAAAAAAAPA/ChQIqAEQqAEiAzE1MykAAAAAAADwP0IICgZkaWFnXzIasyEQAiKkIQq4AgjpCBAqGAEgAS0AAIA/MqQCGhsJAAAAAAAA8D8RAAAAAAAA8D8hmpmZmZk5XEAaGwkAAAAAAADwPxEAAAAAAADwPyGamZmZmTlcQBobCQAAAAAAAPA/EQAAAAAAAPA/IZqZmZmZOVxAGhsJAAAAAAAA8D8RAAAAAAAA8D8hmpmZmZk5XEAaGwkAAAAAAADwPxEAAAAAAADwPyGamZmZmTlcQBobCQAAAAAAAPA/EQAAAAAAAPA/IZqZmZmZOVxAGhsJAAAAAAAA8D8RAAAAAAAA8D8hmpmZmZk5XEAaGwkAAAAAAADwPxEAAAAAAADwPyGamZmZmTlcQBobCQAAAAAAAPA/EQAAAAAAAPA/IZqZmZmZOVxAGhsJAAAAAAAA8D8RAAAAAAAA8D8hmpmZmZk5XEAgAUDpCBCuARoOEgMyNTAZAAAAAAAgbUAaDhIDNDAxGQAAAAAAoGVAGg4SAzI4NRkAAAAAAIBEQBoOEgMyNzIZAAAAAAAAQ0AaDhIDNDI4GQAAAAAAgEFAGg4SAzI3NhkAAAAAAAA3QBoPEgRFODQ5GQAAAAAAADZAGg4SAzQxNBkAAAAAAAA0QBoQEgUyNTAuNhkAAAAAAAAxQBoOEgMyNDQZAAAAAAAAMUAaDhIDNDI3GQAAAAAAAC5AGg4SAzk5OBkAAAAAAAAsQBoOEgM1OTkZAAAAAAAALEAaDRICNDEZAAAAAAAAKkAaDhIDVjQzGQAAAAAAAChAGg4SAzczMBkAAAAAAAAmQBoOEgM3MDcZAAAAAAAAJkAaDhIDNDk2GQAAAAAAACZAGg4SAzI3OBkAAAAAAAAmQBoOEgNWNDUZAAAAAAAAJEAlYeZIQCqcHAoOIgMyNTApAAAAAAAgbUAKEggBEAEiAzQwMSkAAAAAAKBlQAoSCAIQAiIDMjg1KQAAAAAAgERAChIIAxADIgMyNzIpAAAAAAAAQ0AKEggEEAQiAzQyOCkAAAAAAIBBQAoSCAUQBSIDMjc2KQAAAAAAADdAChMIBhAGIgRFODQ5KQAAAAAAADZAChIIBxAHIgM0MTQpAAAAAAAANEAKFAgIEAgiBTI1MC42KQAAAAAAADFAChIICRAJIgMyNDQpAAAAAAAAMUAKEggKEAoiAzQyNykAAAAAAAAuQAoSCAsQCyIDOTk4KQAAAAAAACxAChIIDBAMIgM1OTkpAAAAAAAALEAKEQgNEA0iAjQxKQAAAAAAACpAChIIDhAOIgNWNDMpAAAAAAAAKEAKEggPEA8iAzczMCkAAAAAAAAmQAoSCBAQECIDNzA3KQAAAAAAACZAChIIERARIgM0OTYpAAAAAAAAJkAKEggSEBIiAzI3OCkAAAAAAAAmQAoSCBMQEyIDVjQ1KQAAAAAAACRAChIIFBAUIgM0OTMpAAAAAAAAJEAKEggVEBUiAzMwNSkAAAAAAAAkQAoVCBYQFiIGMjUwLjAxKQAAAAAAACRAChMIFxAXIgRFODg1KQAAAAAAACJAChIIGBAYIgM3MjcpAAAAAAAAIkAKEggZEBkiAzQwMykAAAAAAAAiQAoSCBoQGiIDOTk3KQAAAAAAACBAChIIGxAbIgM3MTUpAAAAAAAAIEAKEggcEBwiAzY4MikAAAAAAAAgQAoTCB0QHSIERTg4OCkAAAAAAAAcQAoSCB4QHiIDNTMwKQAAAAAAABxAChIIHxAfIgMyODApAAAAAAAAHEAKEgggECAiAzcyNikAAAAAAAAYQAoSCCEQISIDNTYwKQAAAAAAABhAChIIIhAiIgM1MTgpAAAAAAAAGEAKEggjECMiAzQxMykAAAAAAAAYQAoVCCQQJCIGMjUwLjAyKQAAAAAAABhAChIIJRAlIgNWNTgpAAAAAAAAFEAKEggmECYiA1Y1NCkAAAAAAAAUQAoSCCcQJyIDNzg4KQAAAAAAABRAChIIKBAoIgM3MzYpAAAAAAAAFEAKEggpECkiAzczMykAAAAAAAAUQAoSCCoQKiIDNzMxKQAAAAAAABRAChIIKxArIgM1ODQpAAAAAAAAFEAKEggsECwiAzQyNCkAAAAAAAAUQAoSCC0QLSIDMzY1KQAAAAAAABRAChIILhAuIgNWMTApAAAAAAAAEEAKEwgvEC8iBEU4NzgpAAAAAAAAEEAKEggwEDAiAzk5NikAAAAAAAAQQAoSCDEQMSIDOTA1KQAAAAAAABBAChIIMhAyIgM4MTMpAAAAAAAAEEAKEggzEDMiAzcyMikAAAAAAAAQQAoSCDQQNCIDNzE5KQAAAAAAABBAChIINRA1IgM0MTIpAAAAAAAAEEAKEgg2EDYiAzM1NykAAAAAAAAQQAoSCDcQNyIDMjkyKQAAAAAAABBAChIIOBA4IgNWNDIpAAAAAAAACEAKEgg5EDkiA1YxNSkAAAAAAAAIQAoTCDoQOiIERTg4NCkAAAAAAAAIQAoSCDsQOyIDODQwKQAAAAAAAAhAChIIPBA8IgM4MTIpAAAAAAAACEAKEgg9ED0iAzc4MCkAAAAAAAAIQAoSCD4QPiIDNzE4KQAAAAAAAAhAChIIPxA/IgM3MTYpAAAAAAAACEAKEghAEEAiAzcxMykAAAAAAAAIQAoSCEEQQSIDNjgxKQAAAAAAAAhAChIIQhBCIgM0OTIpAAAAAAAACEAKEghDEEMiAzQ0MykAAAAAAAAIQAoSCEQQRCIDNDI1KQAAAAAAAAhAChIIRRBFIgMzOTYpAAAAAAAACEAKEghGEEYiA1YxMikAAAAAAAAAQAoSCEcQRyIDOTIzKQAAAAAAAABAChIISBBIIgM4MjUpAAAAAAAAAEAKEghJEEkiAzc5MCkAAAAAAAAAQAoSCEoQSiIDNzgxKQAAAAAAAABAChIISxBLIgM3MjkpAAAAAAAAAEAKEghMEEwiAzcyNCkAAAAAAAAAQAoSCE0QTSIDNzE0KQAAAAAAAABAChIIThBOIgM3MTEpAAAAAAAAAEAKEghPEE8iAzY5NikAAAAAAAAAQAoSCFAQUCIDNTg1KQAAAAAAAABAChIIURBRIgM1NjQpAAAAAAAAAEAKEghSEFIiAzQ4NikAAAAAAAAAQAoSCFMQUyIDNDU3KQAAAAAAAABAChIIVBBUIgM0NTMpAAAAAAAAAEAKEghVEFUiAzQ0MCkAAAAAAAAAQAoSCFYQViIDNDI2KQAAAAAAAABAChIIVxBXIgM0MDIpAAAAAAAAAEAKEghYEFgiAzM2OSkAAAAAAAAAQAoSCFkQWSIDMjk1KQAAAAAAAABAChIIWhBaIgMyODYpAAAAAAAAAEAKFAhbEFsiBTI1MC43KQAAAAAAAABAChUIXBBcIgYyNTAuNTIpAAAAAAAAAEAKFQhdEF0iBjI1MC40MikAAAAAAAAAQAoVCF4QXiIGMjUwLjQxKQAAAAAAAABAChIIXxBfIgNWNjQpAAAAAAAA8D8KEghgEGAiA1YxNCkAAAAAAADwPwoTCGEQYSIERTk2NSkAAAAAAADwPwoTCGIQYiIERTkzNSkAAAAAAADwPwoTCGMQYyIERTkyOSkAAAAAAADwPwoTCGQQZCIERTkyOCkAAAAAAADwPwoTCGUQZSIERTkyNykAAAAAAADwPwoTCGYQZiIERTkxNikAAAAAAADwPwoTCGcQZyIERTg4NykAAAAAAADwPwoTCGgQaCIERTg4MykAAAAAAADwPwoTCGkQaSIERTg4MCkAAAAAAADwPwoTCGoQaiIERTgyMikAAAAAAADwPwoTCGsQayIERTgxOSkAAAAAAADwPwoTCGwQbCIERTgxNykAAAAAAADwPwoSCG0QbSIDOTU2KQAAAAAAAPA/ChIIbhBuIgM5MDgpAAAAAAAA8D8KEghvEG8iAzkwNikAAAAAAADwPwoSCHAQcCIDODkyKQAAAAAAAPA/ChIIcRBxIgM4OTEpAAAAAAAA8D8KEghyEHIiAzg4MykAAAAAAADwPwoSCHMQcyIDODM2KQAAAAAAAPA/ChIIdBB0IgM4MjEpAAAAAAAA8D8KEgh1EHUiAzgyMCkAAAAAAADwPwoSCHYQdiIDODE2KQAAAAAAAPA/ChIIdxB3IgM4MDcpAAAAAAAA8D8KEAh4EHgiATgpAAAAAAAA8D8KEgh5EHkiAzc5OSkAAAAAAADwPwoSCHoQeiIDNzg3KQAAAAAAAPA/ChIIexB7IgM3ODYpAAAAAAAA8D8KEgh8EHwiAzc0NykAAAAAAADwPwoSCH0QfSIDNzM4KQAAAAAAAPA/ChIIfhB+IgM3MzcpAAAAAAAA8D8KEgh/EH8iAzcyMSkAAAAAAADwPwoUCIABEIABIgM3MTcpAAAAAAAA8D8KFAiBARCBASIDNzEyKQAAAAAAAPA/ChQIggEQggEiAzcwNCkAAAAAAADwPwoUCIMBEIMBIgM2OTgpAAAAAAAA8D8KFAiEARCEASIDNjA1KQAAAAAAAPA/ChQIhQEQhQEiAzYwMCkAAAAAAADwPwoUCIYBEIYBIgM1NzMpAAAAAAAA8D8KFAiHARCHASIDNTYyKQAAAAAAAPA/ChQIiAEQiAEiAzU1NSkAAAAAAADwPwoUCIkBEIkBIgM1NTMpAAAAAAAA8D8KFAiKARCKASIDNTM2KQAAAAAAAPA/ChQIiwEQiwEiAzQ5MSkAAAAAAADwPwoUCIwBEIwBIgM0NzMpAAAAAAAA8D8KEwiNARCNASICNDIpAAAAAAAA8D8KFAiOARCOASIDNDE2KQAAAAAAAPA/ChQIjwEQjwEiAzQxNSkAAAAAAADwPwoUCJABEJABIgM0MTEpAAAAAAAA8D8KEwiRARCRASICMzgpAAAAAAAA8D8KFAiSARCSASIDMzYyKQAAAAAAAPA/ChQIkwEQkwEiAzM1NCkAAAAAAADwPwoUCJQBEJQBIgMzNTEpAAAAAAAA8D8KFAiVARCVASIDMzQ4KQAAAAAAAPA/ChQIlgEQlgEiAzM0NykAAAAAAADwPwoTCJcBEJcBIgIzNCkAAAAAAADwPwoUCJgBEJgBIgMzMzcpAAAAAAAA8D8KFAiZARCZASIDMzMyKQAAAAAAAPA/ChQImgEQmgEiAzMxMCkAAAAAAADwPwoUCJsBEJsBIgMzMDMpAAAAAAAA8D8KFAicARCcASIDMjk0KQAAAAAAAPA/ChQInQEQnQEiAzI5MykAAAAAAADwPwoUCJ4BEJ4BIgMyODgpAAAAAAAA8D8KFAifARCfASIDMjg3KQAAAAAAAPA/ChQIoAEQoAEiAzI3NCkAAAAAAADwPwoUCKEBEKEBIgMyNjMpAAAAAAAA8D8KFAiiARCiASIDMjU2KQAAAAAAAPA/ChQIowEQowEiAzI1MykAAAAAAADwPwoXCKQBEKQBIgYyNTAuOTIpAAAAAAAA8D8KFwilARClASIGMjUwLjgxKQAAAAAAAPA/ChYIpgEQpgEiBTI1MC41KQAAAAAAAPA/ChcIpwEQpwEiBjI1MC40MykAAAAAAADwPwoXCKgBEKgBIgYyNTAuMDMpAAAAAAAA8D8KFAipARCpASIDMjA0KQAAAAAAAPA/ChQIqgEQqgEiAzE5OCkAAAAAAADwPwoUCKsBEKsBIgMxNzQpAAAAAAAA8D8KFAisARCsASIDMTM1KQAAAAAAAPA/ChQIrQEQrQEiAzExMikAAAAAAADwP0IICgZkaWFnXzMayQcasgcKtgIIkwkYASABLQAAgD8ypAIaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQCABQJMJEavrTm1EfRlAGVSPBDo1SgFAKQAAAAAAAABAMQAAAAAAABhAOQAAAAAAACJAQqICGhsJAAAAAAAAAEARmpmZmZmZBUAhlrIMcaxjQUAaGwmamZmZmZkFQBEzMzMzMzMLQCFAguLHmGdUQBobCTMzMzMzMwtAEWZmZmZmZhBAIXDOiNLeqmFAGhsJZmZmZmZmEEARMzMzMzMzE0Ahmf+Qfvs66j8aGwkzMzMzMzMTQBEAAAAAAAAWQCHbiv1l955rQBobCQAAAAAAABZAEczMzMzMzBhAIbaEfNCzi2JAGhsJzMzMzMzMGEARmZmZmZmZG0Ahmf+Qfvs66j8aGwmZmZmZmZkbQBFmZmZmZmYeQCEwKqkT0IBYQBobCWZmZmZmZh5AEZqZmZmZmSBAIZRliGNd2FpAGhsJmpmZmZmZIEARAAAAAAAAIkAhLGUZ4lgOdUBCpAIaGwkAAAAAAAAAQBEAAAAAAAAQQCFnZmZmZkZdQBobCQAAAAAAABBAEQAAAAAAABBAIWdmZmZmRl1AGhsJAAAAAAAAEEARAAAAAAAAFEAhZ2ZmZmZGXUAaGwkAAAAAAAAUQBEAAAAAAAAUQCFnZmZmZkZdQBobCQAAAAAAABRAEQAAAAAAABhAIWdmZmZmRl1AGhsJAAAAAAAAGEARAAAAAAAAHEAhZ2ZmZmZGXUAaGwkAAAAAAAAcQBEAAAAAAAAgQCFnZmZmZkZdQBobCQAAAAAAACBAEQAAAAAAACJAIWdmZmZmRl1AGhsJAAAAAAAAIkARAAAAAAAAIkAhZ2ZmZmZGXUAaGwkAAAAAAAAiQBEAAAAAAAAiQCFnZmZmZkZdQCABQhIKEG51bWJlcl9kaWFnbm9zZXMa7AMQAiLWAwq2AgiTCRgBIAEtAACAPzKkAhobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AIAFAkwkQBBoPEgROb25lGQAAAAAAeI9AGg8SBE5vcm0ZAAAAAABAWkAaDxIEPjIwMBkAAAAAAABFQBoPEgQ+MzAwGQAAAAAAADFAJQAAgEAqUAoPIgROb25lKQAAAAAAeI9AChMIARABIgROb3JtKQAAAAAAQFpAChMIAhACIgQ+MjAwKQAAAAAAAEVAChMIAxADIgQ+MzAwKQAAAAAAADFAQg8KDW1heF9nbHVfc2VydW0a4AMQAiLOAwq2AgiTCRgBIAEtAACAPzKkAhobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AIAFAkwkQBBoPEgROb25lGQAAAAAAzJBAGg0SAj44GQAAAAAAgEVAGg8SBE5vcm0ZAAAAAAAAQEAaDRICPjcZAAAAAAAANUAlGAF5QCpMCg8iBE5vbmUpAAAAAADMkEAKEQgBEAEiAj44KQAAAAAAgEVAChMIAhACIgROb3JtKQAAAAAAAEBAChEIAxADIgI+NykAAAAAAAA1QEILCglBMUNyZXN1bHQa5AMQAiLSAwq2AgiTCRgBIAEtAACAPzKkAhobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AIAFAkwkQBBoNEgJObxkAAAAAADiJQBoREgZTdGVhZHkZAAAAAAAQdUAaDRICVXAZAAAAAAAAMEAaDxIERG93bhkAAAAAAAAmQCVN4EpAKk4KDSICTm8pAAAAAAA4iUAKFQgBEAEiBlN0ZWFkeSkAAAAAABB1QAoRCAIQAiICVXApAAAAAAAAMEAKEwgDEAMiBERvd24pAAAAAAAAJkBCCwoJbWV0Zm9ybWluGsQDEAIisAMKtgIIkwkYASABLQAAgD8ypAIaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQCABQJMJEAMaDRICTm8ZAAAAAAAYkkAaERIGU3RlYWR5GQAAAAAAAChAGg8SBERvd24ZAAAAAAAA8D8lk7sCQCo7Cg0iAk5vKQAAAAAAGJJAChUIARABIgZTdGVhZHkpAAAAAAAAKEAKEwgCEAIiBERvd24pAAAAAAAA8D9CDQoLcmVwYWdsaW5pZGUangMQAiKKAwq2AgiTCRgBIAEtAACAPzKkAhobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AIAFAkwkQAhoNEgJObxkAAAAAACySQBoREgZTdGVhZHkZAAAAAAAAIEAlur8BQComCg0iAk5vKQAAAAAALJJAChUIARABIgZTdGVhZHkpAAAAAAAAIEBCDQoLbmF0ZWdsaW5pZGUaoQMQAiKKAwq2AgiTCRgBIAEtAACAPzKkAhobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AIAFAkwkQAhoNEgJObxkAAAAAAEiSQBoREgZTdGVhZHkZAAAAAAAA8D8l9zcAQComCg0iAk5vKQAAAAAASJJAChUIARABIgZTdGVhZHkpAAAAAAAA8D9CEAoOY2hsb3Jwcm9wYW1pZGUa5gMQAiLSAwq2AgiTCRgBIAEtAACAPzKkAhobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AIAFAkwkQBBoNEgJObxkAAAAAAICRQBoREgZTdGVhZHkZAAAAAAAARkAaDRICVXAZAAAAAAAAEEAaDxIERG93bhkAAAAAAAAIQCVy8glAKk4KDSICTm8pAAAAAACAkUAKFQgBEAEiBlN0ZWFkeSkAAAAAAABGQAoRCAIQAiICVXApAAAAAAAAEEAKEwgDEAMiBERvd24pAAAAAAAACEBCDQoLZ2xpbWVwaXJpZGUa9gIQAiLgAgq2AgiTCRgBIAEtAACAPzKkAhobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AIAFAkwkQARoNEgJObxkAAAAAAEySQCUAAABAKg8KDSICTm8pAAAAAABMkkBCDwoNYWNldG9oZXhhbWlkZRrkAxACItIDCrYCCJMJGAEgAS0AAIA/MqQCGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAgAUCTCRAEGg0SAk5vGQAAAAAAsI5AGhESBlN0ZWFkeRkAAAAAACBlQBoNEgJVcBkAAAAAAAAkQBoPEgREb3duGQAAAAAAACRAJQ4KJkAqTgoNIgJObykAAAAAALCOQAoVCAEQASIGU3RlYWR5KQAAAAAAIGVAChEIAhACIgJVcCkAAAAAAAAkQAoTCAMQAyIERG93bikAAAAAAAAkQEILCglnbGlwaXppZGUa5AMQAiLSAwq2AgiTCRgBIAEtAACAPzKkAhobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AIAFAkwkQBBoNEgJObxkAAAAAAACPQBoREgZTdGVhZHkZAAAAAACAY0AaDRICVXAZAAAAAAAALEAaDxIERG93bhkAAAAAAAAiQCWEFiNAKk4KDSICTm8pAAAAAAAAj0AKFQgBEAEiBlN0ZWFkeSkAAAAAAIBjQAoRCAIQAiICVXApAAAAAAAALEAKEwgDEAMiBERvd24pAAAAAAAAIkBCCwoJZ2x5YnVyaWRlGp4DEAIiigMKtgIIkwkYASABLQAAgD8ypAIaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQCABQJMJEAIaDRICTm8ZAAAAAABIkkAaERIGU3RlYWR5GQAAAAAAAPA/Jfc3AEAqJgoNIgJObykAAAAAAEiSQAoVCAEQASIGU3RlYWR5KQAAAAAAAPA/Qg0KC3RvbGJ1dGFtaWRlGucDEAIi0gMKtgIIkwkYASABLQAAgD8ypAIaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQCABQJMJEAQaDRICTm8ZAAAAAACUkEAaERIGU3RlYWR5GQAAAAAAAFpAGg0SAlVwGQAAAAAAABBAGg8SBERvd24ZAAAAAAAAAEAlavQWQCpOCg0iAk5vKQAAAAAAlJBAChUIARABIgZTdGVhZHkpAAAAAAAAWkAKEQgCEAIiAlVwKQAAAAAAABBAChMIAxADIgREb3duKQAAAAAAAABAQg4KDHBpb2dsaXRhem9uZRroAxACItIDCrYCCJMJGAEgAS0AAIA/MqQCGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAgAUCTCRAEGg0SAk5vGQAAAAAAYJBAGhESBlN0ZWFkeRkAAAAAAABdQBoNEgJVcBkAAAAAAAAUQBoPEgREb3duGQAAAAAAAABAJQGUGUAqTgoNIgJObykAAAAAAGCQQAoVCAEQASIGU3RlYWR5KQAAAAAAAF1AChEIAhACIgJVcCkAAAAAAAAUQAoTCAMQAyIERG93bikAAAAAAAAAQEIPCg1yb3NpZ2xpdGF6b25lGpsDEAIiigMKtgIIkwkYASABLQAAgD8ypAIaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQCABQJMJEAIaDRICTm8ZAAAAAABAkkAaERIGU3RlYWR5GQAAAAAAAAhAJeanAEAqJgoNIgJObykAAAAAAECSQAoVCAEQASIGU3RlYWR5KQAAAAAAAAhAQgoKCGFjYXJib3NlGvECEAIi4AIKtgIIkwkYASABLQAAgD8ypAIaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQCABQJMJEAEaDRICTm8ZAAAAAABMkkAlAAAAQCoPCg0iAk5vKQAAAAAATJJAQgoKCG1pZ2xpdG9sGvUCEAIi4AIKtgIIkwkYASABLQAAgD8ypAIaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQCABQJMJEAEaDRICTm8ZAAAAAABMkkAlAAAAQCoPCg0iAk5vKQAAAAAATJJAQg4KDHRyb2dsaXRhem9uZRqdAxACIooDCrYCCJMJGAEgAS0AAIA/MqQCGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAgAUCTCRACGg0SAk5vGQAAAAAARJJAGhESBlN0ZWFkeRkAAAAAAAAAQCXvbwBAKiYKDSICTm8pAAAAAABEkkAKFQgBEAEiBlN0ZWFkeSkAAAAAAAAAQEIMCgp0b2xhemFtaWRlGvACEAIi4AIKtgIIkwkYASABLQAAgD8ypAIaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQCABQJMJEAEaDRICTm8ZAAAAAABMkkAlAAAAQCoPCg0iAk5vKQAAAAAATJJAQgkKB2V4YW1pZGUa9AIQAiLgAgq2AgiTCRgBIAEtAACAPzKkAhobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AIAFAkwkQARoNEgJObxkAAAAAAEySQCUAAABAKg8KDSICTm8pAAAAAABMkkBCDQoLY2l0b2dsaXB0b24a4gMQAiLSAwq2AgiTCRgBIAEtAACAPzKkAhobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AIAFAkwkQBBoNEgJObxkAAAAAADiGQBoREgZTdGVhZHkZAAAAAACgckAaDxIERG93bhkAAAAAAEBWQBoNEgJVcBkAAAAAAEBSQCVN4EpAKk4KDSICTm8pAAAAAAA4hkAKFQgBEAEiBlN0ZWFkeSkAAAAAAKByQAoTCAIQAiIERG93bikAAAAAAEBWQAoRCAMQAyICVXApAAAAAABAUkBCCQoHaW5zdWxpbhrIAxACIqwDCrYCCJMJGAEgAS0AAIA/MqQCGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAgAUCTCRADGg0SAk5vGQAAAAAAMJJAGhESBlN0ZWFkeRkAAAAAAAAYQBoNEgJVcBkAAAAAAADwPyXMTwFAKjkKDSICTm8pAAAAAAAwkkAKFQgBEAEiBlN0ZWFkeSkAAAAAAAAYQAoRCAIQAiICVXApAAAAAAAA8D9CFQoTZ2x5YnVyaWRlLW1ldGZvcm1pbhr8AhACIuACCrYCCJMJGAEgAS0AAIA/MqQCGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAgAUCTCRABGg0SAk5vGQAAAAAATJJAJQAAAEAqDwoNIgJObykAAAAAAEySQEIVChNnbGlwaXppZGUtbWV0Zm9ybWluGoEDEAIi4AIKtgIIkwkYASABLQAAgD8ypAIaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQCABQJMJEAEaDRICTm8ZAAAAAABMkkAlAAAAQCoPCg0iAk5vKQAAAAAATJJAQhoKGGdsaW1lcGlyaWRlLXBpb2dsaXRhem9uZRqAAxACIuACCrYCCJMJGAEgAS0AAIA/MqQCGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAgAUCTCRABGg0SAk5vGQAAAAAATJJAJQAAAEAqDwoNIgJObykAAAAAAEySQEIZChdtZXRmb3JtaW4tcm9zaWdsaXRhem9uZRr/AhACIuACCrYCCJMJGAEgAS0AAIA/MqQCGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAgAUCTCRABGg0SAk5vGQAAAAAATJJAJQAAAEAqDwoNIgJObykAAAAAAEySQEIYChZtZXRmb3JtaW4tcGlvZ2xpdGF6b25lGpEDEAIiggMKtgIIkwkYASABLQAAgD8ypAIaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQCABQJMJEAIaDRICTm8ZAAAAAABAg0AaDRICQ2gZAAAAAABYgUAlAAAAQCoiCg0iAk5vKQAAAAAAQINAChEIARABIgJDaCkAAAAAAFiBQEIICgZjaGFuZ2UamAMQAiKEAwq2AgiTCRgBIAEtAACAPzKkAhobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AIAFAkwkQAhoOEgNZZXMZAAAAAABojEAaDRICTm8ZAAAAAABgcEAlPa4xQCojCg4iA1llcykAAAAAAGiMQAoRCAEQASICTm8pAAAAAABgcEBCDQoLZGlhYmV0ZXNNZWQauwMQAiKoAwq2AgiTCRgBIAEtAACAPzKkAhobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AGhsJAAAAAAAA8D8RAAAAAAAA8D8hZmZmZmZGXUAaGwkAAAAAAADwPxEAAAAAAADwPyFmZmZmZkZdQBobCQAAAAAAAPA/EQAAAAAAAPA/IWZmZmZmRl1AIAFAkwkQAxoNEgJOTxkAAAAAAMiIQBoOEgM+MzAZAAAAAAAAckAaDhIDPDMwGQAAAAAAgFZAJcaoFEAqNwoNIgJOTykAAAAAAMiIQAoSCAEQASIDPjMwKQAAAAAAAHJAChIIAhACIgM8MzApAAAAAACAVkBCDAoKcmVhZG1pdHRlZA=="></facets-overview>';
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
