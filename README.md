# TalkingData AdTracking Challenge

This repository contains some piece of code for Kaggle competition [TalkingData AdTracking Fraud Detection Challenge](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection).

For this competition, the objective is to predict whether a user will download an app after clicking a mobile app advertisement.

Because the dataset is quite big, I use pyspark to extract manually engineered features.

Some techniques like feature hashing can also help reduce the dimensionality and is showcased here to build a logistic regression score.

There is also a quick implementation of an XGBoost Model in R that runs fast especially after reducing the size of the training dataset.

## Data

The dataset covers approximately 200 million clicks over 4 days (3 days for training and last day for test).

Each row of the training data contains a click record, with the following features.

* ip: ip address of click.
* app: app id for marketing.
* device: device type id of user mobile phone (e.g., iphone 6 plus, iphone 7, huawei mate 7, etc.)
* os: os version id of user mobile phone
* channel: channel id of mobile ad publisher
* click_time: timestamp of click (UTC)
* attributed_time: if user download the app for after clicking an ad, this is the time of the app download
* is_attributed: the target that is to be predicted, indicating the app was downloaded

## Feature Engineering

The main handcrafted features are about:

* the time lapse to previous and next click with same ip, app, device, os
* Number of clicks with same (ip, os), (ip, device), (ip, app),.. during the same minute, hour or the whole week.


## Feature Hashing

Another approach to deal with this big data problem is to use feature hashing (random projection technique) applied to the original categoricals features (ip, device, app, os, channel).
I previously did a manual implemntation from scrach but there are easier and more common implementations of feature hashing in famous labraries:

* Spark HashingTF (I share some code here on how to use it)
* Spark FeatureHasher (more direct use but requires spark version 2.3 or later)
* Python hashlib library


## Modeling

The 3 kinds of models trained here are:

* logistic regression on hashed features: trained on 40% of the training data, cross-validation on 10% and scoring on the remaining 50% of the training data. The idea here is to generate a score from the linear model and stack it on a gradient boosting trained later. Using feature hashing can play a considerable role of regularization (reducing the overfitting due to the effect of ips, apps, etc.)
* random forest (serves only for training a stacking model and evaluating the performance of features quickly using spark)
* xgboost (final model to generate predictions using the same features and trained in R)
