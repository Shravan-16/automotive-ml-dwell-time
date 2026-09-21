# automotive-ml-dwell-time
# Overview

This project develops a MATLAB-based machine learning model for
predicting bus dwell time and supporting predictive engine
start-stop decisions.

## Objective

The objective is to predict dwell time at bus stops and determine
whether the engine should be switched off when the predicted
stopping duration is sufficiently long.

## Machine Learning Models

- Linear Regression
- Support Vector Regression (SVM)
- Gradient Boosting Machine (GBM)

## Input Variables

- Boarding Count
- Alighting Count

## Target Variable

- Dwell Time (seconds)

## Start-Stop Strategy

A 40-second dwell-time threshold is used as the engine shutdown
decision threshold.

If predicted dwell time >= 40 seconds:

    Engine OFF

Otherwise:

    Engine ON

## Performance Metrics

The models are evaluated using:

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R²

## Tools

- MATLAB
