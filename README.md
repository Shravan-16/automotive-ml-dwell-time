# automotive-ml-dwell-time
# Overview

This project develops a MATLAB-based machine learning model for predicting bus dwell time and supporting predictive engine start-stop decisions.

## Objective

The objective of the thesis is to investigate if it is possible to use for example map, traffic and vehicle data to plan when to shutdown the engine to maximise downtime with the least amount of engine shutdowns by creating a prediction model for buses using a suitable machine learning (ML) algorithm. One example where this would be applicable is city buses that travel on the same route every day and hence can provide reliable data on when and where it is standing still.

## Machine Learning Models

- Linear Regression
- Support Vector Regression (SVM)
- Gradient Boosting Machine (GBM)
- Random Forest (RF)

## Input Variables

- Boarding Count
- Alighting Count

## Target Variable

- Dwell Time (seconds)

## Start-Stop Strategy

A 40-second dwell-time threshold is used as the engine shutdown decision threshold.

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

- MATLAB/Simulink
