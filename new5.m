%% ============================================================
%  Load Dataset
% ============================================================
data = readtable('bus_schedule_data_more_realistic.csv');

%% ============================================================
%  1. Distribution of Dwell Time
% ============================================================
figure;
histogram(data.DwellTime, 30, 'FaceColor', [0.3 0.5 0.8]);
xlabel('Dwell Time (seconds)');
ylabel('Frequency');
title('Distribution of Dwell Time');
grid on;

%% ============================================================
%  2. Boarding & Alighting Counts
% ============================================================
figure;
subplot(1,2,1);
histogram(data.BoardingCount, 'FaceColor', [0.2 0.7 0.4]);
xlabel('Boarding Count'); ylabel('Frequency');
title('Boarding Count Distribution'); grid on;

subplot(1,2,2);
histogram(data.AlightingCount, 'FaceColor', [0.8 0.4 0.4]);
xlabel('Alighting Count'); ylabel('Frequency');
title('Alighting Count Distribution'); grid on;

%% ============================================================
%  Prepare Features & Target
% ============================================================
X = data{:, {'BoardingCount', 'AlightingCount'}};
y = data{:, 'DwellTime'};

%% ============================================================
%  Linear Regression Model
% ============================================================
linearModel = fitlm(X, y);
y_pred_linear = predict(linearModel, X);

mse_linear = mean((y - y_pred_linear).^2);
mae_linear = mean(abs(y - y_pred_linear));
r2_linear  = 1 - sum((y - y_pred_linear).^2) / sum((y - mean(y)).^2);

%% ============================================================
%  SVM Regression
% ============================================================
svmModel = fitrsvm(X, y);
y_pred_svm = predict(svmModel, X);

mse_svm = mean((y - y_pred_svm).^2);
mae_svm = mean(abs(y - y_pred_svm));
r2_svm  = 1 - sum((y - y_pred_svm).^2) / sum((y - mean(y)).^2);

%% ============================================================
%  Gradient Boosting (GBM)
% ============================================================
gbmModel = fitrensemble(X, y, 'Method', 'LSBoost');
y_pred_gbm = predict(gbmModel, X);

mse_gbm = mean((y - y_pred_gbm).^2);
mae_gbm = mean(abs(y - y_pred_gbm));
r2_gbm  = 1 - sum((y - y_pred_gbm).^2) / sum((y - mean(y)).^2);

%% ============================================================
%  Random Forest (TreeBagger)
% ============================================================
rfModel = TreeBagger(100, X, y, 'Method', 'regression');
y_pred_rf = predict(rfModel, X);

mse_rf = mean((y - y_pred_rf).^2);
mae_rf = mean(abs(y - y_pred_rf));
r2_rf  = 1 - sum((y - y_pred_rf).^2) / sum((y - mean(y)).^2);

%% ============================================================
%  Best Model Comparison
% ============================================================
mse_array = [mse_linear, mse_svm, mse_gbm, mse_rf];
mae_array = [mae_linear, mae_svm, mae_gbm, mae_rf];
r2_array  = [r2_linear, r2_svm, r2_gbm, r2_rf];

model_names = {'Linear', 'SVM', 'GBM', 'Random Forest'};

% Best MSE
[minMSE, idxMSE] = min(mse_array);
fprintf('\nBest Model (MSE): %s (%.2f)\n', model_names{idxMSE}, minMSE);

% Best MAE
[minMAE, idxMAE] = min(mae_array);
fprintf('Best Model (MAE): %s (%.2f)\n', model_names{idxMAE}, minMAE);

% Best R^2
[maxR2, idxR2] = max(r2_array);
fprintf('Best Model (R^2): %s (%.2f)\n', model_names{idxR2}, maxR2);

%% ============================================================
%  Select Only 20 Random Points for Scatter Plots
% ============================================================
rng(1);  % reproducible
plot_idx = randperm(length(y), 20);

y20 = y(plot_idx);
y_pred_linear20 = y_pred_linear(plot_idx);
y_pred_svm20    = y_pred_svm(plot_idx);
y_pred_gbm20    = y_pred_gbm(plot_idx);
y_pred_rf20     = y_pred_rf(plot_idx);

%% ============================================================
%  Scatter Plots (20 Points)
% ============================================================
plotScatter(y20, y_pred_linear20, 'Linear Regression Model');
plotScatter(y20, y_pred_svm20,    'SVM Model');
plotScatter(y20, y_pred_gbm20,    'GBM Model');
plotScatter(y20, y_pred_rf20,     'Random Forest Model');

%% ============================================================
%  Engine Shutdown Threshold Analysis
% ============================================================
engine_threshold = 40;
num_bus_stops = length(y);
num_engine_stops_actual = 100;

num_engine_stops_predicted = sum(y_pred_gbm > engine_threshold);

fprintf('\nActual Engine Stops: %d\n', num_engine_stops_actual);
fprintf('Predicted Engine Stops (GBM >40s): %d\n', num_engine_stops_predicted);

%% ============================================================
%  Engine Shutdown Threshold Analysis (3-Bar Comparison)
% ============================================================
engine_threshold = 40;

% Total number of stops (baseline)
num_engine_stops_actual = length(y);

% Ideal real-life case: using measured dwell time
num_engine_stops_measured = sum(y >= engine_threshold);

% Prediction-based case: using GBM predicted dwell time
num_engine_stops_predicted = sum(y_pred_gbm >= engine_threshold);

fprintf('\nEngine Stop Comparison (Threshold = 40 s)\n');
fprintf('Actual Engine Stops (Baseline): %d\n', num_engine_stops_actual);
fprintf('Measured ≥40s Engine Stops (Ideal): %d\n', num_engine_stops_measured);
fprintf('Predicted ≥40s Engine Stops (GBM): %d\n', num_engine_stops_predicted);

%% ============================================================
%  Bar Plot of Engine Stops
% ============================================================
figure;
bar([num_engine_stops_actual, num_engine_stops_predicted]);
set(gca, 'xticklabel', {'Actual Stops', 'Predicted Stops (GBM)'});
ylabel('Number of Engine Stops');
title('Engine Stop Comparison (>40 sec threshold)');
grid on;

%% ============================================================
%  Bar Plot – Actual vs Measured vs Predicted Engine Stops
% ============================================================
figure;
bar([num_engine_stops_actual, ...
     num_engine_stops_measured, ...
     num_engine_stops_predicted]);

set(gca, 'XTickLabel', ...
    {'Actual Stops', 'Measured ≥40s', 'Predicted ≥40s (GBM)'});

ylabel('Number of Engine Stops');
title('Comparison of Engine Stops Using Measured and Predicted Dwell Time');
grid on;

%% ============================================================
%  End of Script
%  (Function Definitions Below)
% ============================================================


%% ============================================================
%  Function: plotScatter
% ============================================================
function plotScatter(yt, yp, titleText)
    figure;
    scatter(yt, yp, 70, 'o', 'MarkerFaceColor', [0.4 0.6 1], 'MarkerEdgeColor','k');
    hold on;
    plot([min(yt) max(yt)], [min(yt) max(yt)], 'r-', 'LineWidth', 1.5);
    xlabel('Actual Dwell Time (seconds)', 'FontSize', 14);
    ylabel('Predicted Dwell Time (seconds)', 'FontSize', 14);
    title(titleText, 'FontSize', 16);
    grid on;
    hold off;
end
