% Load the data
data = readtable('bus_schedule_data_more_realistic.csv');

%% ===============================
%  1. Distribution of Dwell Time
% ===============================
figure;
histogram(data.DwellTime, 30, 'FaceColor', [0.3 0.5 0.8]);
xlabel('Dwell Time (seconds)');
ylabel('Frequency');
title('Distribution of Dwell Time');
grid on;

%% ===============================
%  2. Boarding & Alighting Counts
% ===============================
figure;
subplot(1,2,1);
histogram(data.BoardingCount, 'FaceColor', [0.2 0.7 0.4]);
xlabel('Boarding Count'); ylabel('Frequency');
title('Boarding Count Distribution'); grid on;

subplot(1,2,2);
histogram(data.AlightingCount, 'FaceColor', [0.8 0.4 0.4]);
xlabel('Alighting Count'); ylabel('Frequency');
title('Alighting Count Distribution'); grid on;

% Features and target variable
X = data{:, {'BoardingCount', 'AlightingCount'}}; % Features Variable
y = data{:, 'DwellTime'}; % Target Variable

% Linear regression model
% Train a linear regression model
linearModel = fitlm(X, y);

% Predictions of Linear regression model
y_pred_linear = predict(linearModel, X);

% Calculate MSE, MAE, R² for Linear regression model
mse_linear = mean((y - y_pred_linear).^2);
mae_linear = mean(abs(y - y_pred_linear));
r2_linear = 1 - sum((y - y_pred_linear).^2) / sum((y - mean(y)).^2);

% Display MSE, MAE, R² for Linear regression model
fprintf('Linear Model MSE: %.2f\n', mse_linear);
fprintf('Linear Model MAE: %.2f\n', mae_linear);
fprintf('Linear Model R²: %.2f\n', r2_linear);

% SVM
% Train an SVM model
svmModel = fitrsvm(X, y);

% Predictions for SVM model
y_pred_svm = predict(svmModel, X);

% Calculate MSE, MAE,R² for SVM model
mse_svm = mean((y - y_pred_svm).^2);
mae_svm = mean(abs(y - y_pred_svm));
r2_svm = 1 - sum((y - y_pred_svm).^2) / sum((y - mean(y)).^2);

% Display MSE, MAE, R² for SVM model
fprintf('SVM Model MSE: %.2f\n', mse_svm);
fprintf('SVM Model MAE: %.2f\n', mae_svm);
fprintf('SVM Model R²: %.2f\n', r2_svm);


% Gradient Boosting Machine model
% Train a Gradient Boosting Machine model
gbmModel = fitrensemble(X, y, 'Method', 'LSBoost');

% Predictions for Gradient Boosting Machine model
y_pred_gbm = predict(gbmModel, X);

% Calculate MSE,MAE,R² for GBM Model
mse_gbm = mean((y - y_pred_gbm).^2);
mae_gbm = mean(abs(y - y_pred_gbm));
r2_gbm = 1 - sum((y - y_pred_gbm).^2) / sum((y - mean(y)).^2);

% Display MSE, MAE, R² for GBM Model
fprintf('GBM Model MSE: %.2f\n', mse_gbm);
fprintf('GBM Model MAE: %.2f\n', mae_gbm);
fprintf('GBM Model R²: %.2f\n', r2_gbm);

% Random Forest model
% Train a Random Forest model
rfModel = TreeBagger(100, X, y, 'Method', 'regression');

% Predictions for Random Forest model
y_pred_rf = predict(rfModel, X);

% Convert predictions to numeric values
%y_pred_rf = str2double(y_pred_rf);

% Calculate MSE,MAE, R² for Random Forest Model
mse_rf = mean((y - y_pred_rf).^2);
mae_rf = mean(abs(y - y_pred_rf));
r2_rf = 1 - sum((y - y_pred_rf).^2) / sum((y - mean(y)).^2);

% Display MSE, MAE, R² for Random Forest Model
fprintf('Random Forest Model MSE: %.2f\n', mse_rf);
fprintf('Random Forest Model MAE: %.2f\n', mae_rf);
fprintf('Random Forest Model R²: %.2f\n', r2_rf);

% Store MSEs, MAEs, and R²s in arrays
mse_array = [mse_linear, mse_svm, mse_gbm, mse_rf];
mae_array = [mae_linear, mae_svm, mae_gbm, mae_rf];
r2_array = [r2_linear, r2_svm, r2_gbm, r2_rf];

%model_names = {'Linear Model', 'SVM Model', 'GBM Model', 'Random Forest Model'};

% Find the index of the model with the lowest MSE
[~, bestModelIdxMSE] = min(mse_array);

% Find the index of the model with the lowest MAE
[~, bestModelIdxMAE] = min(mae_array);

% Find the index of the model with the highest R²
[~, bestModelIdxR2] = max(r2_array);

model_names = {'Linear Model', 'SVM Model', 'GBM Model', 'Random Forest Model'};
% Display the best models
fprintf('The best model based on MSE is: %s with MSE: %.2f\n', model_names{bestModelIdxMSE}, mse_array(bestModelIdxMSE));
fprintf('The best model based on MAE is: %s with MAE: %.2f\n', model_names{bestModelIdxMAE}, mae_array(bestModelIdxMAE));
fprintf('The best model based on R² is: %s with R²: %.2f\n', model_names{bestModelIdxR2}, r2_array(bestModelIdxR2));

% Plot for Actual vs Predicted values for each ML models
% Linear Model
figure;
scatter(y, y_pred_linear);
hold on;
plot([min(y) max(y)], [min(y) max(y)], 'r');
xlabel('Actual Dwell Time (seconds)');
ylabel('Predicted Dwell Time (seconds)');
title('Linear Model');
hold off;

% SVM Model
figure;
scatter(y, y_pred_svm);
hold on;
plot([min(y) max(y)], [min(y) max(y)], 'r');
xlabel('Actual Dwell Time (seconds)');
ylabel('Predicted Dwell Time (seconds)');
title('SVM Model');
hold off;

% GBM Model
figure;
scatter(y, y_pred_gbm);
hold on;
plot([min(y) max(y)], [min(y) max(y)], 'r');
xlabel('Actual Dwell Time (seconds)');
ylabel('Predicted Dwell Time (seconds)');
title('GBM Model');
hold off;

% Random Forest Model
figure;
scatter(y, y_pred_rf);
hold on;
plot([min(y) max(y)], [min(y) max(y)], 'r');
xlabel('Actual Dwell Time (seconds)');
ylabel('Predicted Dwell Time (seconds)');
title('Random Forest Model');
hold off;

% Count the total number of bus stops
num_bus_stops = size(X, 1);

% Threshold for engine shutdown
engine_threshold = 40;

% The number of actual engine stops.
num_engine_stops_actual = 100;

% Count the number of engine stops where predicted dwell time exceeds the threshold
num_engine_stops_predicted = sum(y_pred_gbm > engine_threshold);

% Determine if the engine should shut down based on predicted dwell time
engine_shutdown_predicted = y_pred_gbm > engine_threshold;

% Plot comparison of actual vs predicted dwell times
%figure;
%plot(1:num_bus_stops, y, '-ob', 'LineWidth', 1.5, 'MarkerSize', 4); % Actual dwell time with blue circles
%hold on;
%plot(1:num_bus_stops, y_pred_gbm, '-sr', 'LineWidth', 1.5, 'MarkerSize', 4); % Predicted dwell time with red squares
%legend('Actual Dwell Time', 'Predicted Dwell Time (GBM)');
%xlabel('Bus Stop Number');
%ylabel('Dwell Time (seconds)');
%title('Comparison of Actual Dwell Time vs Predicted Dwell Time');
%hold off;

figure;
plot(1:num_bus_stops, y, '-ob', 'LineWidth', 1.5); % Actual dwell time 
hold on;
plot(1:num_bus_stops, y_pred_gbm, '-sr', 'LineWidth', 1.5); % Predicted dwell time (GBM model)
legend('Actual Dwell Time', 'Predicted Dwell Time (GBM)', 'FontSize', 16);
xlabel('Bus Stop Number', 'FontSize', 16);
ylabel('Dwell Time (seconds)', 'FontSize', 16);
title('Comparison of Actual Dwell Time vs Predicted Dwell Time', 'FontSize', 18);

set(gca, 'FontSize', 14); % Axis tick labels

% Plot the comparison of actual engine stops vs predicted engine stops with threshold 
figure;
bar([num_engine_stops_actual, num_engine_stops_predicted], 'FaceColor', 'flat');
set(gca, 'xticklabel', {'Actual Stops', 'Predicted Stops (GBM)'});
ylabel('Number of Engine Stops');
title('Comparison of Actual Engine Stops vs Predicted Engine Stops (Threshold > 40 Seconds)');
grid on;

% Display the number of actual engine stops and predicted engine stops
disp(['Number of actual engine stops: ', num2str(num_engine_stops_actual)]);
disp(['Number of predicted engine stops (GBM) exceeding 40 seconds: ', num2str(num_engine_stops_predicted)]);