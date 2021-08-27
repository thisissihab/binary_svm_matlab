clear all;
clc;
%% Prepare the dataset

x = banknotes(:, 1:4);
y = banknotes(:,5);

rand = randperm(1372);

xtr = x(rand(1:1100), :);
ytr = y(rand(1:1100), :);

xt = x(rand(1101:end), :);
yt = y(rand(1101:end), :);

%% Training the model
model = fitcsvm(xtr, ytr, 'KernelFunction', 'rbf', ...
    'OptimizeHyperparameters', 'auto', ...
    'HyperparameterOptimizationOptions', struct('AcquisitionFunctionName', ...
    'expected-improvement-plus', 'ShowPlots', true));

%% Test the model
result = predict(model, xt);
accuracy = sum(result == yt)/length(yt)*100;
sp = sprintf("Test Accuracy = %.2f", accuracy);
disp(sp);
