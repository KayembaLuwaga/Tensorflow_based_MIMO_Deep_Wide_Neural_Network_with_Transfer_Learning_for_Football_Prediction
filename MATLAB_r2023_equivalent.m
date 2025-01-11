% Install dependencies
% MATLAB does not use pip; we use built-in functions and toolboxes.

% Load libraries
% MATLAB's equivalents for Python libraries like pandas and scikit-learn
% would be the built-in functions for table and data processing, statistics, etc.

% Load data
filename = 'E2.csv';
data = readtable(filename);

% Display basic data information
disp('Data head:');
disp(head(data));
disp('Data columns:');
disp(data.Properties.VariableNames);
disp('Data info:');
disp(summary(data));

% Define feature and target columns
text_categorical_features = {'Time', 'HoTe', 'AwTe', 'Ref'};
integer_target_columns = {'FTHG', 'FTAG', 'HTHG', 'HTAG', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 'HBP', 'ABP', 'HTTG', 'FTTG', 'TS', 'TST', 'TC', 'TF', 'TY', 'TR'};
y_columns = {'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 'HBP', 'ABP', 'HTTG', 'FTTG', 'TS', 'TST', 'TC', 'TF', 'TY', 'TR'};
text_categorical_target_columns = {'FTR', 'HTR'};

% One-hot encode 'Time' column
X_encoded_time = dummyvar(categorical(data.Time));
X_encoded_time = array2table(X_encoded_time, 'VariableNames', strcat('Time_', string(1:size(X_encoded_time, 2))));

% Combine one-hot-encoded 'Time' with other categorical features
data = removevars(data, {'Time'});
data_encoded = [data X_encoded_time];

% One-hot encode other text-categorical features
for feature = text_categorical_features
    X_encoded_feature = dummyvar(categorical(data.(feature{1})));
    feature_names = strcat(feature{1}, '_', string(1:size(X_encoded_feature, 2)));
    X_encoded_feature = array2table(X_encoded_feature, 'VariableNames', feature_names);
    data_encoded = [data_encoded, X_encoded_feature];
end

% Remove target columns
data_encoded = removevars(data_encoded, integer_target_columns);

% Scale integer target columns
data_scaled = data(:, integer_target_columns);
data_scaled = normalize(data_scaled, 'range', [0 1]);
data_encoded = [data_encoded data_scaled];

% Display the resulting encoded data
disp('Encoded data:');
disp(head(data_encoded));

% Prepare data for training
X = table2array(data_encoded(:, ~ismember(data_encoded.Properties.VariableNames, y_columns)));
y = table2array(data_encoded(:, ismember(data_encoded.Properties.VariableNames, y_columns)));

% Split the data into training and testing sets (90% train, 10% test)
cv = cvpartition(size(X, 1), 'HoldOut', 0.1);
X_train = X(training(cv), :);
X_test = X(test(cv), :);
y_train = y(training(cv), :);
y_test = y(test(cv), :);

% Define the neural network model
layers = [
    featureInputLayer(size(X, 2))
    fullyConnectedLayer(64, 'Activation', 'relu')
    fullyConnectedLayer(32, 'Activation', 'relu')
    fullyConnectedLayer(16, 'Activation', 'relu')
    fullyConnectedLayer(size(y, 2), 'Activation', 'sigmoid')
    classificationLayer
];

% Options for training the model
options = trainingOptions('adam', 'MaxEpochs', 50, 'MiniBatchSize', 16, 'ValidationSplit', 0.1, 'Verbose', false);

% Train the model
model = trainNetwork(X_train, y_train, layers, options);

% Save the model
save('my_model.mat', 'model');

% Evaluate the model on the test set
y_pred = predict(model, X_test);

% Convert predicted probabilities to binary values
y_pred_binary = y_pred > 0.5;

% Calculate mean absolute error for each output
mae_scores = mean(abs(y_pred_binary - y_test), 1);
mean_mae = mean(mae_scores);
disp(['Mean MAE: ', num2str(mean_mae)]);

% Display results for a new input
new_data = table({'15:00'}', {'Blackpool'}', {'Burton'}', {'S Stockbridge'}', 'VariableNames', {'Time', 'HoTe', 'AwTe', 'Ref'});

% One-hot encode 'Time' for new data
new_data_encoded_time = dummyvar(categorical(new_data.Time));
new_data_encoded_time = array2table(new_data_encoded_time, 'VariableNames', strcat('Time_', string(1:size(new_data_encoded_time, 2))));

% Combine new data with other categorical features
new_data_encoded = [new_data(:, 2:end) new_data_encoded_time];

% Reorder columns to match the original model input (X_train columns)
new_data_encoded = new_data_encoded(:, X_train.Properties.VariableNames);
new_data_encoded = table2array(new_data_encoded);

% Make predictions on the new data
prediction = predict(model, new_data_encoded);

% Display the predictions
disp('Prediction:');
disp(prediction);
