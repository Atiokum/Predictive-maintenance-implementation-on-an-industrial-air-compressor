%% Data Loading and Overview
% Load the dataset
data = readtable('data.csv'); % Replace with your dataset path

% Display dataset overview
disp('Dataset Overview:');
disp(head(data));
disp(['Dataset Dimensions: ', num2str(size(data, 1)), ' rows x ', num2str(size(data, 2)), ' columns']);

%% Data Preprocessing
% Convert categorical target variables to numeric
data.bearings = double(strcmp(data.bearings, 'Ok')); % 1 for 'Ok', 0 otherwise
data.wpump = double(strcmp(data.wpump, 'Ok')); % 1 for 'Ok', 0 otherwise
data.radiator = double(strcmp(data.radiator, 'Clean')); % 1 for 'Clean', 0 otherwise
data.exvalve = double(strcmp(data.exvalve, 'Clean')); % 1 for 'Clean', 0 otherwise
data.acmotor = double(strcmp(data.acmotor, 'Stable')); % 1 for 'Stable', 0 otherwise

% Normalize features to [0, 1] range
features = {'rpm', 'motor_power', 'torque', 'outlet_pressure_bar', 'air_flow', ...
    'noise_db', 'outlet_temp', 'wpump_outlet_press', 'water_inlet_temp', ...
    'water_outlet_temp', 'wpump_power', 'water_flow', 'oilpump_power', ...
    'oil_tank_temp', 'gaccx', 'gaccy', 'gaccz', 'haccx', 'haccy', 'haccz'};
X = table2array(data(:, features)); % Predictor variables
X = normalize(X, 'range'); % Normalize predictors

% Define target variables for training separate models
targets = {'bearings', 'wpump', 'radiator', 'exvalve', 'acmotor'};

%% Distribution of Categorical Variables
% Identify categorical columns
categorical_columns = {'bearings', 'wpump', 'radiator', 'exvalve', 'acmotor'}; 

% Set up a figure for multiple subplots
figure;
num_columns = length(categorical_columns);
rows = ceil(num_columns / 3); % Arrange in rows of 3
cols = 3; % Maximum 3 columns per row

% Loop through each categorical column and plot
for i = 1:num_columns
    subplot(rows, cols, i);
    % Convert numeric to categorical for better labeling
    histogram(categorical(data.(categorical_columns{i})), 'FaceColor', [0.2, 0.5, 0.8], 'EdgeColor', 'black');
    title(['Distribution of ', categorical_columns{i}]);
    xlabel(categorical_columns{i});
    ylabel('Count');
end

sgtitle('Distribution of Categorical Variables'); % Super title for all plots

%% Feature Importance Analysis
disp('Performing feature importance analysis...');
for i = 1:length(targets)
    target = targets{i};
    disp(['Calculating feature importance for ', target, '...']);
    
    % Define target variable
    Y = data.(target);
    
    % Train ensemble model for feature importance
    mdl = fitcensemble(X, Y);
    importance = predictorImportance(mdl);
    
    % Plot feature importance
    figure;
    bar(importance, 'FaceColor', [0.2, 0.5, 0.8]);
    title(['Feature Importance for ', target]);
    xlabel('Features');
    ylabel('Importance Score');
    xticks(1:length(features));
    xticklabels(features);
    xtickangle(45);
    grid on;
end

disp('Feature Importance Analysis Completed.');


%% Correlation Map Analysis for Vibration Sensors
% Selected features for correlation map
vibration_features = {'rpm', 'gaccx', 'gaccy', 'gaccz', 'haccx', 'haccy', 'haccz'};

% Extract the data for the selected features
vibration_data = table2array(data(:, vibration_features));

% Calculate the correlation matrix
correlation_matrix = corr(vibration_data, 'Type', 'Spearman');

% Plot the correlation heatmap
figure;
heatmap(vibration_features, vibration_features, correlation_matrix, ...
    'Colormap', jet, 'ColorLimits', [-1, 1], 'CellLabelColor', 'black');
title('Correlation Heatmap of Selected Variables (Vibration Sensors)');
xlabel('Variables');
ylabel('Variables');


%% Training and Testing Neural Network Models
results = struct();

for i = 1:length(targets)
    target = targets{i};
    disp(['Training model for ', target, '...']);
    
    % Extract target variable
    Y = data.(target);
    
    % Split data into training and testing sets
    cv = cvpartition(height(data), 'HoldOut', 0.2); % 80% train, 20% test
    X_train = X(training(cv), :);
    Y_train = Y(training(cv), :);
    X_test = X(test(cv), :);
    Y_test = Y(test(cv), :);
    
    % Define Neural Network
    net = feedforwardnet([10, 10]); % Two hidden layers
    net.trainParam.epochs = 1000; % Maximum epochs
    net.trainParam.lr = 0.005; % Learning rate
    net.divideParam.trainRatio = 0.8;
    net.divideParam.valRatio = 0.2;
    net.divideParam.testRatio = 0; % Test set is external
    
    % Train Neural Network
    [net, tr] = train(net, X_train', Y_train');
    
    % Test Neural Network
    predictions = net(X_test')';
    predictions = round(predictions); % Convert outputs to binary
    
    % Calculate Confusion Matrix
    confusionMat = confusionmat(Y_test, predictions);
    if size(confusionMat, 1) == 1
        if unique(Y_test) == 0
            classLabels = {'No Maintenance Needed'};
        else
            classLabels = {'Maintenance Needed'};
        end
    else
        classLabels = {'No Maintenance Needed', 'Maintenance Needed'};
    end
    
    % Calculate Accuracy
    Accuracy = trace(confusionMat) / sum(confusionMat(:)); % Correct predictions / total predictions
    disp(['Confusion Matrix for ', target, ':']);
    disp(confusionMat);
    disp(['Accuracy: ', num2str(Accuracy * 100), '%']);
    
    % Generate Confusion Matrix Figure
    figure;
    confusionchart(confusionMat, classLabels, ...
        'Title', ['Confusion Matrix for ', target]);
    
    % Store Results
    results.(target).accuracy = Accuracy;
    results.(target).confusionMatrix = confusionMat;
    results.(target).net = net;
end

%% Maintenance Prediction for New Data
disp('Predicting maintenance for new data...');
newData = [450, 1500, 30, 1.2, 300, 40, 75, 3, 45, 50, 2, 100, 1.5, 80, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3];
newData = normalize(newData, 'range'); % Normalize to match training

for i = 1:length(targets)
    target = targets{i};
    net = results.(target).net;
    prediction = net(newData');
    prediction = round(prediction); % Convert to binary output
    
    % Display prediction result
    if prediction == 1
        status = 'Maintenance Needed';
    else
        status = 'No Maintenance Needed';
    end
    disp([target, ': ', status]);
end