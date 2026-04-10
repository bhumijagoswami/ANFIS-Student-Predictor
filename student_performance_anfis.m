% =========================================================================
% Hybrid Intelligent System: Student Performance Predictor (ANFIS)
% Inputs: Attendance, Assignment Marks, Test Marks
% Output: Performance Level (Mapped to: Poor, Average, Good)
% =========================================================================

clc; clear; close all;

disp('Generating synthetic student data for neural network training...');

% 1. Generate Synthetic Training Data
% We simulate 300 students to train the Neural Network
numStudents = 300;
attendance = randi([0, 100], numStudents, 1);
assignments = randi([0, 100], numStudents, 1);
tests = randi([0, 100], numStudents, 1);

% Define true performance logic (Tests weigh heaviest, then assignments)
% Adding some random noise to simulate real-world variance
actualPerformance = (0.2 * attendance) + (0.3 * assignments) + (0.5 * tests) + (randn(numStudents,1)*3);
actualPerformance = max(0, min(100, actualPerformance)); % Bound 0-100

% Combine into a dataset [Input1, Input2, Input3, Output]
studentData = [attendance, assignments, tests, actualPerformance];

% Split data: 80% for training, 20% for testing
numTrain = round(0.8 * numStudents);
trainData = studentData(1:numTrain, :);
testData = studentData(numTrain+1:end, :);

% =========================================================================
% 2. Setup the Fuzzy Inference System (FIS) Architecture
% =========================================================================
disp('Setting up the initial Fuzzy System...');

% Create Grid Partitioning for the inputs (3 inputs, 3 MFs each: Low/Poor, Med/Avg, High/Good)
opt = genfisOptions('GridPartition');
opt.NumMembershipFunctions = [3 3 3]; 
opt.InputMembershipFunctionType = ["gaussmf" "gaussmf" "gaussmf"];

% Generate the initial un-trained FIS
initialFIS = genfis(trainData(:,1:3), trainData(:,4), opt);

% =========================================================================
% 3. Train the Hybrid System (Neural Network Learning)
% =========================================================================
disp('Training the Neuro-Fuzzy System (ANFIS)... Please wait.');

% Configure training options (50 epochs/iterations)
trainOpt = anfisOptions('InitialFIS', initialFIS, 'EpochNumber', 50, 'DisplayANFISInformation', 0, 'DisplayErrorValues', 0);

% Train the FIS using hybrid learning (backpropagation + least squares)
[trainedFIS, trainError] = anfis(trainData, trainOpt);

disp('Training Complete!');

% =========================================================================
% 4. Evaluate and Plot Results (For Screenshots)
% =========================================================================

% --- FIGURE 1: Training Error ---
figure('Name', 'Neural Network Training Error', 'NumberTitle', 'off');
plot(trainError, 'LineWidth', 2, 'Color', 'b');
title('ANFIS Training Error over Epochs');
xlabel('Epoch (Iterations)');
ylabel('Root Mean Square Error (RMSE)');
grid on;

% --- FIGURE 2: Prediction Accuracy ---
% Test the trained system on the unseen testing data
predictedPerformance = evalfis(trainedFIS, testData(:,1:3));

figure('Name', 'Actual vs Predicted Performance', 'NumberTitle', 'off');
plot(testData(:,4), '-o', 'LineWidth', 1.5, 'DisplayName', 'Actual Student Performance');
hold on;
plot(predictedPerformance, '-x', 'LineWidth', 1.5, 'DisplayName', 'ANFIS Predicted Performance');
title('Hybrid System: Actual vs. Predicted Performance Level');
xlabel('Student Sample (Testing Data)');
ylabel('Performance Level (0-100)');

% Map continuous output to Linguistic Variables on the Y-Axis
yline(40, '--r', 'Poor Threshold', 'LabelHorizontalAlignment', 'left');
yline(75, '--g', 'Good Threshold', 'LabelHorizontalAlignment', 'left');
legend('Location', 'best');
grid on; hold off;

% --- FIGURE 3: Neuro-Fuzzy Surface ---
% Shows the learned complex relationship between Tests, Assignments and Output
figure('Name', 'Neuro-Fuzzy Surface Map', 'NumberTitle', 'off');
gensurf(trainedFIS, [2 3], 1); % Inputs 2(Assignments) and 3(Tests) vs Output
title('Learned Surface: Test Marks & Assignments vs. Performance');
zlabel('Performance Level');