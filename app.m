data = readtable("Crop_Recommendation.csv", TextType="string");

% Extract features (Nitrogen, Phosphorus, etc.) and target (crop type)
features = data{:, ["Nitrogen", "Phosphorus", "Potassium", "Temperature", "Humidity", "pH_Value", "Rainfall"]};
target = data.Crop;
[cropNames, ~, ~] = unique(target);
cropNames = string(cropNames);
% Checking for missing data and summary of the table
any(isnan(features));
featuresTable = array2table(features);
% summary(featuresTable);
%{ 
 The data from the file has been checked and there are no missing data so
 there is no need to fill missing data.
%}

% Normalise data so that the result returned by the model is more accurate
features = normalize(features);

% Split data into training (80%) and test (20%) sets
cv = cvpartition(height(data), 'HoldOut', 0.2);
XTrain = features(training(cv), :);
YTrain = target(training(cv));
XTest = features(test(cv), :);
YTest = target(test(cv));

% Train a multi-class classification model using ECOC
% Use SVM as the binary classifier
template = templateSVM();
model = fitcecoc(XTrain, YTrain, 'Learners', template);


% Predict on the test data
predictions = predict(model, XTest);
% Calculate accuracy
accuracy = sum(predictions == YTest) / numel(YTest);
disp(['Accuracy: ', num2str(accuracy * 100), '%']);

% Display confusion matrix
confusionchart(YTest, string(predictions));


% Input from the user (Nitrogen, Phosphorus, etc.)
newData = [55, 40, 50, 28, 60, 6.5, 120];  % Sample new soil data
newData = normalize(newData);  % Normalize input if needed

% Get probabilities for each crop
[predictedLabel, scores] = predict(model, newData);

% Sort the scores to get the top N crops
[~, sortedIndices] = sort(scores, 'descend');

% Display the top 3 crops (for example)
topNCrops = sortedIndices(1:3);  % Adjust N as needed
disp('Top 3 recommended crops:');
for i = 1:length(topNCrops)
    disp(["Crop " + num2str(i) + ": " + cropNames(topNCrops(i))]);
end
