# Tensorflow_based_MIMO_Deep_Wide_Neural_Network_with_Transfer_Learning_for_Football_Prediction
# Abstract
This report delves into the real-world application of Machine Learning and Deep Learning techniques in the domain of sports prediction, specifically focusing on the implementation of a Deep-Wide Neural Network Multiple-Input Multiple-Output (MIMO) model compatible with both numerical and categorical data features and targets. The purpose of this study is to provide insights into the real-world capabilities and limitations of such a predictive model, emphasizing its educational scope.

# Audience
This study is tailored for audiences with a keen interest in the real-world application of deep learning techniques in sports prediction, machine learning enthusiasts, and educators and researchers in the field. A basic understanding of machine learning and deep learning concepts would be beneficial for comprehending the findings discoursed herein.

# 1. Introduction
In recent years, the intersection of deep learning and sports analytics has gained significant attention, offering a novel approach to predicting sports outcomes. The backend code presented herein implements a deep-wide neural network for football prediction using TensorFlow. The model is designed to handle multi-input, multi-output (MIMO) data of both numerical and categorical-text features and targets, incorporating both deep and wide neural network components. The wide part is constructed through one-hot encoding, while the deep part consists of several dense layers, the numbers and dimensions of which the interested audience may consider modifying to adjust the prediction accuracy of the model among other potential model-tuning endeavors. The code also includes transfer learning by saving and reloading the trained model.

## 1.1. Model Architecture
The Deep-Wide Neural Network MIMO model integrates deep neural networks with a wide component, allowing for the extraction of intricate features and capturing broader patterns. This architecture enhances the model's predictive capabilities in the context of sports outcomes, as illustrated in drawing 1.

# 2. Methodology
The study employed a comprehensive dataset comprising historical sports data, encompassing various parameters such as player statistics, team performance, and environmental factors. The Deep-Wide Neural Network MIMO model was implemented and trained using this dataset as holistically discoursed in the Appendix and illustrated in the steps below.

## 2.1. Data Preprocessing and Feature Engineering
The dataset is loaded from a CSV file ('E2.csv') using pandas. The 'Time' column is converted to text, and the converters dictionary is used during reading. Various exploratory data analysis (EDA) steps are performed, including displaying the first few rows, checking column names, data types, descriptive statistics, and dataset information. Text-categorical features ('Time', 'HoTe', 'AwTe', 'Ref') are one-hot encoded, and the resulting columns are added to the dataset. The 'integer_target_columns' are scaled using Min-Max scaling, and the 'True' and 'False' values are replaced with 1 and 0. The one-hot encoding is also applied to the text-categorical target columns ('FTR', 'HTR') as shown below.

```python
# ...Disclaimer: (omitting some code for brevity, demonstration)
file_name = 'E2.csv'
data = pd.read_csv(file_name, converters=converters)
text_categorical_features = ["Time", "HoTe", "AwTe", "Ref"]
integer_target_columns = ["FTHG", "FTAG", "HTHG", "HTAG", "HS", ...]
y_columns = ['FTHG', 'FTAG', 'FTR', 'HTHG', ...]
text_categorical_target_columns = ["FTR", "HTR"]

# One-hot encode the 'Time' column
X_encoded_time = pd.get_dummies(data[['Time']])
# Combine one-hot-encoded 'Time' with other text-categorical features
X_encoded = pd.concat([...], axis=1)
# Replace 'True' with 1 and 'False' with 0 in X_encoded
X_encoded = X_encoded.map(lambda x: 1 if x == True else (0 if x == False else x))
# One-hot encode the text-categorical target columns
y_encoded = pd.get_dummies(data[text_categorical_target_columns])
# Define the scaler
scaler = MinMaxScaler(feature_range=(0, 1))
# Scale the input values of integer_target_columns
data[integer_target_columns] = scaler.fit_transform(data[integer_target_columns])
# Combine one-hot-encoded targets with the original dataframe excluding text-categorical target columns
data_encoded = pd.concat([...], axis=1)
```

## 2.2. Model Definition and Training
The neural network model is defined using TensorFlow's Sequential API. It comprises an input layer, hidden layers with relu activation, and an output layer with sigmoid activation for binary classification. The model is compiled with the Adam optimizer and binary cross-entropy loss. It is then trained on the training set for 50 epochs with a batch size of 16. The training history is saved for visualization as shown below.

```python
# ...Disclaimer: (omitting some code for brevity, demonstration)
# Define the model
model = tf.keras.models.Sequential()
# Input layer
model.add(tf.keras.layers.Input(shape=(X.shape[1],)))

# Hidden layers
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(16, activation='relu'))
# Output layer
model.add(tf.keras.layers.Dense(y.shape[1], activation='sigmoid'))
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.1)
```

## 2.3. Model Evaluation
The model is evaluated on the test set, and predictions are obtained. The predicted probabilities are thresholded at 0.5 to convert them into binary values. Mean Absolute Error (MAE) is calculated for each output column and then averaged to get the mean MAE. The training accuracy and validation loss are plotted over epochs to visualize the model's performance during training as shown below.

```python
# ...Disclaimer: (omitting some code for brevity, demonstration)
# Evaluate the model on the test set
y_pred = model.predict(X_test)
# Decode one-hot encoded predictions and ground truth
y_pred_decoded = pd.DataFrame(scaler.inverse_transform(y_pred), columns=integer_target_columns)
y_test_decoded = pd.DataFrame(scaler.inverse_transform(y_test), columns=integer_target_columns)
# Evaluate performance using mean absolute error
mae = mean_absolute_error(y_test_decoded, y_pred_decoded)
print(f'Mean Absolute Error: {mae}')
```

## 2.4. Model Transfer Learning and Prediction
The trained model is saved to a file ('my_model.h5') and reloaded for making predictions on new data. A sample new dataset is created, preprocessed, and used for prediction. The one-hot encoded part of the prediction output is inverse-transformed, mapping integer values to text values for target columns as shown below.

```python
# ...Disclaimer: (omitting some code for brevity, demonstration)
# Transfer Learning and Predicting
# Save the model
model.save('my_model.h5')
# Access training history
training_accuracy = history.history['accuracy']
validation_loss = history.history['val_loss']

# Plot training accuracy
plt.plot(training_accuracy, label='Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training Accuracy over Epochs')
plt.legend()
plt.show()

# Plot validation loss
plt.plot(validation_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Validation Loss over Epochs')
plt.legend()
plt.show()

# Make predictions on new data
new_data = pd.read_csv('new_data.csv', converters=converters)
# Apply the same preprocessing steps
# ...
# Make predictions
new_data_predictions = model.predict(new_data)

# Sample output of the prediction
# Inputs for Prediction:
# Time     HoTe         AwTe       Ref
# 0 15:00  Blackpool   Burton    S Stockbridge

# Inverse Transformed Text Predictions:
# [['FTR_H', 'HTR_H', '2', '0', '2', '0', '15', '9', '6', '2', '10', '11', '5', '4', '1', '2', '0', '0', '13', '17', '2', '3', '24', '9', '10', '21', '3', '0']]
```

# 3. Results and Discussion
As shown in figure 1 below, the model demonstrated a ceiling prediction accuracy of 65%, showcasing its potential real-world capabilities while emphasizing its educational nature. However, deeper analysis of results demonstrates the contingent nature of the challenges involved in achieving higher accuracy rates. The transfer learning process demonstrates the ability to save and reload the model for future predictions and the deployment of the model in minimal Python/Anaconda runtime environments such as mobile applications and lightweight web-based applications. The inverse transformation of predictions allows interpreting the model outputs in the higher-language level original format.

#4. Conclusion
The presented Deep Learning code implements a deep-wide neural network for football prediction, showcasing data preprocessing, feature engineering, model training, evaluation, transfer learning, and prediction. Further optimization and fine-tuning can be explored based on the observed performance so as to transcend its capped prediction accuracy beyond 65%. The transfer learning aspect adds flexibility for deploying the model in real-world scenarios such as lightweight mobile and or web-based applications.

# Disclaimer
This study was conducted only for educational purposes. The author(s) explicitly disclaim any responsibility for any illegitimate application of the Deep-Wide Neural Network MIMO model for sports prediction in any locality. The model has intentionally been sub-optimized, limiting its prediction accuracy to a ceiling of 65%. If the audience chooses to use the ML/DL model discoursed herein for sports prediction, they should be fully aware of the author's noninvolvement and understand the model's capabilities/limitations. The author does not discourage or encourage the application of the model for any commercial or consequential use. Users are urged to exercise caution and responsibility in interpreting and applying the findings of this study.
