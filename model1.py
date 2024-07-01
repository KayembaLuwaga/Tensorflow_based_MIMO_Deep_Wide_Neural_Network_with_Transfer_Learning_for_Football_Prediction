import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

file_name = 'E2.csv'
# Define a function to convert time to text
def convert_time_to_text(value):
    return str(value)

# Specify the converters dictionary
converters = {'Time': convert_time_to_text}

# Read the CSV file with converters
data = pd.read_csv(file_name, converters=converters)

# Display the first few rows of your dataset to check the column names
print(data.head())
print(data.columns)
print(data.dtypes)
print(data.describe())
print(data.info())

# Define the feature columns and target columns
text_categorical_features = ["Time", "HoTe", "AwTe", "Ref"]
integer_target_columns = ["FTHG", "FTAG", "HTHG", "HTAG", "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR", "HBP", "ABP", "HTTG", "FTTG", "TS", "TST", "TC", "TF", "TY", "TR"]
y_columns = ['FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 'HBP', 'ABP', 'HTTG', 'FTTG', 'TS', 'TST', 'TC', 'TF', 'TY', 'TR']
text_categorical_target_columns = ["FTR", "HTR"]

# One-hot encode the 'Time' column
X_encoded_time = pd.get_dummies(data[['Time']])
X_encoded_time.columns = ['Time_' + str(col) for col in X_encoded_time.columns]  # Add a prefix to the column names
X_encoded_time.reset_index(drop=True, inplace=True)  # Reset the index for proper concatenation

# Combine one-hot-encoded 'Time' with other text-categorical features
X_encoded = pd.concat([data.drop(['Time'], axis=1), X_encoded_time, pd.get_dummies(data[text_categorical_features])], axis=1)
X_encoded = X_encoded.drop(integer_target_columns, axis=1)
X_encoded = X_encoded.drop(["HoTe", "AwTe", "Ref", "FTR", "HTR"], axis=1)
# Replace 'True' with 1 and 'False' with 0 in X_encoded
X_encoded = X_encoded.map(lambda x: 1 if x == True else (0 if x == False else x))

# One-hot encode the text-categorical target columns
y_encoded = pd.get_dummies(data[text_categorical_target_columns])
y_encoded_copy = y_encoded
y_encoded_copy = y_encoded_copy.map(lambda x: 1 if x == True else (0 if x == False else x))
print("One-hot encoded dummies for y_encoded: ")
print(y_encoded)

# Define the scaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Scale the input values of integer_target_columns
data[integer_target_columns] = scaler.fit_transform(data[integer_target_columns])
y_encoded = pd.concat([y_encoded, data[integer_target_columns]], axis=1)
# Replace 'True' with 1 and 'False' with 0 in y_encoded
y_encoded = y_encoded.map(lambda x: 1 if x == True else (0 if x == False else x))

# Combine one-hot-encoded targets with the original dataframe excluding text-categorical target columns
data_encoded = pd.concat([X_encoded, y_encoded], axis=1)

# Print the resulting dataframe
print("data_encoded: ")
print(data_encoded)

X = X_encoded
y = y_encoded
print("X: ")
print(X)
print("y: ")
print(y)

# Assuming X and y are your input and output dataframes
# Convert X and y to numpy arrays
X = X.values
y = y.values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Define the model
model = tf.keras.models.Sequential()

# Input layer
print("X.shape[1]: ")
print(X.shape[1])
model.add(tf.keras.layers.Input(shape=(X.shape[1],)))

# Hidden layers
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='relu'))

# Output layer
print("y.shape[1]: ")
print(y.shape[1])
model.add(tf.keras.layers.Dense(y.shape[1], activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
# model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.1)
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.1)

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

# Evaluate the model on the test set
y_pred = model.predict(X_test)

# Convert predicted probabilities to binary values
y_pred_binary = (y_pred > 0.5).astype(int)
y_pred_binary = y_pred
print("X_test: ")
print(X_test)
print("y_pred_binary: ")
print(y_pred_binary)

# Evaluate the model on the test set for each output column
mae_scores = []
for i in range(y.shape[1]):
    y_true_column = y_test[:, i]
    y_pred_column = y_pred[:, i]

    mae = mean_absolute_error(y_true_column, y_pred_column)
    mae_scores.append(mae)

# Calculate the mean MAE across all output columns
mean_mae = sum(mae_scores) / len(mae_scores)
print(f'Mean MAE: {mean_mae}')

# Define the new data
new_data = pd.DataFrame({'Time': ['15:00'], 'HoTe': ['Blackpool'], 'AwTe': ['Burton'], 'Ref': ['S Stockbridge']})

# One-hot encode the 'Time' column
new_data_encoded_time = pd.get_dummies(new_data[['Time']])
new_data_encoded_time.columns = ['Time_' + str(col) for col in new_data_encoded_time.columns]  # Add a prefix to the column names
new_data_encoded_time.reset_index(drop=True, inplace=True)  # Reset the index for proper concatenation

# Combine one-hot-encoded 'Time' with other text-categorical features
new_data_encoded = pd.concat([new_data.drop(['Time'], axis=1), new_data_encoded_time, pd.get_dummies(new_data[text_categorical_features])], axis=1)
new_data_encoded = new_data_encoded.reindex(columns=X_encoded.columns, fill_value=0)  # Reorder columns to match the order in X_train_df.columns
new_data_encoded = new_data_encoded.astype(np.float32)  # Convert NumPy array to float32
print(" new_data_encoded: ")
print(new_data_encoded)

# Make predictions on the new_data_encoded
prediction = model.predict(new_data_encoded)
print("prediction: ")
print(prediction)
print(prediction.shape)

# Inverse transform the one-hot encoded part of the prediction output
def inverse_transform_prediction(prediction, text_categorical_target_columns, y_encoded_copy, integer_target_columns, scaler):
    # Extract the one-hot encoded part of the prediction for target columns
    num_columns = len(y_encoded_copy.columns)
    y_pred_one_hot = prediction[:, :num_columns]

    # Inverse transform the scaled integer predictions
    y_pred_integer_scaled = prediction[:, num_columns:]
    y_pred_integer = scaler.inverse_transform(y_pred_integer_scaled)

    # Map integer values to text values for target columns using y_encoded_copy
    y_pred_text_target_categorical = [
        [
            y_encoded_copy.columns[i] for i in range(len(y_encoded_copy.columns) // 2) if row[i] == 1
        ] +
        [
            y_encoded_copy.columns[i] for i in range(len(y_encoded_copy.columns) // 2, len(y_encoded_copy.columns)) if
            row[i] == 1
        ]
        for row in np.round(y_pred_one_hot)
    ]

    # Extract the last character from 'FTR' and 'HTR' columns
    ftr_htr_cols = ['FTR', 'HTR']
    y_pred_text_target_categorical = [val[-1] if col in ftr_htr_cols else val for val, col in
                                       zip(y_pred_text_target_categorical, y_encoded_copy)]

    # Convert the predicted integer values to text values for integer_target_columns
    y_pred_text_target_integer = [
        [str(round(value)) for value in row]
        for row in y_pred_integer
    ]

    # Concatenate the categorical and integer predictions
    y_pred_text_target = [
        cat + intg for cat, intg in zip(y_pred_text_target_categorical, y_pred_text_target_integer)
    ]

    return y_pred_text_target

# Use the function to inverse transform the predictions
y_pred_text = inverse_transform_prediction(prediction, text_categorical_target_columns, y_encoded_copy, integer_target_columns, scaler)

# Display the inverse transformed predictions
print("Inputs for Prediction:")
print(new_data)
print("Inverse Transformed Text Predictions:")
print(y_pred_text)
