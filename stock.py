import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report

# Load the dataset
df = pd.read_csv('data_6.csv', parse_dates=['date'])
df.set_index('date', inplace=True)

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# Split the data into training and testing sets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[0:train_size, :]
test_data = scaled_data[train_size:len(scaled_data), :]

# Define the time steps and features
time_steps = 12
features = 1

# Create the training and testing data in a sequential format
def create_data(data):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i - time_steps:i, 0])
        y.append(data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], features))
    return X, y

X_train, y_train = create_data(train_data)
X_test, y_test = create_data(test_data)

# Define the BiLSTM model
model = Sequential()
model.add(Bidirectional(LSTM(50, return_sequences=True), input_shape=(time_steps, features)))
model.add(Bidirectional(LSTM(50)))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=4, batch_size=64)

# Evaluate the model
train_loss = model.evaluate(X_train, y_train)
test_loss = model.evaluate(X_test, y_test)
print('Train Loss:', train_loss)
print('Test Loss:', test_loss)

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform([y_train])
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform([y_test])

# Plot the predictions for the last 50 values
# Plot the predictions for the first 50 values
plt.figure(figsize=(12, 6))
x_values = np.arange(1, 366)
plt.subplot(2, 1, 1)
plt.plot(x_values, train_predict[:365], color='red', label='Predicted Train Stock')
plt.plot(x_values, y_train.flatten()[:365], color='blue', label='Actual Train Stock')
plt.title('Training Set Predictions vs Actual')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(x_values, test_predict[:365], color='green', label='Predicted Test Stock')
plt.plot(x_values, y_test.flatten()[:365], color='orange', label='Actual Test Stock')
plt.title('Testing Set Predictions vs Actual')
plt.legend()
plt.show()