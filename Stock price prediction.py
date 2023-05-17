#!/usr/bin/env python
# coding: utf-8

# In[48]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Load the data
data = pd.read_csv("AAPL.csv")

# Step 2: Preprocess the data (assuming "Date" column is already in datetime format)
data.set_index("Date", inplace=True)

# Step 3: Train and test split
train_data, test_data = train_test_split(data, test_size=0.7, shuffle=False)

# Step 4: Prepare the training data
X_train = np.array(range(len(train_data))).reshape(-1, 1)
y_train = train_data["Stock Price"].values

# Step 5: Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Prepare the test data
X_test = np.array(range(len(train_data), len(data))).reshape(-1, 1)
y_test = test_data["Stock Price"].values

# Step 7: Predict the training data
y_pred_train = model.predict(X_train)

# Step 8: Predict the test data
y_pred_test = model.predict(X_test)


# Display the model accuracy
y_pred_train = model.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
print("Training RMSE:", train_rmse)



# In[49]:


# Step 9: Plot the training data prediction
plt.figure(figsize=(12, 6))
plt.plot(train_data.index, y_train, label="Actual")
plt.plot(train_data.index, y_pred_train, label="Predicted")
plt.title("Training Data - Actual vs Predicted")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()
plt.show()


# In[ ]:




