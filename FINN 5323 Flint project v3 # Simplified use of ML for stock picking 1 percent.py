

# Simplified use of ML for stock picking v2

# Import packages and data
# %%
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Download historical data for a stock (e.g., Apple Inc.)
data = yf.download('NVDA', start='2020-01-01', end='2023-01-01')

# Fetch real-time data for the last 30 days
today = datetime.now().strftime('%Y-%m-%d')
thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
real_time_data = yf.download('NVDA', start=thirty_days_ago, end=today)

# Append the new real-time data to the existing dataframe
data = pd.concat([data, real_time_data], axis=0)

# Calculate 1% increase target
look_forward_days = 10
data['Future Price'] = data['Close'].shift(-look_forward_days)
data['Target'] = (data['Future Price'] > data['Close'] * 1.01).astype(int)

# Drop rows with NaN values
data.dropna(inplace=True)

# Import model
# %%
features = ['Open', 'High', 'Low', 'Close', 'Volume']
X = data[features]
y = data['Target']

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions for the next 10 days
latest_data = data.tail(10)  # Assuming you want to make predictions for the latest available data
X_latest = latest_data[features]
latest_predictions = model.predict(X_latest)

# Print out predictions
print("Predictions for the next 10 days:")
print(latest_predictions)

# Validating the model with K Fold Cross Validation
# %%
scores = cross_val_score(model, X, y, cv=5)
print("Cross-validated scores:", scores)

# Visualize
# %%
plt.plot(scores)
plt.title('Model Accuracy Across 5-Fold Cross-Validation')
plt.ylabel('Accuracy')
plt.xlabel('Fold')
plt.show()

# Data elements importance and analysis
# %%
feature_importance = np.abs(model.coef_[0])
features = np.array(features)
indices = np.argsort(feature_importance)

plt.title('Feature Importances')
plt.barh(range(len(indices)), feature_importance[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

# Model tuning
# %%
param_grid = {'C': [0.1, 1, 10, 100], 'solver': ['lbfgs', 'liblinear']}
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)


# %%
