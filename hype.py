import csv
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random

# Load CSV manually
file_path = "stock_data.csv"  # Update path if needed
data = []

with open(file_path, "r") as file:
    reader = csv.reader(file)
    headers = next(reader)  # Extract column names
    for row in reader:
        data.append(row)

# Detect and convert date format
def parse_date(date_str):
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):  # Try different date formats
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            pass
    raise ValueError(f"Date format not recognized: {date_str}")

# Convert data into usable format
dates = [parse_date(row[0]) for row in data]  # Extract dates
values = {col: [float(row[i]) for row in data] for i, col in enumerate(headers[1:], start=1)}

# Streamlit Sidebar - Select target column
st.sidebar.header("Stock Prediction Settings")
target_column = st.sidebar.selectbox("Select Target Stock", ["AMZN", "DPZ", "BTC", "NFLX"])

# Use only DPZ for regression
X = values["DPZ"]
y = values[target_column]

# Split data (80% train, 20% validation)
split_idx = int(0.8 * len(y))
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

# Perform Hyperparameter Tuning (Grid Search for Best Parameters)
best_b0, best_b1 = None, None
best_mse = float("inf")

def simple_linear_regression(x, y):
    """Implements Simple Linear Regression manually."""
    n = len(x)
    mean_x, mean_y = sum(x) / n, sum(y) / n
    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    denominator = sum((x[i] - mean_x) ** 2 for i in range(n))
    b1 = numerator / denominator
    b0 = mean_y - b1 * mean_x
    return b0, b1

# Try different random subsets of training data (Hyperparameter Search)
for _ in range(10):  # Try 10 different models with varied subsets
    random_idx = random.sample(range(len(X_train)), int(0.8 * len(X_train)))
    X_sub = [X_train[i] for i in random_idx]
    y_sub = [y_train[i] for i in random_idx]
    
    b0, b1 = simple_linear_regression(X_sub, y_sub)
    
    val_predictions = [b0 + b1 * x for x in X_val]
    mse = sum((y_val[i] - val_predictions[i]) ** 2 for i in range(len(y_val))) / len(y_val)
    
    if mse < best_mse:
        best_mse = mse
        best_b0, best_b1 = b0, b1

st.write(f"Optimized Validation Mean Squared Error: {best_mse:.2f}")

# Predict Future Stock Prices
future_days = st.sidebar.slider("Select Prediction Days (1-30)", 1, 30, 7)
last_date = dates[-1]
future_dates = [last_date + timedelta(days=i) for i in range(1, future_days + 1)]
future_x = X[-future_days:]  # Use last known DPZ values for prediction
future_predictions = [best_b0 + best_b1 * x for x in future_x]

# Display Predictions
st.header(f"Stock Price Prediction for {target_column}")
st.write(f"Forecasting the next {future_days} days")

st.table({"Date": [d.strftime('%Y-%m-%d') for d in future_dates], "Predicted Price": future_predictions})

# Historical Data
st.subheader("Stock Price Chart")
years_back = 3  # Change manually as needed
start_date = last_date - timedelta(days=365 * years_back)
historical_data = [(dates[i], values[target_column][i]) for i in range(len(dates)) if dates[i] >= start_date]

# Plot Data
st.subheader("Stock Price Chart")
fig, ax = plt.subplots(figsize=(10, 5))

# Set Black Background
fig.patch.set_facecolor('black')  # Background color outside the plot
ax.set_facecolor('black')  # Background color inside the plot

# Plot Historical Data
historical_dates = [d for d, _ in historical_data]
historical_prices = [p for _, p in historical_data]
ax.plot(historical_dates, historical_prices, label="Historical Data", color="blue", linewidth=2)

# Plot Future Predictions
ax.plot(future_dates, future_predictions, linestyle="dashed", marker="o", label="Future Prediction", color="red", linewidth=2)

# Customize Graph
ax.set_title(f"{target_column} Stock Price Prediction (Next {future_days} Days)", fontsize=14, fontweight='bold', color="white")
ax.set_xlabel("Date", fontsize=12, color="white")
ax.set_ylabel("Price", fontsize=12, color="white")
ax.legend(facecolor="black", edgecolor="white", labelcolor="white")
plt.xticks(rotation=45, color="white")
plt.yticks(color="white")

# Show Graph in Streamlit
st.pyplot(fig)

st.success("Prediction Completed! ✅")
