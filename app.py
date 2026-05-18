from flask import Flask, render_template, request, redirect, url_for
import webbrowser
import pandas as pd
import numpy as np
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime

# Use non-interactive backend for server
matplotlib.use('Agg')

app = Flask(__name__)

# Load Pre-trained Model
model = load_model("bitcoin_price_prediction_model.h5")

# Helper Function to Convert Matplotlib Plots to HTML
def plot_to_html(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    buf.close()
    plt.close(fig)  # close the figure to release memory
    return f"data:image/png;base64,{data}"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        stock = request.form.get("stock")
        no_of_days = int(request.form.get("no_of_days"))
        return redirect(url_for("predict", stock=stock, no_of_days=no_of_days))
    # pass a title so base.html's {{ title }} is not empty
    return render_template("index.html", title="Crypto Price Predictor")

@app.route("/predict")
def predict():
    stock = request.args.get("stock", "BTC-USD")
    no_of_days = int(request.args.get("no_of_days", 10))

    # Fetch Stock Data (10 years)
    end = datetime.now()
    start = datetime(end.year - 10, end.month, end.day)
    stock_data = yf.download(stock, start, end)

    if stock_data.empty:
        return render_template("result.html", title="Results", error="Invalid stock ticker or no data available.")

    # Use Close price only
    close_df = stock_data[['Close']].copy()

    # Ensure there is enough data (we need at least 100 + 1 rows)
    if len(close_df) < 101:
        return render_template("result.html", title="Results", error="Not enough historical data to make predictions (need >= 101 rows).")

    # Fit scaler on the entire dataset (important!)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_all = scaler.fit_transform(close_df)  # shape = (n_rows, 1)

    # Create train/test split index (90% train, 10% test)
    splitting_len = int(len(scaled_all) * 0.9)

    # Prepare x_test and y_test sequences from scaled_all starting at splitting_len
    test_scaled = scaled_all[splitting_len:]  # scaled values for test portion

    x_test = []
    y_test = []
    for i in range(100, len(test_scaled)):
        x_test.append(test_scaled[i - 100:i])   # 100 timesteps
        y_test.append(test_scaled[i])           # next timestep

    x_test = np.array(x_test)  # shape: (num_samples, 100, 1)
    y_test = np.array(y_test)  # shape: (num_samples, 1)

    # If x_test is empty (shouldn't be because of length check) handle gracefully
    if x_test.size == 0:
        return render_template("result.html", title="Results", error="Not enough data to build test sequences.")

    # Ensure x_test shape matches model input: (samples, timesteps, features)
    # Optionally print shapes for debugging:
    # print("x_test.shape:", x_test.shape, "y_test.shape:", y_test.shape)

    # Predictions on test set
    predictions = model.predict(x_test)  # expected shape: (num_samples, 1) or (num_samples, 1, 1)
    # Normalize shapes: if model returned (num_samples, 1, 1), reshape to (num_samples, 1)
    if predictions.ndim == 3:
        predictions = predictions.reshape(predictions.shape[0], predictions.shape[-1])

    inv_predictions = scaler.inverse_transform(predictions)
    inv_y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Align indices for plotting: test starts at original index splitting_len + 100
    test_start_idx = splitting_len + 100
    test_index = stock_data.index[test_start_idx: test_start_idx + len(inv_y_test)]

    plotting_data = pd.DataFrame({
        'Original Test Data': inv_y_test.flatten(),
        'Predicted Test Data': inv_predictions.flatten()
    }, index=test_index)

    # Plot 1: Original Closing Prices (full series)
    fig1 = plt.figure(figsize=(12, 5))
    plt.plot(stock_data['Close'], label='Close Price')
    plt.title("Closing Prices Over Time")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    original_plot = plot_to_html(fig1)

    # Plot 2: Original vs Predicted Test Data
    fig2 = plt.figure(figsize=(12, 5))
    plt.plot(plotting_data['Original Test Data'], label="Original Test Data")
    plt.plot(plotting_data['Predicted Test Data'], label="Predicted Test Data", linestyle="--")
    plt.title("Original vs Predicted Closing Prices (Test Set)")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    predicted_plot = plot_to_html(fig2)

    # Plot 3: Future Predictions (recursive)
    # Prepare the last 100 scaled values from the entire scaled series
    last_100_scaled = scaled_all[-100:].reshape(1, 100, 1)  # shape (1,100,1)
    future_predictions = []

    for _ in range(no_of_days):
        next_scaled = model.predict(last_100_scaled)  # shape maybe (1,1) or (1,1,1)
        # Normalize shape to (1,1)
        if next_scaled.ndim == 3:
            next_scaled = next_scaled.reshape(1, -1)  # becomes (1,1)

        # Extract the scalar scaled value, inverse transform to original price
        next_value_inv = scaler.inverse_transform(next_scaled.reshape(-1, 1)).flatten()[0]
        future_predictions.append(next_value_inv)

        # Append the predicted scaled value to the end and slide window
        # next_scaled.reshape(1,1,1) will be appended to last_100_scaled (1,100,1) -> result (1,100,1)
        next_for_append = next_scaled.reshape(1, 1, 1)
        last_100_scaled = np.append(last_100_scaled[:, 1:, :], next_for_append, axis=1)

    future_predictions = np.array(future_predictions).flatten()

    fig3 = plt.figure(figsize=(12, 5))
    plt.plot(range(1, no_of_days + 1), future_predictions, marker='o', label="Predicted Future Prices")
    plt.title("Future Close Price Predictions")
    plt.xlabel("Days Ahead")
    plt.ylabel("Predicted Close Price")
    plt.grid(alpha=0.3)
    plt.legend()
    future_plot = plot_to_html(fig3)

    return render_template(
        "result.html",
        title=f"Results for {stock}",
        stock=stock,
        original_plot=original_plot,
        predicted_plot=predicted_plot,
        future_plot=future_plot,
        enumerate=enumerate,
        future_predictions=future_predictions
    )

if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:5000/")
    app.run(debug=True)
