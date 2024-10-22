import numpy as np

# Calculate the Mean Squared Error (MSE) between the actual test values and the forecasted values
def rmsse(train, test, forecast):

    # Forecast MSE error
    forecast_mse = np.mean((test - forecast) ** 2, axis=0)

    # In-sample variation or error, normalizing for scale
    train_mse = np.mean((np.diff(np.trim_zeros(train)) ** 2))

    # Return the Root Mean Squared Scaled Error (RMSSE)
    # RMSSE = sqrt(MSE of forecast / MSE of training)
    return np.sqrt(forecast_mse / train_mse)

