
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import joblib

# Load data
data = pd.read_csv('timeseries_data.csv')
X = data[['feature']].values
y = data['value'].values

# Train a model
model = LinearRegression()
model.fit(X, y)

# Predict on new data
new_X = [[i] for i in range(len(data)+1, len(data)+4)]
predictions = model.predict(new_X)

from datetime import datetime

def parse_date(date_string):
    for fmt in ("%Y-%m-%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(date_string, fmt)
        except ValueError:
            pass
    raise ValueError(f"no valid date format found for {date_string}")

# Usage
date_string = "24-02-2023"
date_obj = parse_date(date_string)

# Save predictions
#new_dates = [(datetime.strptime(data['date'].iloc[-1], "%Y-%m-%d") + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 4)]
new_dates = [(parse_date(data['date'].iloc[-1]) + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 4)]


from datetime import datetime

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"predictions_{current_time}.csv"

# Rest of your code, then save the predictions:



prediction_df = pd.DataFrame({'date': new_dates, 'prediction': predictions})
prediction_df.to_csv('predictions.csv', index=False)
#prediction_df.to_csv(output_file, index=False)
