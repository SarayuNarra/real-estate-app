# RealtyAI: Smart Real Estate Insight Platform 🏠

RealtyAI is a **Streamlit** web application that predicts current real-estate prices and forecasts future price trends using machine learning and time-series models. It is designed as a practical tool for learning and showcasing end-to-end ML deployment with an interactive UI.

***

## Features

- 🔮 **Price prediction** for a single property using a trained regression pipeline (bagging regressor + preprocessing).  
- 📊 **Future price forecasting** for U.S. states using pre-trained **Prophet** time-series models.  
- 🖥️ **Interactive UI** with sidebar navigation and custom CSS styling for a modern look.  
- ⚙️ **Model loading from disk** using `joblib` with a compatibility patch for older scikit-learn pipelines.

***

## Tech Stack

- **Frontend / App:** Streamlit  
- **ML Model (tabular):** Scikit-learn Bagging Regressor + preprocessing pipeline  
- **Time Series Forecasting:** Prophet  
- **Model Serialization:** Joblib (`.pkl` files)  
- **Language:** Python 3.

***

## Project Structure

```text
.
├── app.py                     # Main Streamlit application
├── requirements.txt           # Python dependencies
├── models/
│   ├── bagging_regressor_model.pkl
│   ├── real_estate_pipeline.pkl
│   └── all_prophet_models.pkl
└── README.md
```

***

## Setup and Installation

1. **Clone the repository**

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

2. **Create and activate a virtual environment (optional)**

```bash
python -m venv venv
venv\Scripts\activate      # On Windows
source venv/bin/activate   # On macOS / Linux
```

3. **Create `requirements.txt`**

Example contents:

```txt
streamlit
pandas
scikit-learn
joblib
prophet
```

Install dependencies:

```bash
pip install -r requirements.txt
```

4. **Place model files**

Ensure these files are available (either in your Downloads folder or in `models/` with updated paths in `app.py`):

- `bagging_regressor_model.pkl`  
- `real_estate_pipeline.pkl`  
- `all_prophet_models.pkl`

***

## How to Run

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

***

## Usage

### 🏡 Home

- Overview of the platform and its capabilities.  
- Explains price prediction and trend forecasting in simple terms.

### 💰 Price Prediction

1. Go to **“💰 Price Prediction”** in the sidebar.  
2. Fill in:
   - `Square Feet`
   - `Bedrooms`
   - `Bathrooms`
   - `Location`
   - `Year Built`  
3. Click **“🔮 Predict Price”** to get the **estimated property price in ₹** from the ML pipeline.

### 📈 Forecast Future Prices

1. Go to **“📈 Forecast Future Prices”** in the sidebar.  
2. Select a **U.S. state** from the dropdown.  
3. Click **“📈 Show Forecast”**.  
4. The app uses the Prophet model for that state to forecast the next 12 months and plots the predicted trend (`yhat`) as a line chart.

***

## Model Details

### Price Prediction Model

- Uses a scikit-learn **pipeline** to preprocess features such as:
  - `sqft`, `bedrooms`, `bathrooms`, `location`, `year_built`
  - Additional engineered fields like `Total_Area`, `City`, etc.  
- The preprocessed data is passed into a **bagging regressor** to predict the property price.

### Time-Series Forecasting (Prophet)

- Each state has an individual **Prophet** model trained on its historical real-estate time series.  
- The app creates a future dataframe for 12 months (`freq="M"`) and predicts `yhat`, which is displayed as the expected price trend.

***

## Possible Improvements

- Replace Downloads-based paths with a `models/` folder inside the repo.  
- Add evaluation metrics (MAE, RMSE, R²) and dataset description.  
- Deploy on Streamlit Community Cloud, Render, or similar hosting.

***

## License

Add your preferred license here, for example:

```text
MIT License
```
