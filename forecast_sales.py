import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime

# Data loading and cleaning
def load_and_clean_data(file_path):
    try:
        # Load the dataset
        df = pd.read_excel(file_path)
        print("Data loaded successfully.")
        
        # Check for missing columns
        required_columns = ['Invoice', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'Price', 'Customer ID', 'Country']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing columns: {missing_columns}")
            return None
        
        print(f"Columns in the dataset: {df.columns}")
        
        # Clean the data: drop rows where 'Customer ID' or 'Quantity' are missing
        df.dropna(subset=['Customer ID', 'Quantity'], inplace=True)
        print("Data cleaned successfully.")
        
        return df
    
    except Exception as e:
        print(f"Error loading or cleaning data: {e}")
        return None

# Data preparation for ARIMA
def prepare_data_for_arima(df):
    try:
        # Convert 'InvoiceDate' to datetime
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        
        # Extract Year-Month from 'InvoiceDate'
        df['YearMonth'] = df['InvoiceDate'].dt.to_period('M')
        
        # Group data by 'YearMonth' to get monthly sales
        monthly_sales = df.groupby('YearMonth').agg({'Quantity': 'sum', 'Price': 'mean'}).reset_index()
        
        # Create a new column for total sales: Quantity * Price
        monthly_sales['TotalSales'] = monthly_sales['Quantity'] * monthly_sales['Price']
        
        # Prepare data for ARIMA
        sales_data = monthly_sales[['YearMonth', 'TotalSales']].rename(columns={'YearMonth': 'ds', 'TotalSales': 'y'})
        
        # Convert 'ds' from Period type to Timestamp type
        sales_data['ds'] = sales_data['ds'].dt.to_timestamp()
        
        # Check data structure for ARIMA
        print("Data prepared for ARIMA:")
        print(sales_data.head())
        
        return sales_data
    
    except Exception as e:
        print(f"Error preparing data for ARIMA: {e}")
        return None

# Sales forecasting using ARIMA
def forecast_sales_arima(monthly_sales):
    try:
        # Fit ARIMA model (p=1, d=1, q=1) - You can adjust these parameters
        model = ARIMA(monthly_sales['y'], order=(1, 1, 1))
        model_fit = model.fit()
        
        # Forecast the next 12 months
        forecast = model_fit.forecast(steps=12)
        
        # Plot the forecast
        plt.plot(monthly_sales['ds'], monthly_sales['y'], label='Historical Data')
        forecast_dates = pd.date_range(start=monthly_sales['ds'].iloc[-1], periods=13, freq='M')[1:]
        plt.plot(forecast_dates, forecast, label='Forecasted Data', color='red')
        
        plt.title('Sales Forecast for the Next 12 Months (ARIMA)')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.show()
        
        return forecast
    except Exception as e:
        print(f"Error forecasting sales using ARIMA: {e}")
        return None

# Main function to run the entire process
def main():
    file_path = "C:\\Users\\thiru\\Documents\\online_retail_II.xlsx"  # Modify with your actual file path
    df = load_and_clean_data(file_path)
    
    if df is not None:
        sales_data = prepare_data_for_arima(df)
        
        if sales_data is not None:
            forecast_result = forecast_sales_arima(sales_data)
            if forecast_result is not None:
                print("Sales forecasting completed successfully.")
            else:
                print("Sales forecasting failed.")
        else:
            print("Data preparation for ARIMA failed.")
    else:
        print("Data loading or cleaning failed.")

if __name__ == "__main__":
    main()
