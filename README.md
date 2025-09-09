# NCR Ride Bookings Analysis & Revenue Prediction

## 🎓 Project Overview

**Research Question:**  
*"What factors influence ride success and revenue generation in NCR ride bookings?"*

This repository contains the **NCR Ride Bookings Analysis & Revenue Prediction** project. It analyzes ride booking data to uncover patterns in ride success, cancellations, and revenue, and builds a machine learning model to predict booking revenue from ride and temporal features. The project illustrates end-to-end data science workflows, including data cleaning, exploratory analysis, visualization, and machine learning.


## 📥 Dataset Source

The dataset analyzed in this project is publicly available on Kaggle:  
[https://www.kaggle.com/datasets/yashdevladdha/uber-ride-analytics-dashboard](https://www.kaggle.com/datasets/yashdevladdha/uber-ride-analytics-dashboard)

Thanks to @yashdevladdha for sharing this data on Kaggle!

## 📁 Project Files

```
ncr-ride-analysis/
├── Analysis.py             # Main data analysis and modeling script
├── Analysis.ipynb          # Detailed step-by-step Jupyter notebook walkthrough
├── ncr_ride_bookings.csv  # Dataset file (not included here)
├── README.md              # This documentation file
├── requirements.txt       # Python package dependencies (optional)
```

## ✨ Features

- Fast data loading and processing using **Polars**  
- Detailed **data cleaning** and missing value handling  
- Creation of time-based features and cancellation flags  
- Calculation of key business metrics such as ride success rate and revenue  
- Comprehensive **visualizations**:
  - Success rate breakdown by vehicle type, hour, and day of week  
  - Booking status distribution  
  - Revenue patterns by vehicle type and hour  
  - Revenue distribution and relationship to ride distance  
- **Machine Learning** model (Random Forest) for predicting ride revenue  
- Feature importance insights from the prediction model  


## ⚙️ Setup Instructions

### Prerequisites

- Python 3.7+ installed

### Installation

Install required Python packages from requirements.txt using the below command:
```
make install
```

### Run the analysis
```
make all
```

## 🚀 How to Run

Simply execute the analysis script:

```
make run
```

This will:

- Load and clean the data  
- Print summary statistics  
- Display interactive plots for key metrics and distributions  
- Train and evaluate a Random Forest regression model with output performance metrics and feature importance plots  


## 🔄 Analysis Workflow

1. **Data Loading & Cleaning**  
   - Handles null values related to cancellations naturally  
   - Removes duplicates  
   - Parses dates and times, deriving hour, weekday, month, and weekend flags  

2. **Business Metrics Computation**  
   - Success rate, total and average revenue, ratings, and cancellations counts  

3. **Data Visualization**  
   - Success rate and cancellation patterns by vehicle type, time, and day  
   - Revenue distribution and dynamics relative to vehicle type and ride distance  

4. **Revenue Prediction Modeling**  
   - Builds cyclical time features and encodes categorical location and vehicle data  
   - Trains Random Forest regressor and reports R² and RMSE  
   - Visualizes actual vs predicted revenue and feature importances  


## 🔍 Key Findings

- Missing values appearing in columns like bookings, ratings, and payment methods are expected and consistent with ride cancellations and booking outcomes. Related columns such as cancellation flags and reasons appear mutually exclusively, logically reflecting event occurrence and context.  

- **Total rides:** 150,000  
- **Successful rides:** 93,000 (62% success rate)  
- **Total revenue (successful rides):** ₹47,260,574  
- **Average ride value:** ₹508.18  
- **Average ride distance:** 26 km  
- **Average ratings:** Drivers 4.23/5, Customers 4.40/5  
- **Cancellations:** Customers 10,500, Drivers 27,000  

- Success rates are stable across vehicle types, times of day, and days of week. Uber XL, Bike, and Go Mini vehicles show marginally higher success rates.  

- Revenue analysis shows Auto generates the highest total revenue, with Go Mini and Go Sedan following. Uber XL yields the least revenue overall.  

- Average revenue is fairly constant over the hours in a day (₹500-₹520).  

- The revenue distribution is skewed toward many low-value rides, with few high-value bookings.  

- Revenue varies significantly even for rides of similar distances, suggesting trip distance alone is a poor predictor of booking value.  


## 📈 Visualizations

You will see the following plots during execution:

- Success Rate by Vehicle Type  
- Success Rate by Hour of Day  
- Success Rate by Day of Week  
- Booking Status Distribution (Pie Chart)  
- Total Revenue by Vehicle Type  
- Average Revenue by Hour  
- Revenue Distribution Histogram  
- Revenue vs Distance Scatter Plot  
- Actual vs Predicted Revenue Scatter Plot (ML Model)  
- Feature Importance Bar Chart (ML Model)  


## 📚 Additional Resources

For a **detailed step-by-step walkthrough** of the complete data analysis and modeling process — including more thorough explanations, exploratory data analysis, visualizations, and detailed interpretation of results — please refer to the **`Analysis.ipynb` Jupyter Notebook** provided alongside the script.

This notebook offers:

- Interactive exploration of the dataset  
- Clear, annotated code cells demonstrating all cleaning, feature engineering, and modeling steps  
- Inline plots and visualizations for immediate insight  
- Detailed commentary on findings and their implications  

If you want a guided and interactive experience exploring the data and results visually, working through the `Analysis.ipynb` notebook is highly recommended.

## 📜 Author

- Author: PRANSHUL BHATNAGAR  
- Date: 08 September 2025  

