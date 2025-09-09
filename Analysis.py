
import os
import sys
import logging
import warnings

import polars as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# Set global plot styles
plt.style.use('default')
sns.set_palette("husl")

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

def load_data(filepath):
    """
    Load the ride bookings dataset into a Polars DataFrame.
    Args:
        filepath (str): Path to the CSV file containing the dataset.

    Returns:
        pl.DataFrame: A Polars DataFrame with the loaded data.
    """

    if not os.path.isfile(filepath):
        logging.error(f"File not found: {filepath}")
        sys.exit(1)
    df = pl.read_csv(filepath, null_values=["", " ", "NA", "N/A", "NULL", "null", "?"])
    logging.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def check_duplicates(df):
    """
    Check for duplicate rows in a Polars DataFrame and remove them.
    Args:
        df (pl.DataFrame): Input Polars DataFrame to check for duplicates.

    Returns:
        pl.DataFrame: DataFrame with duplicate rows removed.
    """

    dup_count = df.is_duplicated().sum()
    logging.info("Checking for duplicates...")
    print(f"Duplicate rows: {dup_count}")
    df = df.unique(maintain_order=True)
    logging.info("Removing duplicates...")
    print(f"After deleting duplication rows: {df.shape[0]} rows")
    return df

def parse_datetime(df):
    """
    Parse and combine separate 'Date' and 'Time' columns into a single datetime column.
    Args:
        df (pl.DataFrame): Input Polars DataFrame containing 'Date' and 'Time' columns.

    Returns:
        pl.DataFrame: DataFrame with a new 'DateTime' column of type `Datetime`.
    """

    logging.info("Parsing Date Time Columns...")
    df = df.with_columns([
        pl.col("Date").alias("Date_str"),
        (pl.col("Date").alias("Date_str") + " " + pl.col("Time"))
        .str.strptime(pl.Datetime, "%d/%m/%y %H:%M:%S", strict=False)
        .alias("DateTime")
    ])
    df = df.drop("Date_str")
    return df

def add_time_features(df):
    """
    Extract and add time-based features from the 'DateTime' column.
    Args:
        df (pl.DataFrame): Input Polars DataFrame with a 'DateTime' column.

    Returns:
        pl.DataFrame: DataFrame with new time-related columns:
            - 'Hour' (int): Hour of the day (0–23).
            - 'Month' (int): Month of the year (1–12).
            - 'IsWeekend' (bool): True if Saturday or Sunday, otherwise False.
            - 'DayOfWeek' (str): Name of the day (e.g., "Monday").
    """

    logging.info("Extracting Date Time Features...")
    df = df.with_columns([
        pl.col("DateTime").dt.hour().alias("Hour"),
        pl.col("DateTime").dt.weekday().alias("DayOfWeekNum"),
        pl.col("DateTime").dt.month().alias("Month"),
        (pl.col("DateTime").dt.weekday() >= 5).alias("IsWeekend")
    ])
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    week_cols = [weekdays[x] if x is not None and 0 <= x < 7 else None for x in df["DayOfWeekNum"]]
    df = df.with_columns([
        pl.Series("DayOfWeek", week_cols)
    ]).drop("DayOfWeekNum")
    return df

def clean_numeric(df):
    """
    Clean and standardize numeric columns by casting them to float type.
    Args:
        df (pl.DataFrame): Input Polars DataFrame.

    Returns:
        pl.DataFrame: DataFrame with specified numeric columns cast to Float64.
    """

    logging.info("Cleaning Numeric Columns...")
    numeric_cols = ["Booking Value", "Ride Distance", "Driver Ratings", "Customer Rating"]
    for col in numeric_cols:
        if col in df.columns:
            df = df.with_columns([
                pl.col(col).cast(pl.Float64)
            ])
    return df

def create_flags(df):
    """
    Generate boolean flags and status categories from booking information.
    Args:
        df (pl.DataFrame): Input Polars DataFrame with booking-related columns.

    Returns:
        pl.DataFrame: DataFrame with new columns:
            - 'Is_Successful' (bool): True if booking was completed.
            - 'Is_Cancelled_Customer' (bool): True if cancelled by customer.
            - 'Is_Cancelled_Driver' (bool): True if cancelled by driver.
            - 'Status_Category' (str): One of {"Completed", "Cancelled", "No Driver Found", "Other"}.
    """
     
    logging.info("Creating Flags and Status Categories...")
    df = df.with_columns([
        (pl.col("Booking Status") == "Completed").alias("Is_Successful"),
        pl.col("Cancelled Rides by Customer").is_not_null().alias("Is_Cancelled_Customer"),
        pl.col("Cancelled Rides by Driver").is_not_null().alias("Is_Cancelled_Driver")
    ])
    df = df.with_columns([
        pl.when(pl.col("Booking Status") == "Completed").then(pl.lit("Completed"))
        .when(pl.col("Booking Status").str.contains("Cancelled")).then(pl.lit("Cancelled"))
        .when(pl.col("Booking Status") == "No Driver Found").then(pl.lit("No Driver Found"))
        .otherwise(pl.lit("Other"))
        .alias("Status_Category")
    ])
    return df

def basic_metrics(df):
    """
    Compute and display key business metrics from ride booking data.
    Args:
        df (pl.DataFrame): Input Polars DataFrame containing booking information 
                           and derived flags such as 'Is_Successful', 
                           'Is_Cancelled_Customer', and 'Is_Cancelled_Driver'.

    Returns:
        None: Prints metrics to the console.
    """

    logging.info("Calculating Useful Business Metrics...")
    total_rides = df.height
    successful_rides = df.filter(pl.col("Is_Successful")).height
    print(f"-> Total rides: {total_rides}")
    print(f"-> Successful rides: {successful_rides} ({successful_rides*100/total_rides:.1f}%)")
    successful_df = df.filter(pl.col("Is_Successful"))
    print(f"-> Revenue: ₹{successful_df['Booking Value'].sum():,.2f}")
    print(f"-> Average ride value: ₹{successful_df['Booking Value'].mean():.2f}")
    print(f"-> Average distance: {successful_df['Ride Distance'].mean():.2f} km")
    print(f"-> Customer cancellations: {df['Is_Cancelled_Customer'].sum()}")
    print(f"-> Driver cancellations: {df['Is_Cancelled_Driver'].sum()}")

def plot_success_patterns(df):
    """
    Visualize booking success patterns across multiple dimensions.
    Args:
        df (pl.DataFrame): Input Polars DataFrame with derived features such as
                           'Is_Successful', 'Hour', 'DayOfWeek', 'Vehicle Type',
                           and 'Status_Category'.

    Returns:
        None: Displays a matplotlib figure with four subplots.

    Subplots Generated:
        - Success Rate by Vehicle Type (bar chart).
        - Success Rate by Hour of Day (line plot).
        - Success Rate by Day of Week (bar chart).
        - Overall Booking Status Distribution (pie chart).
    """

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Success rate by Vehicle Type
    vehicle_types = df["Vehicle Type"].unique()
    vehicle_success_data = []
    for vt in vehicle_types:
        vt_df = df.filter(pl.col("Vehicle Type") == vt)
        total = vt_df.height
        success = vt_df.filter(pl.col("Is_Successful")).height
        rate = success / total if total > 0 else 0
        vehicle_success_data.append((vt, total, success, rate))
    vehicle_success = pd.DataFrame(vehicle_success_data, columns=["Vehicle Type", "Total_Rides", "Successful_Rides", "Success_Rate"]).sort_values("Success_Rate", ascending=False)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    axes[0, 0].bar(vehicle_success["Vehicle Type"], vehicle_success["Success_Rate"] * 100, color= colors)
    axes[0, 0].set_title("Success Rate by Vehicle Type")
    axes[0, 0].set_ylabel("Success Rate (%)")
    axes[0, 0].tick_params(axis="x", rotation=45)

    # Success rate by Hour
    hours = sorted(df["Hour"].unique())
    hourly_success_data = []
    for h in hours:
        h_df = df.filter(pl.col("Hour") == h)
        total = h_df.height
        success = h_df.filter(pl.col("Is_Successful")).height
        rate = success / total if total > 0 else 0
        hourly_success_data.append((h, total, success, rate))
    hourly_success = pd.DataFrame(hourly_success_data, columns=["Hour", "Total_Rides", "Successful_Rides", "Success_Rate"])
    hourly_success["Hour"] = pd.to_numeric(hourly_success["Hour"], errors='coerce') 
    axes[0, 1].plot(hourly_success["Hour"], hourly_success["Success_Rate"] * 100, marker="o", color="blue")
    axes[0, 1].set_title("Success Rate by Hour of Day")
    axes[0, 1].set_xlabel("Hour")
    axes[0, 1].set_ylabel("Success Rate (%)")
    axes[0, 1].grid(True, alpha=0.3)

    # Success rate by Day of Week
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    daily_success_data = []
    for day in days:
        day_df = df.filter(pl.col("DayOfWeek") == day)
        total = day_df.height
        success = day_df.filter(pl.col("Is_Successful")).height
        rate = success / total if total > 0 else 0
        daily_success_data.append((day, total, success, rate))
    daily_success = pd.DataFrame(daily_success_data, columns=["DayOfWeek", "Total_Rides", "Successful_Rides", "Success_Rate"])
    colors = [ "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33", "#f40202", "#f781bf"]
    axes[1, 0].bar(daily_success["DayOfWeek"], daily_success["Success_Rate"] * 100, color=colors)
    axes[1, 0].set_title("Success Rate by Day of Week")
    axes[1, 0].set_ylabel("Success Rate (%)")
    axes[1, 0].tick_params(axis="x", rotation=45)

    # Booking status distribution
    status_categories = df["Status_Category"].unique()
    status_counts_data = []
    for status in status_categories:
        count = df.filter(pl.col("Status_Category") == status).height
        status_counts_data.append((status, count))
    status_counts = pd.DataFrame(status_counts_data, columns=["Status_Category", "Count"])
    axes[1, 1].pie(status_counts["Count"], labels=status_counts["Status_Category"], autopct="%1.1f%%")
    axes[1, 1].set_title("Overall Booking Status Distribution")

    plt.tight_layout()
    plt.show()

def plot_revenue_analysis(df):
    """
    Visualize revenue patterns across vehicle types, time, and ride distance.
    Args:
        df (pl.DataFrame): Input Polars DataFrame with derived columns such as 
                           'Is_Successful', 'Vehicle Type', 'Hour', 
                           'Booking Value', and 'Ride Distance'.

    Returns:
        None: Displays a matplotlib figure with four subplots.

    Subplots Generated:
        - Total Revenue by Vehicle Type (bar chart).
        - Average Revenue by Hour of Day (line plot).
        - Revenue Distribution (histogram).
        - Revenue vs Ride Distance (scatter plot).
    """

    successful_rides = df.filter(pl.col("Is_Successful"))
    vehicle_types = df["Vehicle Type"].unique()
    vehicle_revenue_data = []
    for vt in vehicle_types:
        vt_df = successful_rides.filter(pl.col("Vehicle Type") == vt)
        total_rides = vt_df.height
        total_revenue = vt_df["Booking Value"].sum() if total_rides > 0 else 0
        avg_revenue = vt_df["Booking Value"].mean() if total_rides > 0 else 0
        vehicle_revenue_data.append((vt, total_rides, total_revenue, avg_revenue))
    total_revenue_all = sum(x[2] for x in vehicle_revenue_data)
    vehicle_revenue_data = [
        (vt, tr, rev, avg, (rev / total_revenue_all * 100 if total_revenue_all > 0 else 0))
        for vt, tr, rev, avg in vehicle_revenue_data
    ]
    vehicle_revenue = pd.DataFrame(
        vehicle_revenue_data,
        columns=["Vehicle Type", "Ride_Count", "Total_Revenue", "Avg_Revenue", "Revenue_Share"]
    ).sort_values("Revenue_Share", ascending=False)
    # Revenue by Hour
    hours = range(24)
    hourly_revenue_data = []
    for h in hours:
        h_df = successful_rides.filter(pl.col("Hour") == h)
        avg_revenue = h_df["Booking Value"].mean() if h_df.height > 0 else 0
        hourly_revenue_data.append((h, avg_revenue))
    hourly_revenue = pd.DataFrame(hourly_revenue_data, columns=["Hour", "Avg_Revenue"])
    hourly_revenue["Hour"] = pd.to_numeric(hourly_revenue["Hour"], errors='coerce')
    revenue_values = successful_rides["Booking Value"].to_list()
    distance_values = successful_rides["Ride Distance"].to_list()
    revenue_vs_distance = successful_rides["Booking Value"].to_list()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    axes[0,0].bar(vehicle_revenue["Vehicle Type"], vehicle_revenue["Total_Revenue"], color=colors)
    axes[0,0].set_title("Total Revenue by Vehicle Type")
    axes[0,0].set_ylabel("Revenue (₹)")
    axes[0,0].tick_params(axis="x", rotation=45)
    axes[0,1].plot(hourly_revenue["Hour"], hourly_revenue["Avg_Revenue"], marker='o', color="orange", linewidth=2)
    axes[0,1].set_title("Average Revenue by Hour")
    axes[0,1].set_xlabel("Hour of Day")
    axes[0,1].set_ylabel("Avg Revenue (₹)")
    axes[0,1].grid(True, alpha=0.3)
    axes[1,0].hist(revenue_values, bins=50, alpha=0.7, color="blue", edgecolor='black')
    axes[1,0].set_title("Revenue Distribution")
    axes[1,0].set_xlabel("Booking Value (₹)")
    axes[1,0].set_ylabel("Frequency")
    axes[1,1].scatter(distance_values, revenue_vs_distance, alpha=0.5, color="blue")
    axes[1,1].set_title("Revenue vs Distance")
    axes[1,1].set_xlabel("Distance (km)")
    axes[1,1].set_ylabel("Revenue (₹)")
    plt.tight_layout()
    plt.show()

def revenue_prediction_model(df):
    """
    Build and evaluate a revenue prediction model for successful rides using Random Forest.
    Args:
        df (pl.DataFrame): Input Polars DataFrame with columns including 
                           'Is_Successful', 'Booking Value', 'Hour', 'Month',
                           'IsWeekend', 'Vehicle Type', 'Pickup Location', and 'Drop Location'.

    Returns:
        None: Logs model performance metrics (R², RMSE) and displays a scatter plot 
              comparing actual vs predicted revenue.
    """

    logging.info("Building Revenue Prediction Model...")
    df_revenue_model = df.filter((pl.col("Is_Successful") == True) & pl.col("Booking Value").is_not_null())
    n_samples = df_revenue_model.height
    logging.info(f"Samples for revenue prediction: {n_samples}")

    # Cyclical time features
    df_model_features = df_revenue_model.with_columns([
        pl.col("Hour").map_elements(lambda x: np.sin(2 * np.pi * x / 24), return_dtype=pl.Float64).alias("Hour_Sin"),
        pl.col("Hour").map_elements(lambda x: np.cos(2 * np.pi * x / 24), return_dtype=pl.Float64).alias("Hour_Cos"),
        pl.col("Month").map_elements(lambda x: np.sin(2 * np.pi * x / 12), return_dtype=pl.Float64).alias("Month_Sin"),
        pl.col("Month").map_elements(lambda x: np.cos(2 * np.pi * x / 12), return_dtype=pl.Float64).alias("Month_Cos"),
        pl.col("IsWeekend").cast(pl.Int64).alias("IsWeekend_Numeric")
    ])
    le_vehicle = LabelEncoder()
    le_pickup = LabelEncoder()
    le_drop = LabelEncoder()
    df_model_features = df_model_features.with_columns([
        pl.Series("Vehicle_Type_Encoded", le_vehicle.fit_transform(df_model_features["Vehicle Type"].to_list())),
        pl.Series("Pickup_Encoded", le_pickup.fit_transform(df_model_features["Pickup Location"].to_list())),
        pl.Series("Drop_Encoded", le_drop.fit_transform(df_model_features["Drop Location"].to_list()))
    ])
    feature_columns = [
        "Hour_Sin", "Hour_Cos", "Month_Sin", "Month_Cos",
        "IsWeekend_Numeric", "Vehicle_Type_Encoded",
        "Pickup_Encoded", "Drop_Encoded"
    ]
    X = df_model_features.select(feature_columns).to_pandas()
    y = df_model_features["Booking Value"].to_pandas()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_revenue = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_revenue.fit(X_train, y_train)
    y_pred = rf_revenue.predict(X_test)
    rev_r2 = r2_score(y_test, y_pred)
    rev_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    logging.info(f"Revenue Prediction R²: {rev_r2:.3f}")
    logging.info(f"Revenue Prediction RMSE: ₹{rev_rmse:.2f}")

    # Plot: Actual vs. Predicted Revenue
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Revenue')
    plt.ylabel('Predicted Revenue')
    plt.title('Revenue Prediction: Actual vs Predicted')
    plt.tight_layout()
    plt.show()

    # Feature importance plot
    revenue_feature_importance = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": rf_revenue.feature_importances_
    }).sort_values("Importance", ascending=False)
    plt.figure(figsize=(8, 5))
    sns.barplot(x="Importance", y="Feature", data=revenue_feature_importance, orient="h", color="red")
    plt.title("Feature Importances for Revenue Prediction")
    plt.tight_layout()
    plt.show()

def run_pipeline(input_file):
    """
    Execute the full ride analysis pipeline from data loading to revenue prediction.

    Args:
        input_file (str): Path to the input data file to be processed.

    Returns:
        None: Prints initial data checks, logs key metrics, generates visualizations,
              and builds a revenue prediction model.
    """

    logging.info("Starting ride analysis pipeline.")
    df = load_data(input_file)
    print("Here's how are dataframe looks like:")
    print(df.head())  # for initial sanity check
    df = check_duplicates(df)
    df = parse_datetime(df)
    df = add_time_features(df)
    df = clean_numeric(df)
    df = create_flags(df)
    basic_metrics(df)
    plot_success_patterns(df)
    plot_revenue_analysis(df)
    revenue_prediction_model(df)
    logging.info("Pipeline complete.")


if __name__ == "__main__":
    file_path= "ncr_ride_bookings.csv"
    run_pipeline(file_path)