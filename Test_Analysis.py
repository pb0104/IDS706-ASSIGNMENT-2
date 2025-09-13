import os
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import polars as pl

from Analysis import (
    parse_datetime,
    add_time_features,
    clean_numeric,
    create_flags,
    check_duplicates,
    basic_metrics,
    run_pipeline,
    revenue_prediction_model,
)


class TestUnitTests(unittest.TestCase):
    def setUp(self):
        self.df = pl.DataFrame({
            "Date": ["01/01/25", "02/01/25"],
            "Time": ["12:00:00", "13:00:00"],
            "Booking Value": [100, 200],
            "Ride Distance": [5.0, 10.0],
            "Driver Ratings": [4.5, 3.8],
            "Customer Rating": [4.0, 3.5],
            "Booking Status": ["Completed", "Cancelled by Customer"],
            "Vehicle Type": ["Sedan", "SUV"],
            "Cancelled Rides by Customer": [None, 1],
            "Cancelled Rides by Driver": [None, None],
            "Pickup Location": ["LocA", "LocB"],
            "Drop Location": ["LocC", "LocD"],
        })

    def test_parse_datetime(self):
        parsed = parse_datetime(self.df)
        self.assertIn("DateTime", parsed.columns)

    def test_add_time_features(self):
        parsed = parse_datetime(self.df)
        enriched = add_time_features(parsed)
        self.assertIn("Hour", enriched.columns)
        self.assertIn("DayOfWeek", enriched.columns)

    def test_clean_numeric(self):
        cleaned = clean_numeric(self.df)
        self.assertEqual(cleaned["Booking Value"].dtype, pl.Float64)

    def test_create_flags(self):
        flags = create_flags(self.df)
        self.assertIn("Is_Successful", flags.columns)
        self.assertIn("Status_Category", flags.columns)

    def test_check_duplicates(self):
        dup_df = self.df.vstack(self.df)
        deduped = check_duplicates(dup_df)
        self.assertEqual(deduped.shape[0], 2)


class TestIntegrationTests(unittest.TestCase):
    def setUp(self):
        # Increase rows to ensure train/test split works
        self.rows = 20
        self.df = pl.DataFrame({
            "Date": ["01/01/25"] * self.rows,
            "Time": ["12:00:00"] * self.rows,
            "Booking Value": np.random.randint(100, 500, self.rows).tolist(),
            "Ride Distance": np.random.rand(self.rows).tolist(),
            "Driver Ratings": np.random.rand(self.rows).tolist(),
            "Customer Rating": np.random.rand(self.rows).tolist(),
            "Booking Status": ["Completed"] * self.rows,
            "Vehicle Type": ["Sedan"] * self.rows,
            "Cancelled Rides by Customer": [None] * self.rows,
            "Cancelled Rides by Driver": [None] * self.rows,
            "Pickup Location": ["LocA"] * self.rows,
            "Drop Location": ["LocB"] * self.rows,
        })

    def test_basic_metrics_integration(self):
        parsed = parse_datetime(self.df)
        enriched = add_time_features(parsed)
        enriched = clean_numeric(enriched)
        enriched = create_flags(enriched)
        basic_metrics(enriched)

    def test_pipeline_data_transform(self):
        # Write dataframe to a temporary CSV file
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        tmp.close()
        self.df.write_csv(tmp.name)

        # Run the full pipeline
        run_pipeline(tmp.name)

        # Clean up temporary file
        os.remove(tmp.name)



class TestSystemTests(unittest.TestCase):
    def setUp(self):
        self.rows = 20
        self.df = pl.DataFrame({
            "Date": ["01/01/25"] * self.rows,
            "Time": ["12:00:00"] * self.rows,
            "Booking Value": np.random.randint(100, 500, self.rows).tolist(),
            "Ride Distance": np.random.rand(self.rows).tolist(),
            "Driver Ratings": np.random.rand(self.rows).tolist(),
            "Customer Rating": np.random.rand(self.rows).tolist(),
            "Booking Status": ["Completed"] * self.rows,
            "Vehicle Type": ["Sedan"] * self.rows,
            "Cancelled Rides by Customer": [None] * self.rows,
            "Cancelled Rides by Driver": [None] * self.rows,
            "Pickup Location": ["LocA"] * self.rows,
            "Drop Location": ["LocB"] * self.rows,
        })

    @patch("matplotlib.pyplot.show")
    def test_revenue_model_end_to_end(self, mock_show):
        parsed = parse_datetime(self.df)
        enriched = add_time_features(parsed)
        enriched = clean_numeric(enriched)
        enriched = create_flags(enriched)

        revenue_prediction_model(enriched)
        self.assertTrue(mock_show.called)


class TestCoverageBoost(unittest.TestCase):
    def setUp(self):
        # Increasing rows to ensure train/test split works
        self.rows = 50
        self.df = pl.DataFrame({
            "Date": ["01/01/25"] * self.rows,
            "Time": ["12:00:00"] * self.rows,
            "Booking Value": np.random.randint(100, 500, self.rows).tolist(),
            "Ride Distance": np.random.rand(self.rows).tolist(),
            "Driver Ratings": np.random.rand(self.rows).tolist(),
            "Customer Rating": np.random.rand(self.rows).tolist(),
            "Booking Status": ["Completed"] * self.rows,
            "Vehicle Type": ["Sedan"] * self.rows,
            "Cancelled Rides by Customer": [None] * self.rows,
            "Cancelled Rides by Driver": [None] * self.rows,
            "Pickup Location": ["LocA"] * self.rows,
            "Drop Location": ["LocB"] * self.rows,
        })

    @patch("matplotlib.pyplot.show")
    def test_revenue_prediction_model_runs(self, mock_show):
        df = parse_datetime(self.df)
        df = add_time_features(df)
        df = clean_numeric(df)
        df = create_flags(df)

        # Call the revenue prediction model
        revenue_prediction_model(df)

        # Ensure plotting function was called
        self.assertTrue(mock_show.called)

    @patch("matplotlib.pyplot.show")
    def test_run_pipeline_end_to_end(self, mock_show):
        # Write dataframe to a temporary CSV file
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        tmp.close()
        self.df.write_csv(tmp.name)

        # Run the full pipeline
        run_pipeline(tmp.name)

        # Clean up temporary file
        os.remove(tmp.name)

        # Ensure plotting function was called
        self.assertTrue(mock_show.called)


if __name__ == "__main__":
    unittest.main()
