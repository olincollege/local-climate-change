"""
Tests helper functions with multiple test cases.
"""


import pytest
import pandas as pd
import numpy as np

from helpers import (
    convert_year,
    clean_data,
    compile_csvs,
    filter_month
)

convert_year_cases = [
    # Check a standard case, with a short list of sequential dates.
    (pd.Series([["1873", "01"], ["1873", "02"], ["1873", "03"]]),
     [1873+1/12, 1873+2/12, 1873+3/12]),

    # Check a case when the dates are out of sequential order.
    (pd.Series([["1873", "04"], ["1773", "01"], ["1973", "03"]]),
     [1873+4/12, 1773+1/12, 1973+3/12]),

    # Check a case with years with different numbers of digits.
    (pd.Series([["73", "01"], ["873", "02"], ["1873", "03"]]),
     [73+1/12, 873+2/12, 1873+3/12]),

    # Check a case with only one date to change.
    (pd.Series([["2000", "05"]]),
     [2000+5/12]),
]

# creating test data for clean_data
cdtest1 = pd.DataFrame({'date': ['2000-01', '2000-02', '2000-03'],
                        'avgLowTemp':
                        [12, 14, 15],
                        'avgHighTemp':
                        [40, 32, 45]})
cdtest1a = pd.DataFrame({'date': [2000+2/12, 2000+3/12],
                        'avgLowTemp': [14, 15],
                         'avgHighTemp': [32, 45]})

cdtest2 = pd.DataFrame({'date': ['1778-01', '1778-03', '1780-05', '1781-01'],
                        'avgLowTemp':
                        [4, 123, 4325, 2],
                        'avgHighTemp':
                        [1233, 2, -3, 45]})
cdtest2a = pd.DataFrame({'date': [],
                        'avgLowTemp': [],
                         'avgHighTemp': []})

cdtest3 = pd.DataFrame({'date': ['2000-01'],
                        'avgLowTemp':
                        [12],
                        'avgHighTemp':
                        [40]})
cdtest3a = pd.DataFrame({'date': [],
                        'avgLowTemp': [],
                         'avgHighTemp': []})

cdtest4 = pd.DataFrame({'date': ['2001-01', '2001-01', '2001-01', '2001-01'],
                        'avgLowTemp':
                        [22, 22, 22, 22],
                        'avgHighTemp':
                        [54, 54, 54, 54]})
cdtest4a = pd.DataFrame({'date': [2001+1/12, 2001+1/12],
                        'avgLowTemp': [22, 22],
                         'avgHighTemp': [54, 54]})

clean_data_cases = [
    # Test a standard case with 3 months of data.
    (cdtest1, 'clean data test 1', cdtest1a),

    # Test a case in which all of the data is dropped from the DataFrame.
    (cdtest2, 'clean data test 2', cdtest2a),

    # Test a case with only one row of data.
    (cdtest3, 'clean data test 3', cdtest3a),

    # Test a case where the same data is repeated in each row of the DataFrame.
    (cdtest4, 'clean data test 4', cdtest4a),
]

compile_csvs_cases = [
    {'Temperature Experiment':
        pd.DataFrame(data={
            'date': np.divide(range(24013, 24072), 12),
            'avgHighTemp': range(-8, 51),
            'avgLowTemp': range(-18, 41)
        }, dtype=np.float64)
     }
]

# creating test data for filter_month
fmtest1 = pd.DataFrame({'months': list(range(1, 13)),
                       'avgLowTemp':
                        [12, 14, 15, 12, 11, 10, 20, 43, 20, 12, 15, 12],
                        'avgHighTemp':
                            [40, 32, 45, 46, 50, 34, 65, 42, 41, 45, 53, 56]})
fmtest1a = pd.DataFrame({'months': [1],
                        'avgLowTemp': [12],
                         'avgHighTemp': [40]})

fmtest2 = pd.DataFrame({'months': list(range(1, 6)),
                       'avgLowTemp':
                        [12, 14, 15, 12, 11],
                        'avgHighTemp':
                            [40, 32, 45, 46, 50]})
fmtest2a = pd.DataFrame({'months': [4],
                        'avgLowTemp': [12],
                         'avgHighTemp': [46]})

fmtest3 = pd.DataFrame({'months': list(range(1, 37)),
                       'avgLowTemp':
                        list(range(10, 46)),
                        'avgHighTemp':
                            list(range(50, 86))})
fmtest3a = pd.DataFrame({'months': [1, 13, 25],
                        'avgLowTemp': [10, 22, 34],
                         'avgHighTemp': [50, 62, 74]})

fmtest4 = pd.DataFrame({'months': [1],
                       'avgLowTemp':
                        [34],
                        'avgHighTemp':
                            [84]})
fmtest4a = pd.DataFrame({'months': [1],
                        'avgLowTemp':
                         [34],
                         'avgHighTemp':
                            [84]})

filter_month_cases = [
    # Check a standard case, with data covering one year.
    (fmtest1, 1, fmtest1a),

    # Check a DataFrame with less than a year's worth of data.
    (fmtest2, 4, fmtest2a),

    # Check a longer case, with multiple years worth of data.
    (fmtest3, 1, fmtest3a),

    # Check a DataFrame with only one month of data.
    (fmtest4, 1, fmtest4a)
]


@ pytest.mark.parametrize("input_year_series, float_dates_list",
                          convert_year_cases)
def test_convert_year(input_year_series, float_dates_list):
    """
    Tests the convert_year function against several test cases.

    Args:
        input_year_series (Series): Series of string dates to input into
        convert_year
        float_dates_list (List): List of dates to check convert_year output
        against
    """
    assert convert_year(input_year_series) == float_dates_list


@ pytest.mark.parametrize("dataframe, city, output_dataframe",
                          clean_data_cases)
def test_clean_data(dataframe, city, output_dataframe):
    """
    Tests clean_data function against several test cases.

    Args:
        dataframe (DataFrame): data set to input into clean_data
        city (str): string containing key to test data set
        output_dataframe (DataFrame): Processed DataFrame to compare clean_data
        output against.
    """
    assert all(clean_data(dataframe, city).reset_index == output_dataframe)


@ pytest.mark.parametrize("df_dict", compile_csvs_cases)
def test_compile_csvs(df_dict):
    """
    Test if compile_csvs is working against a test data set.

    Args:
        df_dict (dictionary): Dictionary containing a singular test dataframe
    """
    assert all(compile_csvs()[
        'Temperature Experiment'] == df_dict['Temperature Experiment'])


@ pytest.mark.parametrize("data, month_num, returned_df", filter_month_cases)
def test_filter_month(data, month_num, returned_df):
    """
    Tests the filter_month function against several test cases.

    Args:
        data (DataFrame): DataFrame to input into filter_month
        month_num (int): integer indicating which month to filter by
        returned_df (DataFrame): Filtered DataFrame to compare filter_month
        output against.
    """
    assert all(filter_month(data, month_num).reset_index == returned_df)
