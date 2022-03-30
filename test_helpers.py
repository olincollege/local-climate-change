import pytest
import pandas as pd
import numpy as np

from helpers import (
    convert_year,
    clean_data,
    compile_CSVs,
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

clean_data_cases = [
    ()
]

compile_CSVs_cases = [
    {'Temperature Experiment':
        pd.DataFrame(data={
            'date': np.divide(range(24013, 24072), 12),
            'avgHighTemp': range(-8, 51),
            'avgLowTemp': range(-18, 41)
        }, dtype=np.float64)
     }
]

filter_month_cases = [
    # Check a standard case, with a short list of sequential date.
    ()
]


@pytest.mark.parametrize("input_year_series, float_dates_list",
                         convert_year_cases)
def test_convert_year(input_year_series, float_dates_list):
    """_summary_

    Args:
        input_year_series (_type_): _description_
        float_dates_list (_type_): _description_
    """
    assert convert_year(input_year_series) == float_dates_list


@pytest.mark.parametrize("dataframe, city, output_dataframe",
                         clean_data_cases)
def test_clean_data(dataframe, city, output_dataframe):
    """_summary_

    Args:
        dataframe (_type_): _description_
        city (_type_): _description_
        output_dataframe (_type_): _description_
    """
    assert clean_data(dataframe, city) == output_dataframe


@pytest.mark.parametrize("df_dict", compile_CSVs_cases)
def test_compile_CSVs(df_dict):
    """Test if 

    Args:
        df_dict (dictionary): Dictionary containing a singular test dataframe
    """
    assert all(compile_CSVs()[
        'Temperature Experiment'] == df_dict['Temperature Experiment'])


@pytest.mark.parametrize("data, month_num, returned_df", filter_month_cases)
def test_filter_month(data, month_num, returned_df):
    """_summary_

    Args:
        data (_type_): _description_
        month_num (_type_): _description_
        returned_df (_type_): _description_
    """
    assert filter_month(data, month_num) == returned_df
