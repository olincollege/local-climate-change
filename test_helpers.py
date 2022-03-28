import pytest

from helpers import (
    convert_year,
    compile_CSVs,
    clean_data,
    filter_month
)

convert_year_cases = [

]

filter_month_cases = [

]


@pytest.mark.parametrize("input_year_series, float_dates_list", \
                            convert_year_cases)
def test_convert_year(input_year_series, float_dates_list):
    """_summary_

    Args:
        input_year_series (_type_): _description_
        float_dates_list (_type_): _description_
    """
    assert convert_year(input_year_series) == float_dates_list


@pytest.mark.parametrize("data, month_num, returned_df", filter_month_cases)
def test_filter_month(data, month_num, returned_df):
    """_summary_

    Args:
        data (_type_): _description_
        month_num (_type_): _description_
        returned_df (_type_): _description_
    """
    assert filter_month(data, month_num) == returned_df
