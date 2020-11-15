"""Tests for mock_data.py.
"""

import numpy as np

import mock_data


def test_mock_file_properties_df():
    """test for mock_data.mock_file_properties_df() func.
    """
    index_length_of_df = 101
    expected_columns = [
        'region',
        'site_name',
        'locationId',
        'site_id',
        'recorderId',
        'timestamp',
        'year',
        'month',
        'day',
        'hour_min_sec',
        'timestampEnd',
        'durationSec',
    ]
    expected_columns.sort()
    expected_columns_types = {
        'site_id': np.dtype('O'),
        'locationId': np.dtype('O'),
        'site_name': np.dtype('O'),
        'recorderId': np.dtype('O'),
        'hour_min_sec': np.dtype('O'),
        'year': np.dtype('O'),
        'month': np.dtype('O'),
        'day': np.dtype('O'),
        'region': np.dtype('O'),
        'timestamp': np.dtype('<M8[ns]'),
        'durationSec': np.dtype('O'),
        'timestampEnd': np.dtype('<M8[ns]')
    }
    mocked_file_properties_df = mock_data.mock_file_properties_df(
        index_length=index_length_of_df,
        semantic_errors=False,
        structural_errors=False,
    )
    mock_column_values = sorted(list(mocked_file_properties_df.columns.values))
    mock_column_dtypes = dict(mocked_file_properties_df.dtypes.items())
    assert len(mocked_file_properties_df.index) == index_length_of_df
    assert mock_column_values == expected_columns
    assert mock_column_dtypes == expected_columns_types
