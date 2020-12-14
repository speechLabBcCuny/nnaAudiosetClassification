import pytest
from nna import dataimport


test_data_megan_excell_row2yaml_code = [
    (
        {
            'Anthro/Bio': '',
            'Category': 'Bug',
            'Specific Category': ''
        },
        '1.3.0',
    ),
    (
        {
            'Anthro/Bio': 'TEST',
            'Category': 'BUg',
            'Specific Category': ''
        },
        '1.3.0',
    ),
    (
        {
            'Anthro/Bio': 'antH',
            'Category': '',
            'Specific Category': ''
        },
        '0.0.0',
    ),
]

@pytest.mark.parametrize('row, expected_output',
                         test_data_megan_excell_row2yaml_code)
def test_megan_excell_row2yaml_code(row, expected_output):
    output = dataimport.megan_excell_row2yaml_code(row)

    assert expected_output == output