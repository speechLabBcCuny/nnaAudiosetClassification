"""Tests for fileUtils.py"""

import pytest

import pandas as pd
from pathlib import Path

from nna import fileUtils

test_data_standard_path_style = [
    ("/folder1/folder2/",
     pd.Series({
         "locationId": 11,
         "year": 2019,
         "region": "Anwr"
     }), "", "", Path("/folder1/folder2/Anwr/11/2019/")),
    ("/folder1/folder2/",
     pd.Series({
         "locationId": "11",
         "year": "2019",
         "region": "Anwr"
     }), "", "", Path("/folder1/folder2/Anwr/11/2019/")),
    ("/folder1/folder2/",
     pd.Series({
         "locationId": "11",
         "year": "2019",
         "region": "Anwr"
     }), "", "", Path("/folder1/folder2/Anwr/11/2019/")),
    ("folder1",
     pd.Series({
         "locationId": "11",
         "year": "2019",
         "region": "Anwr"
     }), "", "", Path("folder1/Anwr/11/2019/")),
    ("/folder1/folder2/",
     pd.Series({
         "locationId": 11,
         "year": "2019",
         "region": "Anwr"
     },
               name="S4A10292_20190615_094602.wav"), "_vgg", "_vgg",
     Path("/folder1/folder2/Anwr/11/" +
          "2019/S4A10292_20190615_094602_vgg/S4A10292_20190615_094602_vgg")),
    ("/folder1/folder2/",
     pd.Series({
         "locationId": "11",
         "year": "2019",
         "region": "Prudhoe"
     },
               name="S4A10292_20190615_094602.wav"), "", "_vgg",
     Path("/folder1/folder2/Prudhoe/11/" +
          "2019/S4A10292_20190615_094602_vgg")),
    ("/folder1/folder2/",
     pd.Series({
         "locationId": "11",
         "year": "2019",
         "region": "Prudhoe"
     },
               name="S4A10292_20190615_094602.wav"), "_XXX", "_YYY",
     Path("/folder1/folder2/Prudhoe/11/" +
          "2019/S4A10292_20190615_094602_XXX/S4A10292_20190615_094602_YYY")),
    ("folder1/folder2",
     pd.Series({
         "locationId": "11",
         "year": "2019",
         "region": "Prudhoe"
     },
               name="S4A10292_20190615_094602.wav"), "_XXX", "",
     Path("folder1/folder2/Prudhoe/11/" +
          "2019/S4A10292_20190615_094602_XXX/")),
]


@pytest.mark.parametrize(
    "parent_path, row, sub_directory_addon, file_name_addon,expected",
    test_data_standard_path_style)
def test_standard_path_style(parent_path, row, sub_directory_addon,
                             file_name_addon, expected):
    output_path = fileUtils.standard_path_style(parent_path, row,
                                                sub_directory_addon,
                                                file_name_addon)
    assert output_path == expected


# TEST/XX/2018/
test_data_parse_file_path = [
    ("S4A10327_20190531_060000_embeddings000.npy", 0, {
        "timestamp": "20190531_060000",
        "region": "",
        "locationId": "",
        "year": "",
        "part_index": 0,
    }),
    ("/S4A10307_20190731_221602_XXX11.flac", 0, {
        "timestamp": "20190731_221602",
        "region": "",
        "locationId": "",
        "year": "",
        "part_index": 11,
    }),
    ("/2019/S4A10307_20190731_221602_XXX010.flac", 0, {
        "timestamp": "20190731_221602",
        "region": "",
        "locationId": "",
        "year": "2019",
        "part_index": 10,
    }),
    ("2019/S4A10307_20190731_221602_XXX010.flac", 0, {
        "timestamp": "20190731_221602",
        "region": "",
        "locationId": "",
        "year": "2019",
        "part_index": 10,
    }),
    ("/17/2019/S4A10307_20190731_221602_XXX010.flac", 0, {
        "timestamp": "20190731_221602",
        "region": "",
        "locationId": "17",
        "year": "2019",
        "part_index": 10,
    }),
    ("/tank/data/nna/real/prudhoe/17/2019/S4A10307_20190731_221602.flac", 0, {
        "timestamp": "20190731_221602",
        "region": "prudhoe",
        "locationId": "17",
        "year": "2019",
        "part_index": None,
    }),
    ("/tank/data/nna/real/prudhoe/17/2019/S4A10307_20190731_221602_XXX000.flac",
     0, {
         "timestamp": "20190731_221602",
         "region": "prudhoe",
         "locationId": "17",
         "year": "2019",
         "part_index": 0,
     }),
    ("/tank/data/nna/real/prudhoe/17/2019/S4A10307_20190731_221602_XXX010.flac",
     0, {
         "timestamp": "20190731_221602",
         "region": "prudhoe",
         "locationId": "17",
         "year": "2019",
         "part_index": 10,
     }),
    ("YY/XX/tank/nna/real/prudhoe/17/2019/S4A10307_20190731_221602_XXX010.flac",
     0, {
         "timestamp": "20190731_221602",
         "region": "prudhoe",
         "locationId": "17",
         "year": "2019",
         "part_index": 10,
     }),
]


@pytest.mark.parametrize("file_path, debug,expected",
                         test_data_parse_file_path)
def test_parse_file_path(file_path, debug, expected):
    output = fileUtils.parse_file_path(file_path, debug=debug)

    assert output == expected


prudhoeAndAnwr4photoExp_dataV1 = pd.read_pickle(
    "./data/database/prudhoeAndAnwr4photoExp_dataV1.pkl")
#

test_data_match_path_info2row = [
    ({
        "timestamp": "20190811_103000",
        "region": "prudhoe",
        "locationId": "12",
        "year": "2019",
    }, prudhoeAndAnwr4photoExp_dataV1, 0,
     prudhoeAndAnwr4photoExp_dataV1.loc[Path(
         "/tank/data/nna/real/prudhoe/12/2019/S4A10274_20190811_103000.flac")]
     ),
    ({
        "timestamp": "20190621_000000",
        "region": "anwr",
        "locationId": "35",
        "year": "2019",
    }, prudhoeAndAnwr4photoExp_dataV1, 0,
     prudhoeAndAnwr4photoExp_dataV1.loc[Path(
         "/tank/data/nna/real/anwr/35/2019/S4A10272_20190621_000000.flac")]),
]


@pytest.mark.parametrize("path_info, file_properties_df, debug,expected",
                         test_data_match_path_info2row)
def test_match_path_info2row(path_info, file_properties_df, debug, expected):
    output = fileUtils.match_path_info2row(path_info, file_properties_df,
                                           debug)

    assert list(output[1].items()) == list(expected.items())
