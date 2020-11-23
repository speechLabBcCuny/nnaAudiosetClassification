"""Tests for fileUtils.py"""

from typing import List, Sequence

import pytest
import csv

import pandas as pd
from pathlib import Path

from nna import fileUtils
from testparams import INPUTS_OUTPUTS_PATH

IO_fileUtils_path = INPUTS_OUTPUTS_PATH / "fileUtils"

test_data_save_to_csv = [
    (
        (IO_fileUtils_path / "save_to_csv" / "outputs" / "test"),
        [
            ("V1_firstLine-FirstItem", "firstLine-SecondItem"),
            ("V1_secondLine-FirstItem", "secondLine-SecondItem"),
        ],
        [
            ["V1_firstLine-FirstItem", "firstLine-SecondItem"],
            ["V1_secondLine-FirstItem", "secondLine-SecondItem"],
        ],  # result should be previous lines and this one
    ),
    (
        (IO_fileUtils_path / "save_to_csv" / "outputs" / "test.csv"),
        [
            ("V2_firstLine-FirstItem", "firstLine-SecondItem"),
            ("V2_secondLine-FirstItem", "secondLine-SecondItem"),
        ],
        [
            ["V2_firstLine-FirstItem", "firstLine-SecondItem"],
            ["V2_secondLine-FirstItem", "secondLine-SecondItem"],
        ],  # result should be previous lines and this one
    ),
    (
        (IO_fileUtils_path / "save_to_csv" / "outputs" / "test2"),
        [
            ("V3_firstLine-FirstItem", "firstLine-SecondItem"),
            ("V3_secondLine-FirstItem", "secondLine-SecondItem"),
        ],
        [
            ["V3_firstLine-FirstItem", "firstLine-SecondItem"],
            ["V3_secondLine-FirstItem", "secondLine-SecondItem"],
        ],  # result should be previous lines and this one
    ),
]


@pytest.fixture(scope="function")
def output_folder(request):
    # print("setup")
    file_name, lines, expected = request.param
    # print(file_name.exists())
    file_name = Path(file_name).with_suffix(".csv")
    file_name.parent.mkdir(parents=True, exist_ok=True)
    yield (file_name, lines, expected)
    print("teardown")
    print(file_name)
    file_name.unlink(missing_ok=True)


@pytest.mark.parametrize(
    "output_folder",
    test_data_save_to_csv,
    indirect=True,
)
def test_save_to_csv(
        # file_name,
        # lines,
        # expected,
    output_folder):  #pylint:disable=W0621
    file_name, lines, expected = output_folder
    fileUtils.save_to_csv(file_name, lines)

    rows: List[Sequence] = []
    with open(file_name, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append(row)

    assert expected == rows


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
    ("folder1", pd.Series({
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
          "2019/S4A10292_20190615_094602__vgg/S4A10292_20190615_094602__vgg")),
    ("/folder1/folder2/",
     pd.Series({
         "locationId": "11",
         "year": "2019",
         "region": "Prudhoe"
     },
               name="S4A10292_20190615_094602.wav"), "", "vgg",
     Path("/folder1/folder2/Prudhoe/11/" +
          "2019/S4A10292_20190615_094602_vgg")),
    ("/folder1/folder2/",
     pd.Series({
         "locationId": "11",
         "year": "2019",
         "region": "Prudhoe"
     },
               name="S4A10292_20190615_094602.wav"), "XXX", "YYY",
     Path("/folder1/folder2/Prudhoe/11/" +
          "2019/S4A10292_20190615_094602_XXX/S4A10292_20190615_094602_YYY")),
    ("folder1/folder2",
     pd.Series({
         "locationId": "11",
         "year": "2019",
         "region": "Prudhoe"
     },
               name="S4A10292_20190615_094602.wav"), "XXX", "",
     Path("folder1/folder2/Prudhoe/11/" +
          "2019/S4A10292_20190615_094602_XXX/")),
    ("/folder1/folder2/", {
        "locationId": 11,
        "year": 2019,
        "region": "Anwr",
        "name": "S4A10292_20190615_094602.wav",
    }, "", "", Path("/folder1/folder2/Anwr/11/2019/")),
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


@pytest.mark.parametrize("file_path, debug,expected", test_data_parse_file_path)
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
         "/tank/data/nna/real/prudhoe/12/2019/S4A10274_20190811_103000.flac")]),
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
    output = fileUtils.match_path_info2row(path_info, file_properties_df, debug)

    assert list(output[1].items()) == list(expected.items())
