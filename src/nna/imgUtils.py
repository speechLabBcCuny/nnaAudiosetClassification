from pathlib import Path
import sqlite3

import pandas as pd
import numpy as np
import datetime

from PIL import Image, ExifTags


def readTimeLapseDatabase(pathDatasets,):
    """
    Load TimeLapse program database to dataframes

    find all files ending with .ddb at pathDatasets, read dataTable and store in pandas dataFrame
    return a dict of dataFrames, key is the location

    makes Caribou counts as integers and creates a "timestamp" column as datetime type
    """
    datasetList=[[m for m in i.glob("*") if ".ddb" in str(m) ][0] for i in (Path(pathDatasets).glob("*"))]
    labeledImgLocations = [m.parent.stem for m in datasetList]
    dictDataTable = {}
    for location,path2dataset in zip(labeledImgLocations,datasetList):
        con = sqlite3.connect(path2dataset)

        cur = con.cursor()

        dataTable=pd.read_sql('SELECT * FROM dataTable;',con)
        dataTable['Caribou']=dataTable['Caribou'].astype(int)

        start_times = dataTable["Date"]+"_"+dataTable["Time"]

        start_times=start_times.apply(lambda x:datetime.datetime.strptime(x, '%d-%b-%Y_%H:%M:%S') )
        dataTable["timestamp"]=start_times

        dictDataTable[location] = dataTable.copy()
        con.close()

    return dictDataTable

def imgDataTable2Numpy(dictDataTable):
    dictDataTable_new_index = dictDataTable.copy()
    dictNumpyTables = {}
    for dataTableID in dictDataTable_new_index:
        dictNumpyTables[dataTableID] = {}

        previousLen = len(dictDataTable_new_index[dataTableID])
        dictDataTable_new_index[dataTableID] = dictDataTable_new_index[dataTableID].set_index(['timestamp'],)
        dictDataTable_new_index[dataTableID] = dictDataTable_new_index[dataTableID].sort_index()

        caribou_np = dictDataTable_new_index[dataTableID]["Caribou"].values
        timeIndex_np = pd.Series(dictDataTable_new_index[dataTableID].index).values

        dictNumpyTables[dataTableID]['Caribou'] = caribou_np
        dictNumpyTables[dataTableID]['timestamp'] = timeIndex_np

        newLen=len(dictDataTable_new_index[dataTableID])
        if previousLen!=newLen:
            print("ERROR in imgDataTable2Numpy")
            print(previousLen,newLen)

    return dictNumpyTables



def get_labeled_exif(image_path):
    img = Image.open(image_path)
    img_exif = img.getexif()
    exifData = {}

    if img_exif is None:
        print("Sorry, image has no exif data.")
    else:
        for tag, value in img_exif.items():
            decodedTag = ExifTags.TAGS.get(tag, tag)
            exifData[decodedTag] = value
    return exifData
