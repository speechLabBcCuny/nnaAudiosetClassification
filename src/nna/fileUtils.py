from PIL import Image, ExifTags
import glob

import subprocess
from pathlib import Path

import time,datetime
from datetime import timedelta


import os
import sys

from pydub import AudioSegment

from IPython.display import display

import csv
def save_to_csv(file_name,lines):
    file_name=Path(file_name).with_suffix('.csv')
    with open(str(file_name), mode='a') as labels_file:
        label_writer = csv.writer(labels_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for line in lines:
            label_writer.writerow(line)

def standardPathStyle(parentPath,row,subDirectoryAddon=None,fileNameAddon=None):
    src=Path(parentPath) / row.region /row.locationId/ row.year
    if subDirectoryAddon or fileNameAddon:
        fileName=Path(row.name)
    if subDirectoryAddon:
        src = src / (fileName.stem +subDirectoryAddon)
    if fileNameAddon:
        src= src / (fileName.stem +fileNameAddon)
    return src

def npy2originalFile(thePath,inputPath,outputPath,file_properties_df,
                     subDirectoryAddon=None,fileNameAddon=None,debug=0):
#     thePath.parents[parentDistance]
    if debug>0: print("func: npy2originalFile, inputs:",thePath,inputPath,
                     outputPath,"file_properties_df")
    relative2Main=thePath.relative_to(outputPath)
    fileName=relative2Main.parents[0].stem

    # find possible files in the file properties
    region=relative2Main.parts[0]
    locationId=relative2Main.parts[1]
    # here [1:3] does not work for 0813_091810_embeddings025.npy
    # [-3:-1] works for both S4A10327_20190531_060000_embeddings000.npy
    timestamp = "_".join(fileName.split("_")[-3:-1])
    yearFileName = timestamp.split("_")[0][0:4]
    yearFolder = relative2Main.parts[2]
    if yearFileName!=yearFolder and (region!="stinchcomb" and locationId!="20-Umiat"):
        print("ERROR File is in the wrong year folder ",thePath)
    if region=="stinchcomb" and locationId=="20-Umiat":
        year = yearFileName
    else:
        year = yearFolder
    isRegion=file_properties_df.region==region
    islocationID=file_properties_df.locationId==locationId
    isYear=file_properties_df.year==year

    truthTable=isRegion &  islocationID &  isYear
    filteredProperties=file_properties_df[truthTable]

    for row in filteredProperties.iterrows():
        if timestamp in str(row[0]):
            return row[0]
    if debug>0: print(timestamp,filteredProperties)
    return -1


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



# TODO ,work with relative paths not absolute
def read_file_properties(mp3_files_path_list):
    if type(mp3_files_path_list) is not list:
        with open(str(mp3_files_path_list)) as f:
            lines=f.readlines()
            mp3_files_path_list=[line.strip() for line in lines]

    site_names=[]
    hours=[]
    exceptions=[]
    file_properties={}
    for apath in mp3_files_path_list:
        apath=Path(apath)
        name=apath.stem
        if len(apath.parents)==6:
            site_name=apath.parent.stem
        else:
            site_name=" ".join(apath.parent.parent.stem.split(" ")[1:])
    #     print(site_name)
        file_id=name
        name=name.split("_")
        #ones without date folder
        if len(apath.parents)==7 and len(name)==3:
            site_name_tmp=apath.parent.stem.split(" ")
            if len(site_name_tmp)==1:
                site_name=site_name_tmp[0]
            else:
#                 print("2")
                site_name=" ".join(site_name_tmp[1:])
    #         print(site_name)
    #         print(apath)
            site_id=name[-3]
            site_names.append(site_name)
            date=name[-2]
            hour_min_sec=name[-1]
            hour=hour_min_sec[0:2]
            hours.append(hour)
            year,month,day=date[0:4],date[4:6],date[6:8]
        #usual ones
        elif len(name)==3:
            site_id=name[-3]
            site_names.append(site_name)
            date=name[-2]
            hour_min_sec=name[-1]
            hour=hour_min_sec[0:2]
            hours.append(hour)
            year,month,day=date[0:4],date[4:6],date[6:8]
        # stem does not have site_id in it
        elif len(name)==2:
            site_id="USGS"
            site_names.append(site_name)
            date=name[-2]
            hour_min_sec=name[-1]
            hour=hour_min_sec[0:2]
            hours.append(hour)
            month,day=date[0:2],date[2:4]
    #         year=Path(apath).parent.stem.split(" ")[0]

        # files with names that does not have fixed rule
        else:
            exceptions.append(apath)
        file_properties[apath]={"site_id":site_id,"site_name":site_name,
                                "hour_min_sec":hour_min_sec,"year":year,"month":month,"day":day}
        str2timestamp(file_properties[apath])
    return file_properties,exceptions

# TODO ,work with relative paths not absolute IMPORTANT fix that
# then update
def read_file_properties_v2(mp3_files_path_list):
    if type(mp3_files_path_list) is str:
        with open(str(mp3_files_path_list)) as f:
            lines=f.readlines()
            mp3_files_path_list=[line.strip() for line in lines]

    exceptions=[]
    file_properties={}
    for apath in mp3_files_path_list:
        apath=Path(apath)
        #usual ones
        if len(apath.parents)==8:
            recorderId_startDateTime=apath.stem

            recorderId_startDateTime=recorderId_startDateTime.split("_")
            recorderId=recorderId_startDateTime[0]

            date=recorderId_startDateTime[1]
            year,month,day=date[0:4],date[4:6],date[6:8]

            hour_min_sec=recorderId_startDateTime[2]
            if hour_min_sec==None:
                print(apath)
            hour=hour_min_sec[0:2]
            locationId = apath.parts[6]
            region= apath.parts[5]

            site_name=""

            file_properties[apath]=str2timestamp({"site_id":locationId,"locationId":locationId,
                                                  "site_name":site_name,"recorderId":recorderId,
                                "hour_min_sec":hour_min_sec,"year":year,"month":month,"day":day
                               ,"region":region})

        else:
            exceptions.append(apath)

    return file_properties,exceptions


# example usage in ../notebooks/Labeling/save_file_properties.ipynb
def getLength(input_video):
    cmd=['ffprobe', '-i', '{}'.format(input_video), '-show_entries' ,'format=duration', '-v', 'quiet' ]
    result = subprocess.Popen(cmd, stdout=subprocess.PIPE,stderr=subprocess.PIPE,)
    output = result.communicate(b'\n')
    return output[0].decode('ascii').split("\n")[1].split("=")[1]



def list_files(search_path="/search_path/",ignore_folders=None,filename="*.*"):
    if ignore_folders==None:
        ignore_folders=[]
    if search_path[-1]!="/":
        search_path+="/"
    all_path=glob.glob(search_path+"**/"+filename,recursive=True)

    all_path=set(all_path)
    for folder in ignore_folders:
        ignore_paths = set( glob.glob(folder + "**/*.*",recursive=True) )
        all_path=all_path.difference(ignore_paths)
    all_path=sorted(list(all_path))
    return all_path


def str2timestamp(fileinfo_dict):
    # x=file_properties[file]
#         print(x)
    hour_min_sec=fileinfo_dict["hour_min_sec"]
    hour=int(hour_min_sec[:2])
    minute=int(hour_min_sec[2:4])
    second=int(hour_min_sec[4:6])
    year = int(fileinfo_dict["year"])

    timestamp=datetime.datetime(year, int(fileinfo_dict["month"]), int(fileinfo_dict["day"]),
                hour=hour, minute=minute, second=second, microsecond=0)
    fileinfo_dict["timestamp"]=timestamp
    return fileinfo_dict


# start_time='01-07-2016_18:33:00' # or datetime object
def find_files(location,start_time,end_time,length,file_properties_df):
    import pandas as pd

    file_properties_df=file_properties_df.sort_values(by=['timestamp'])
    if location in file_properties_df["site_id"].values:
        loc_key="site_id"
    elif location in  file_properties_df["site_name"].values :
        loc_key="site_name"
    else:
        print("Location not found")
        print("Possible names and ids:")
        for site_name,site_id in set(zip(file_properties_df.site_name, file_properties_df.site_id)):
            print(site_name,"---",site_id)


    if type(start_time)==str:
        start_time = datetime.datetime.strptime(start_time, '%d-%m-%Y_%H:%M:%S')

    site_filtered = file_properties_df[file_properties_df[loc_key]==location]
    # print(site_filtered)
    if length!=0:
        end_time = start_time + datetime.timedelta(seconds=length)
    else:
        end_time=datetime.datetime.strptime(end_time, '%d-%m-%Y_%H:%M:%S')

    if not(start_time) or not(end_time):
        print("time values should be given")

    # first and last recordings from selected site
    first,last=site_filtered["timestamp"][0],site_filtered["timestamp"][-1]
    # make sure start or end time time are withing possible range
    beginning,end=max(start_time,first),min(end_time,last)

    start_file=site_filtered[site_filtered["timestamp"]<=beginning].iloc[-1:]

    time_site_filtered=site_filtered[site_filtered["timestamp"]>beginning]

    time_site_filtered=time_site_filtered[time_site_filtered["timestamp"]<end]

    time_site_filtered=pd.concat([time_site_filtered,start_file])

    sorted_filtered=time_site_filtered.sort_values(by=['timestamp'])
    # print(time_site_filtered)
    if len(sorted_filtered.index)==0:
        print("No records for these times at {} ".format(location))
        print("Earliest {}  and latest {}".format(first,last))

    return sorted_filtered,start_time,end_time


def find_filesfunc_inputs(location,start_time,end_time,length,buffer,file_properties_df):

    if location in file_properties_df["site_id"].values:
        loc_key="site_id"
    elif location in  file_properties_df["site_name"].values :
        loc_key="site_name"
    else:
        print("Location not found")
        print("Possible names and ids:")
        for site_name,site_id in set(zip(file_properties_df.site_name, file_properties_df.site_id)):
            print(site_name,"---",site_id)

    if not(start_time):
        print("start time value should be given")
    if not(end_time) and length<=0:
        print("end time value should be given or lenght should be bigger than 0")

    if type(start_time)==str:
        start_time = datetime.datetime.strptime(start_time, '%Y-%m-%d_%H:%M:%S')

    # if length is given then overwrite end time
    if length>0:
        end_time = start_time + datetime.timedelta(seconds=length)
    elif type(end_time)==str:
        end_time = datetime.datetime.strptime(start_time, '%Y-%m-%d_%H:%M:%S')

    # buffer changes start and end times, we keep original values
    if buffer>0:
        start_time_org,end_time_org = start_time,end_time
        start_time_buffered=start_time - datetime.timedelta(seconds=buffer)
        end_time_buffered=start_time + datetime.timedelta(seconds=buffer+length)
        start_time,end_time=start_time_buffered,end_time_buffered
    else:
        start_time_org,end_time_org = start_time,end_time

    # print("start:",start_time,"end",end_time)
    try:
        return start_time,end_time,loc_key,start_time_org,end_time_org
    except:
        return None

def find_filesv2(location,start_time,end_time,length,buffer,file_properties_df):
    import pandas as pd
    start_time,end_time,loc_key,start_time_org,end_time_org=find_filesfunc_inputs(location,
                                                                    start_time,end_time,length,buffer,file_properties_df)

    # sorted here
    file_properties_df=file_properties_df.sort_values(by=['timestamp'])


    site_filtered = file_properties_df[file_properties_df[loc_key]==location]


    # first and last recordings from selected site
    first,last=site_filtered["timestamp"][0],site_filtered["timestamp"][-1]
    # make sure start or end time time are withing possible range
    # print("start time",start_time,"first",first)
    beginning,end=max(start_time,first),min(end_time,last)
    # print(beginning,end)
    start_file=site_filtered[site_filtered["timestamp"]<=beginning].iloc[-1:]

    time_site_filtered=site_filtered[site_filtered["timestamp"]>beginning]

    time_site_filtered=time_site_filtered[time_site_filtered["timestamp"]<end]

    time_site_filtered=pd.concat([time_site_filtered,start_file])

    # remove ones that has an end before our start
    # print("beginning",beginning)
    time_site_filtered=time_site_filtered[time_site_filtered["timestampEnd"]>=beginning]

    sorted_filtered=time_site_filtered.sort_values(by=['timestamp'])
    # print(time_site_filtered)
#     if len(sorted_filtered.index)==0:
#         print("No records for these times at {} ".format(location))
#         print("Earliest {}  and latest {}".format(first,last))

    return sorted_filtered,start_time,end_time,start_time_org,end_time_org




def get_audio(sorted_filtered,start_t,end_t,display_flag=True,save=True,file_name="output",tmpfolder="./trap_photo_audio/"):

    total_seconds=0
    for i,f in enumerate(sorted_filtered.iterrows()):

        # in audio file, where should excerpt starts and ends
        start_seconds= max(((start_t-f[-1].timestamp)).total_seconds(),0)
        end_seconds = max((end_t-f[-1].timestamp).total_seconds(),0)

        excerpt_length=end_seconds-start_seconds

        # these are minutes and seconds are for naming
        # which seconds of the query audio in this file
        start_minute, start_second = divmod(int(total_seconds), 60)
        start_minute, start_second= str(start_minute), str(start_second)
        end_minute, end_second = divmod(int(total_seconds+excerpt_length), 60)
        end_minute, end_second= str(end_minute), str(end_second)

        file_name_specific=file_name+"_"+start_minute+"m_"+start_second+"s__"+end_minute+"m_"+end_second+"s"

        mp3_file_path=f[0]
        file_extension=str(Path(mp3_file_path).suffix)


        save_audiofile(mp3_file_path,file_extension,file_name_specific,start_seconds,end_seconds,tmpfolder)

        if display_flag:
            display_audio(tmpfolder,file_name_specific,file_extension)
        #if not save:
            #delete the file
        total_seconds+=excerpt_length

def save_audiofile(mp3_file_path,file_extension,file_name,start_seconds,end_seconds,tmpfolder):
    from .labeling_utils import ffmpeg_split_mp3
    # if end_seconds bigger than file, ffmpeg ignores it, if both out of order than output is emtpy
    ffmpeg_split_mp3(mp3_file_path,start_seconds,end_seconds,tmpfolder=tmpfolder)

    try:
        os.rename(tmpfolder+"output"+file_extension,tmpfolder+file_name+file_extension)

    except:
        print("{}".format(sys.exc_info()[0]))


def display_audio(tmpfolder,file_name,file_extension):
    try:
        if file_extension in [".mp3",".MP3"]:
            mp3file=AudioSegment.from_mp3(Path(tmpfolder+file_name+file_extension))
        else:
            mp3file=AudioSegment.from_file(Path(tmpfolder+file_name+file_extension))
        display(mp3file)
    except:
        print("{}".format(sys.exc_info()[0]))

def query_audio(location,start_time,end_time,length,buffer,
                file_properties_df,file_name,display_flag=True,
                save=True,tmp_folder="./tmp_audio_excerpt/"):

    output = find_filesv2(location,start_time,end_time,length,0,file_properties_df)
    sorted_filtered,start_time,end_time,start_time_org,end_time_org = output

    # if there is no file without buffer then search again with buffer
    if len(sorted_filtered.index)==0 and buffer>0:

        output = find_filesv2(location,start_time_org,end_time_org,length,buffer,file_properties_df)
        sorted_filtered,start_time,end_time,start_time_org,end_time_org = output

        closestLeft=sorted_filtered[sorted_filtered["timestampEnd"]<start_time_org][-1:]
        closestRight=sorted_filtered[sorted_filtered["timestamp"]>end_time_org][:1]
        if len(sorted_filtered.index)==0:
            print("Recording not found")
        elif len(closestLeft.index)==0:
            start_time=closestRight["timestamp"][0]
            end_time=closestRight["timestamp"][0] + datetime.timedelta(seconds=length)
            file_name+="_earlier_"+start_time.strftime('%Y-%m-%d_%H:%M:%S')

            get_audio(closestRight,start_time,end_time,
                      display_flag=display_flag,save=save,file_name=file_name,tmpfolder=tmp_folder)
        elif len(closestRight.index)==0:
            start_time=closestLeft["timestampEnd"][0] - datetime.timedelta(seconds=length)
            end_time=closestLeft["timestampEnd"][0]
            file_name+="_later_"+start_time.strftime('%Y-%m-%d_%H:%M:%S')
            get_audio(closestLeft,start_time,end_time,
                      display_flag=display_flag,save=save,file_name=file_name,tmpfolder=tmp_folder)

    else:
    #     print(sorted_filtered)
        file_name+="_exact_"+start_time_org.strftime('%Y-%m-%d_%H:%M:%S')
        get_audio(sorted_filtered,start_time_org,end_time_org,
                  display_flag=display_flag,save=save,file_name=file_name,tmpfolder=tmp_folder)
    print(file_name)
    return (sorted_filtered)




def findPhoto(location,timestamp,imgOnlyDate,buffer=datetime.timedelta(seconds=1)):
    """
    Example
    --------
    import datetime

    from IPython.display import display, Image
    import pickle

    # information about images
    with open("../../data/imgOnlyDateV1.pkl", 'rb') as f:
        imgOnlyDate=pickle.load(f)


    # query
    location="35"
    start_time='2019-06-05_00:00:00' # YYYY-MM-DD_HH:MM:SS or datetime object
    # if there is no recording in given timestamp, it searches before and after,
    # buffer is how far to look in seconds
    buffer=1800


    timestamp=datetime.datetime.strptime(start_time, '%Y-%m-%d_%H:%M:%S')
    end_time=None

    imgTime,imgPath=findPhoto(location,timestamp,imgOnlyDate,buffer=buffer)


    if imgTime!=-1:
        display(Image(imgPath,width=600))
    dataPoint="'{},{},{},{}',".format(location,start_time,imgPath)
    print(dataPoint)
    """
    from bisect import bisect_left

    index=bisect_left(imgOnlyDate[location], (timestamp,""))
    if index==len(imgOnlyDate[location]):
        index-=1

    if index==-1:
        index=1

    left=imgOnlyDate[location][index-1]
    right=imgOnlyDate[location][index]
#     print(index)
#     print(left[0],right[0])

    if timestamp==left[0]:
        return left
    if timestamp==right[0]:
        return right
    leftDistance=abs(timestamp-left[0])
    rightDistance=abs(right[0]-timestamp)
#     print(leftDistance,rightDistance)
    if leftDistance<=rightDistance and leftDistance<=buffer:
        print(leftDistance)
        return left
    if rightDistance<=buffer:
        print(rightDistance)
        return right
    return (-1,-1)
