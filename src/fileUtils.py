from PIL import Image, ExifTags
import glob

import subprocess
from pathlib import Path

import time,datetime
from datetime import timedelta


import os
import sys

from labeling_utils import ffmpeg_split_mp3
from pydub import AudioSegment

from IPython.display import display

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

# TODO ,work with relative paths not absolute
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


# example usage in ../notebooks/save_file_properties.ipynb
def getLength(input_video):
    cmd=['ffprobe', '-i', '{}'.format(input_video), '-show_entries' ,'format=duration', '-v', 'quiet' ]
    result = subprocess.Popen(cmd, stdout=subprocess.PIPE,stderr=subprocess.PIPE,)
    output = result.communicate(b'\n')
    return output[0].decode('ascii').split("\n")[1].split("=")[1]



def list_files(search_path="/search_path/",ignore_folders=[]):
    all_path=glob.glob(search_path+"**/*.*",recursive=True)

    all_path=set(all_path)
    for folder in ignore_folders:
        ignore_paths = set( glob.glob(folder + "**/*.*",recursive=True) )
        all_path=all_path.difference(ignore_paths)
    return list(all_path)


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
    return start_time,end_time,loc_key,start_time_org,end_time_org

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
    print(file_name)
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

def query_audio(location,start_time,end_time,length,buffer,file_properties_df,file_name,display_flag=True,save=True,tmp_folder="./tmp_audio_excerpt/"):

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

    return (sorted_filtered)
