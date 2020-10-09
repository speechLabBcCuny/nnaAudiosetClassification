import math
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
from matplotlib import pyplot

from datetime import datetime,timedelta

from .fileUtils import standardPathStyle,list_files


#https://stackoverflow.com/questions/30079590/use-matplotlib-color-map-for-color-cycle
def get_cycle(cmap, N=None, use_index="auto"):
    if isinstance(cmap, str):
        if use_index == "auto":
            if cmap in ['Pastel1', 'Pastel2', 'Paired', 'Accent',
                        'Dark2', 'Set1', 'Set2', 'Set3',
                        'tab10', 'tab20', 'tab20b', 'tab20c']:
                use_index=True
            else:
                use_index=False
        cmap = matplotlib.cm.get_cmap(cmap)
    if not N:
        N = cmap.N
    if use_index=="auto":
        if cmap.N > 100:
            use_index=False
        elif isinstance(cmap, LinearSegmentedColormap):
            use_index=False
        elif isinstance(cmap, ListedColormap):
            use_index=True
    if use_index:
        ind = np.arange(int(N)) % cmap.N
        return pyplot.cycler("color",cmap(ind))
    else:
        colors = cmap(np.linspace(0.2,1,N))
        return pyplot.cycler("color",colors)


def createTimeIndex(selected_areas,file_properties_df,freq):
    times=[]
    # FIND earliest and latest time for time scale
    # lists in selected_areas_dict is ordered by time
    for i,area in enumerate(selected_areas):
        # get timestamp values from file_properties
        area_filtered=file_properties_df[file_properties_df.site_id==area]
        if area_filtered.size>0:
            start=area_filtered.iloc[0]["timestamp"]
            end=area_filtered.iloc[-1]["timestamp"]

            times.extend([start,end])
        else:
            print("{}, do not have any files".format(area))

    times.sort()
    all_start=times[0].replace(hour=0,minute=0,second=0)
    all_end=times[-1].replace(hour=23,minute=59,second=59)


    def days_hours_minutes(td):
        return td.days, td.seconds//3600, (td.seconds//60)%60

    #create date axis indexes depending on start,end along with frequency
    if "H" in freq:
        number_hours=3600
        count=int(freq[:-1])
        extra=math.ceil(3/count)
        periods=((all_end-all_start).total_seconds()//(number_hours*count)+48)
        globalindex = pd.date_range(all_start, periods=periods, freq=freq)
    elif "D" in freq:
        periods=(all_end-all_start).days+3
        globalindex = pd.date_range(all_start, periods=periods, freq=freq)
    elif "T" in freq:
        globalindex = pd.date_range(all_start,all_end, freq=freq)
    else:
        globalindex = pd.date_range(all_start,all_end, freq=freq)

    return globalindex,all_start,all_end



# result_path="/scratch/enis/data/nna/real/"
def loadResults(allSegments,prob2binaryFlag,threshold=0.5,channel=1):
    # try:
    if type(allSegments)!=list:
        allSegments=[allSegments]

    results=[]
    for filename in allSegments:
        filename = Path(filename)
        if not filename.exists():
            data = []
        data = np.load(filename)
        if prob2binaryFlag==True:
            data=prob2binary(data,threshold=threshold,channel=channel)
        results.append(data)

    results=np.concatenate(results)
    return results

def prob2binary(result,threshold=0.5,channel=1):
    if channel==2:
        result=np.min(result,axis=1)
    result[result>threshold]=1
    result[result<=threshold]=0
    result=result[:(result.size//10)*10]
    result=result.reshape(10,-1).max(axis=0)
    return result

# with open("/home/enis/projects/nna/data/8tags_on_8sites_DF.pkl", 'ab') as  dffile:
#             # source, destination
#     pickle.dump(df_dict, dffile)
def file2TableDict(selected_areas,model_tag_names,globalindex,globalcolumns,
                    file_properties_df,freq,dataFreq="10S",dataThreshold=0.5,
                    channel=1,gathered_results_perTag=None,
                    result_path=None,fileNameAddon="",prob2binaryFlag=True):
    # using gathered_results_perTag dictionary or  result_path to create
    # a pandas dataframe for visualizations

    # dataFreq is sampling frequency of the data,
    #most of the time we have predictions for each 10 second

    df_dict={key: None for (key) in selected_areas}
    no_result_paths=[]

    #we need to load it from files
    if gathered_results_perTag==None and (result_path==None):
        print("ERROR: gathered_results_perTag or (result_path and subDirectoryAddon )should be defined")
        return (None,None)


    for i,area in enumerate(selected_areas):
        df_sums = pd.DataFrame(index=globalindex, columns=globalcolumns).fillna(0)
        df_count = pd.DataFrame(index=globalindex, columns=globalcolumns).fillna(0)

        for modelTagName in model_tag_names:
    #         for afile in selected_areas_dict[area]:
            area_filtered=file_properties_df[file_properties_df.site_id==area]
            for afile,row in area_filtered.iterrows():
        #         data=gathered_results[afile][0]
                afile=Path(afile)
                # we either load data from multiple files or from single one
                if gathered_results_perTag==None:
                    # TODO, make _FCmodel variable
                    checkFolder=standardPathStyle(result_path,row,subDirectoryAddon=modelTagName
                                        ,fileNameAddon=fileNameAddon)
                    allSegments = list_files(str(checkFolder)+"/")
                    allSegments.sort()
                    if not allSegments:
                        data=np.empty(0)
                    else:
                        data=loadResults(allSegments,prob2binaryFlag=prob2binaryFlag,
                                        threshold=dataThreshold,channel=channel)
                        # gathered_results[file]=result[:]
                else:
                    data=gathered_results_perTag[modelTagName].get(afile,np.empty(0))[:]
                    if data.size!=0 and prob2binaryFlag==True:
                        data=prob2binary(data,threshold=0.5,channel=channel)

                if data.size==0:
                    no_result_paths.append(afile)
                    continue

                start=file_properties_df.loc[afile]["timestamp"]
                end =start+timedelta(seconds=(10*(len(data)-1)))
                index = pd.date_range(start,end, freq=dataFreq)
                df_afile=pd.DataFrame(data,index=index,columns=[modelTagName])
                # df_afile_grouped = df_afile.groupby([pd.Grouper(freq=freq)])
                # counts=df_afile_grouped.count()
                # sums=df_afile_grouped.sum()
                globalindexStart=globalindex.searchsorted(df_afile.index[0])
                globalindexStart= 0 if globalindexStart==0 else globalindexStart-1
                globalindexEnd=globalindex.searchsorted(df_afile.index[-1])
                globalindexEnd= globalindexEnd+1 if globalindexEnd==globalindexStart else globalindexEnd
                theBins=pd.cut(df_afile.index,globalindex[globalindexStart:globalindexEnd+1])
                # theBins=pd.cut(df_afile.index,globalindex)
                df_afileGrouped=df_afile.groupby(theBins)
                sums=df_afileGrouped.agg("sum")
                counts=df_afileGrouped.agg("count")
                sums.set_index(sums.index.categories.left,inplace=True)
                counts.set_index(counts.index.categories.left,inplace=True)

                df_count=df_count.add(counts, fill_value=0) #df_count.update(counts)
                df_sums=df_sums.add(sums, fill_value=0) #df_sums.update(sums)

        df_dict[area]=(df_count.copy(),df_sums.copy())

    return df_dict,no_result_paths

def reverseTableDict(selected_areas,df_dict,model_tag_names):
# Reverse order of TAG and AREA in the dataframe
# This graph for each tag for all areas

    df_dict_reverse={}
    for i,area in enumerate(selected_areas):
        df_count,df_sums=df_dict[area]
        for tagname in model_tag_names:
            df_dict_reverse.setdefault(tagname,[[],[]])
            df_dict_reverse[tagname][0].append(df_count[tagname])
            df_dict_reverse[tagname][1].append(df_sums[tagname])

    for tagname in model_tag_names:

        df_count=pd.concat(df_dict_reverse[tagname][0],axis=1)
        df_sums= pd.concat(df_dict_reverse[tagname][1],axis=1)
        df_count.columns,df_sums.columns = selected_areas,selected_areas
        df_dict_reverse[tagname]=df_count,df_sums

    return df_dict_reverse

def rawFile2Csv(csvPath,selected_areas,model_tag_names,globalindex,globalcolumns,
                    file_properties_df,freq,dataFreq="10S",dataThreshold=0.5,
                    channel=1,gathered_results_perTag=None,
                    result_path=None,fileNameAddon="",prob2binaryFlag=True):
    # using gathered_results_perTag dictionary or  result_path to create
    # a pandas dataframe for visualizations

    # dataFreq is sampling frequency of the data,
    #most of the time we have predictions for each 10 second
    if dataFreq!="10S":
        print("ERROR: this function does not do aggregation, set dataFreq to 10S")
        return None

    csvFilesWritten=[]
    no_result_paths=[]

    #we need to load it from files
    if gathered_results_perTag==None and (result_path==None):
        print("ERROR: gathered_results_perTag or (result_path and subDirectoryAddon )should be defined")
        return (None,None)


    for i,area in enumerate(selected_areas):
#         df_sums = pd.DataFrame(index=globalindex, columns=globalcolumns).fillna(0)
#         df_count = pd.DataFrame(index=globalindex, columns=globalcolumns).fillna(0)

        for modelTagName in model_tag_names:
            dfRawList=[]
    #         for afile in selected_areas_dict[area]:
            area_filtered=file_properties_df[file_properties_df.site_id==area]
            for afile,row in area_filtered.iterrows():
    ####################
                afile=Path(afile)
                # we either load data from multiple files or from single one
                if gathered_results_perTag==None:
                    checkFolder=standardPathStyle(result_path,row,subDirectoryAddon=modelTagName
                                        ,fileNameAddon=fileNameAddon)
                    allSegments = list_files(str(checkFolder)+"/")
                    if not allSegments:
                        data=np.empty(0)
                    else:
                        data=loadResults(allSegments,prob2binaryFlag=prob2binaryFlag,
                                        threshold=dataThreshold,channel=channel)
                        # gathered_results[file]=result[:]
                else:
                    data=gathered_results_perTag[modelTagName].get(afile,np.empty(0))[:]
                    if data.size!=0 and prob2binaryFlag==True:
                        data=prob2binary(data,threshold=0.5,channel=channel)

                if data.size==0:
                    no_result_paths.append(afile)
                    continue

                start=file_properties_df.loc[afile]["timestamp"]
                end =start+timedelta(seconds=(10*(len(data)-1)))
                index = pd.date_range(start,end, freq=dataFreq)
                df_afile=pd.DataFrame(data,index=index,columns=[modelTagName])
        ####################
                dfRawList.append(df_afile)
            if dfRawList:
                dfRaw=pd.concat(dfRawList)
                dfRaw=dfRaw.sort_index()
                csvFilename="_".join([area,modelTagName[1:]+".csv"])
                dfRaw.to_csv((csvPath/csvFilename),index_label="TimeStamp",header=[modelTagName[1:]])
                csvFilesWritten.append((csvPath/csvFilename))

    return csvFilesWritten,no_result_paths
