import math
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
from matplotlib import pyplot

from datetime import datetime,timedelta



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
        globalindex = pd.date_range(all_start, periods=((all_end-all_start).total_seconds()//(number_hours*count)+48), freq=freq)
    elif "D" in freq:
        globalindex = pd.date_range(all_start, periods=(all_end-all_start).days+3, freq=freq)
    elif "T" in freq:
        globalindex = pd.date_range(all_start,all_end, freq=freq)

    return globalindex,all_start,all_end

def standardPathStyle(parentPath,row,subDirectoryAddon=None,fileNameAddon=None):
    src=Path(parentPath) / row.region /row.locationId/ row.year
    if subDirectoryAddon or fileNameAddon:
        fileName=Path(row.name)
    if subDirectoryAddon:
        src = src / (fileName.stem +subDirectoryAddon)
    if fileNameAddon:
        src= src / (fileName.stem +fileNameAddon)
    return src

# result_path="/scratch/enis/data/nna/real/"
def loadResults(filename,threshold=0.5,channel=1):
    # try:
    if not filename.exists():
        return []
    result=np.load(filename)
    prob2binary(result,threshold=0.5,channel=channel)

    return result

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
                    file_properties_df,freq,dataFreq="10S",dataThreshold=0.5,channel=1,gathered_results_perTag=None,result_path=None):



    df_dict={key: None for (key) in selected_areas}
    no_result_paths=[]

    #we need to load it from files
    if gathered_results_perTag==None and result_path==None:
        print("ERROR: gathered_results_perTag or result_path should be defined")
        return None

    for i,area in enumerate(selected_areas):
        df_sums = pd.DataFrame(index=globalindex, columns=globalcolumns).fillna(0)
        df_count = pd.DataFrame(index=globalindex, columns=globalcolumns).fillna(0)

        for tag_name in model_tag_names:
    #         for afile in selected_areas_dict[area]:
            area_filtered=file_properties_df[file_properties_df.site_id==area]
            for afile,row in area_filtered.iterrows():
        #         data=gathered_results[afile][0]
                afile=Path(afile)
                # we either load data from multiple files or from single one
                if gathered_results_perTag==None:
                    # TODO, make _FCmodel variable
                    filename=standardPathStyle(result_path,row,subDirectoryAddon="_FCmodel"
                                        ,fileNameAddon="_FCmodel000.npy")
                    data=loadResults(filename,threshold=dataThreshold,channel=channel)
                        # gathered_results[file]=result[:]
                else:
                    data=gathered_results_perTag[tag_name].get(afile,[])[:]
                    data=prob2binary(data,threshold=0.5,channel=1)

                if type(data)==list:
                    no_result_paths.append(afile)
                    pass

                start=file_properties_df.loc[afile]["timestamp"]
                index = pd.date_range(start,start+timedelta(seconds=(10*(len(data)-1))), freq=dataFreq)

                df_afile=pd.DataFrame(data,index=index,columns=[tag_name])

                df_afile_grouped = df_afile.groupby([pd.Grouper(freq=freq)])
                counts=df_afile_grouped.count()
                sums=df_afile_grouped.sum()
                df_count.update(counts)
                df_sums.update(sums)

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
