"Place Holder"
import pandas as pd
import datetime

import matplotlib.pylab as pl
import nna

# In[4]:


data_folder = "/home/enis/projects/nna/data/"
results_folder = "/home/enis/projects/nna/results/"

id2name = {}
id2name["_CABLE"] = "Cable"
id2name["_RUNNINGWATER"] = "Running Water"
id2name["_INSECT"] = "Insect"
id2name["_RAIN"] = "Rain"
id2name["_WATERBIRD"] = "Water Bird"
id2name["_WIND"] = "Wind"
id2name["_SONGBIRD"] = "Songbird"
id2name["_AIRCRAFT"] = "Aircraft"

# file_properties_df=pd.read_pickle("../../data/stinchcomb_dataV1.pkl")
file_properties_df = pd.read_pickle(
    "../../../data/prudhoeAndAnwr4photoExp_dataV1.pkl")

#important to keep them in order
file_properties_df.sort_values(by=["timestamp"], inplace=True)

# delete older than 2016
fromtime = datetime.datetime(2016, 1, 1, 0)
file_properties_df = file_properties_df[
    file_properties_df.timestamp >= fromtime]
all_areas = sorted(pd.unique(file_properties_df.site_id.values))

# In[9]:

# import pandas as pd
# file_properties_df=pd.read_pickle("../../data/stinchcomb_dataV1.pkl")
# file_properties_df2=pd.read_pickle("../../data/realdata_v2No_stinchcomb.pkl")

# PARAMS
# FREQS to reduce results
# freq="30min"
# freq="2H"
freq = "270min"
# freq="135min"
# freq="continous"

# possible places to pick
# sorted(pd.unique(file_properties_df.site_id.values))
# areas to be visualized

# globalindex,all_start,all_end=createTimeIndex(selected_areas,file_properties_df,freq)

# selected_tag_name="_SONGBIRD"

# weather_cols=[]
vis_file_path = "/home/enis/projects/nna/results/vis/testtestV1/"

result_path = "/scratch/enis/data/nna/real/"
model_tag_names = [
    "CABLE", "RUNNINGWATER", "INSECT", "RAIN", "WATERBIRD", "WIND", "SONGBIRD",
    "AIRCRAFT"
]


def main():
    cmap = pl.cm.tab10
    aCmap = cmap
    my_cmaps = nna.visutils.add_normal_dist_alpha(aCmap)
    for selected_area in all_areas[1:2]:
        print(selected_area, all_areas.index(selected_area))
        nna.visutils.vis_preds_with_clipping(selected_area, file_properties_df,
                                             freq, model_tag_names, my_cmaps,
                                             result_path, data_folder,
                                             vis_file_path, id2name)


if __name__ == "__main__":
    # execute only if run as a script
    main()
