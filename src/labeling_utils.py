import subprocess
import ipywidgets as widgets
from IPython.display import Audio,display,HTML
import random
from pathlib import Path

import time,datetime
from ipywidgets import Output
from IPython.display import clear_output

from os import listdir
import os
import re
import csv

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
        site_name=" ".join(apath.parent.parent.stem.split(" ")[1:])
    #     print(site_name)
        file_id=name
        name=name.split("_")
        #ones without date folder
        if len(apath.parents)==7 and len(name)==3:
            site_name=" ".join(apath.parent.stem.split(" ")[1:])
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
    #     if apath not in exceptions :
    #         if "2013" in (site_id,",",site_name,",",hour_min_sec,year,month,day):
    #             print((site_id,",",site_name,",",hour_min_sec,year,month,day))
    #             print(apath)
    #         print("year",year,"month",month,day)
    #         print(site_id,",",site_name,",",hour_min_sec,year,month,day)

        file_properties[apath]={"site_id":site_id,"site_name":site_name,
                                "hour_min_sec":hour_min_sec,"year":year,"month":month,"day":day}
    return file_properties,exceptions





def cut_random_file(mp3_files_path_list,length=10,split_folder="./splits",longest_minute=49*60,depth=0):
    input_mp3_file=random.choice(mp3_files_path_list)

    start_minute=random.randint(0,longest_minute)
    start_second=random.randint(0,59)

    extra_minute=(start_second+10)//60

    end_minute=start_minute+extra_minute
    end_second=(start_second+length)%60

    start_time = "{}.{}".format(start_minute,start_second)
    end_time = "{}.{}".format(end_minute,end_second)
#     print(input_mp3_file,start_time,end_time)

    result=splitmp3(str(input_mp3_file),split_folder,start_time,end_time)

    if result==0 and depth<3:
        print(input_mp3_file,start_time,end_time)
        cut_random_file(mp3_files_path_list,length=length,split_folder=split_folder,
                        longest_minute=longest_minute,depth=depth+1)

    else:
        return result

def splitmp3(input_mp3_file,split_folder,start_time,end_time,depth=5):
    # -f increases precision (ONLY mp3)
    # -t
    # -d folder
    # input
    #end time minute.seconds
    # start_time
    cmd = ['mp3splt','-f','-d', split_folder, input_mp3_file, start_time, end_time]

    # custom name (@f_@n+@m:@s+@M:@S)
    #cmd+=["-o","temp"+str(file_index)]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    o, e = proc.communicate()


    if proc.returncode!=0:
        print('Output: ' + o.decode('ascii'))
        print('Error: '  + e.decode('ascii'))
        return 0
    else:
#         print('Output: ' + o.decode('ascii'))
        split_file=re.search('File "(.*)"',o.decode('ascii') ).group(1)
#         print(split_file)
        return Path(split_file)

def play_html_modify(mp3file,items):
    out=items["mp3_output"]
    with out:
        clear_output()
        displayed=display(HTML("<audio controls  loop autoplay><source src={} type='audio/mpeg'></audio>".format(mp3file)))


def save_to_csv(file_name,lines):
    file_name=Path(file_name).with_suffix('.csv')
    with open(str(file_name), mode='a') as labels_file:
        label_writer = csv.writer(labels_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for line in lines:
            label_writer.writerow(line)

def load_labels(csv_file="assets/class_labels_indices.csv"):
    import csv
    if os.path.exists(csv_file):
        csvfile=open(str(csv_file), newline='')
        csv_lines=csvfile.readlines()
        csvfile.close()
    else:
        import requests
        url="https://raw.githubusercontent.com/qiuqiangkong/audioset_classification/master/metadata/class_labels_indices.csv"
        with requests.Session() as s:
            download = s.get(url)
            decoded_content = download.content.decode('utf-8')
            csv_lines=decoded_content.splitlines()
    labels=[]
    reader = csv.reader(csv_lines, delimiter=',')
    headers=next(reader)
    for row in reader:
      labels.append(row[2])
    return labels

class labeling_UI:
    def __init__ (self,tags,samples_dir="./",username="",RESULTS_DIR="./",TEST_MODE=False,model_tags={},tag_threshold=0.1):
        self.TEST_MODE=TEST_MODE
        self.tag_threshold = tag_threshold
        self.model_tags=model_tags
        self.samples_dir=samples_dir
        self.tags=tags
        self.RESULTS_DIR=RESULTS_DIR

        now = datetime.datetime.now()
        timestamp=now.strftime('%m-%d-%y_%H:%M:%S')
        self.csv_file_name = "nna_labels_"+username+"_"+timestamp

        self.mp3_splitted_files = listdir(str(self.samples_dir))

        self.mp3_splitted_files = [Path(f) for f in self.mp3_splitted_files if ".mp3" in f[-4:].lower()]
        if self.model_tags:
            self.mp3_splitted_files = [f for f in self.mp3_splitted_files if model_tags.get(f.name,None)!=None]



        random.shuffle(self.mp3_splitted_files)

        self.current_mp3=self.mp3_splitted_files.pop()
        self.labels=load_labels()


        suggested_tags=self.get_AI_tags()
        # debug_view = widgets.Output(layout={'border': '1px solid black'})
        # display(debug_view)
        text=widgets.Text(
            value=None,
            placeholder='Other tags (coma seperated)',
            description='',
            disabled=False
        )
        save_button=widgets.Button(
            value=False,
            description='Save',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Description',
            icon='check'
        )

        self.items={name+"_static_TagButton":widgets.Button(
                                    value=False,
                                    description=name,
                                    disabled=False,
                                    icon="square",
                                    layout=widgets.Layout(width='auto')) for name in self.tags}

        for tag in suggested_tags:
            self.items[tag+"_Predicted_TagButton"]=widgets.Button(
                                                        value=False,
                                                        description=tag,
                                                        disabled=False,
                                                        icon="square",
                                                        layout=widgets.Layout(width='auto'))


        self.items["extra_Text"]=text
        self.items["save_Button"]=save_button

        static_grid_list=[ value for key,value in self.items.items() if "_static_TagButton" in key ]
        static_grid=widgets.GridBox(static_grid_list, layout=widgets.Layout(grid_template_columns="repeat(3, 200px)"))

        predicted_grid_list=[ value for key,value in self.items.items() if "_Predicted_TagButton" in key ]
        predicted_grid=widgets.GridBox(predicted_grid_list, layout=widgets.Layout(grid_template_columns="repeat(3, 200px)"))
        self.predicted_explanation=widgets.HTML(
            value=" <b>Computer generated labels, might be unrelated</b>",
            placeholder='',
            description='',
        )

        self.items["mp3_output"] = widgets.Output()
        self.items["predicted_TagsOutput"] = widgets.Output()
        self.items["static_TagsOutput"] = widgets.Output()

        with self.items["static_TagsOutput"]:
            clear_output()
            display(static_grid)


        with self.items["predicted_TagsOutput"]:
            clear_output()
            display(self.predicted_explanation)
            display(predicted_grid)


        display(self.items["static_TagsOutput"])
        display(self.items["extra_Text"])
        if self.model_tags:
            display(self.items["predicted_TagsOutput"])
        display(self.items["save_Button"])

        display(self.items["mp3_output"])

        self.items["save_Button"].on_click(self.save_button_func)
        for checkbox in self.items.keys():
            if "TagButton" in checkbox:
                self.items[checkbox].on_click(self.my_event_handler2)

        self.items["question_answered"]=False

        play_html_modify(self.samples_dir / self.current_mp3,self.items)
#         print(self.samples_dir / self.current_mp3)
    # @debug_view.capture(clear_output=True)
    def save_button_func(self,btn_object):
        csv_input=[str(self.current_mp3.name)]
        for checkbox in self.items.keys():
            if "TagButton" in checkbox:
                if self.items[checkbox].icon=="check-square":
                    csv_input.append(self.items[checkbox].description)
        if self.items["extra_Text"].value:
            extra_Text_values=self.items["extra_Text"].value.split(",")
            csv_input.extend(extra_Text_values)
#         csv_input=",".join(csv_input)
        for checkbox in self.items.keys():
            if "TagButton" in checkbox:
                self.items[checkbox].value=False
                self.items[checkbox].icon="square"
        self.items["extra_Text"].value=""
        self.items["save_Button"].icon="square"
        previous = self.current_mp3
        self.current_mp3 = self.mp3_splitted_files.pop()
        suggested_tags=self.get_AI_tags()
        self.update_suggested_tags(suggested_tags)
        play_html_modify( self.samples_dir / self.current_mp3,self.items)

        # print(csv_input,self.current_mp3)
        if not self.TEST_MODE:
#             Path(previous).unlink()
            save_to_csv(Path(self.RESULTS_DIR) / self.csv_file_name,[csv_input])


    def my_event_handler2(self,btn_object):
        if btn_object.icon=="check-square":
            btn_object.icon="square"
        else:
            btn_object.icon="check-square"

    # @debug_view.capture(clear_output=True)
    def get_AI_tags(self):
        # get AI generated labels
        original,audioop=self.model_tags.get(str(self.current_mp3.name),(None,None))
        if (original,audioop) == (None,None):
            return []
        suggested_tags_index,suggested_tags_prob=(original[0]+audioop[0],original[1]+audioop[1])
        suggested_tags=set()
        for tag_index,prob in zip(suggested_tags_index,suggested_tags_prob):
            if prob>self.tag_threshold:
                tag=self.labels[tag_index]
                suggested_tags.add(tag)

        suggested_tags=sorted(list(suggested_tags-set(self.tags)))
#         suggested_tags_dict={} # [("speech",0.2)]
#         suggested_tags_set=set()
#         for tag_index,prob in zip(suggested_tags_index,suggested_tags_prob):
#             if prob>self.tag_threshold:
#                 tag=self.labels[tag_index]
#                 if tag not in suggested_tags_set:
#                     suggested_tags_dict[tag]=prob
#                     suggested_tags_set.add(tag)
#                 else:
#                     suggested_tags_dict[tag]= prob if prob>suggested_tags_dict[tag] else suggested_tags_dict[tag]
#         suggested_tags=suggested_tags_dict.items()

        return suggested_tags

    # @debug_view.capture(clear_output=True)
    def update_suggested_tags(self,suggested_tags):
        if not suggested_tags:
            return None
        old_suggested_tags=[ key for key,value in self.items.items() if "_Predicted_TagButton" in key ]
        for key in old_suggested_tags:
            del self.items[key]
        # print(suggested_tags)
        for tag in suggested_tags:
            self.items[tag+"_Predicted_TagButton"]=widgets.Button(
                                                        value=False,
                                                        description=tag,
                                                        disabled=False,
                                                        icon="square",
#                                                         style={'description_width': 'initial'},

                                                        layout=widgets.Layout(width='auto')
                                                                    )
        for checkbox in self.items.keys():
            if "_Predicted_TagButton" in checkbox:
                self.items[checkbox].on_click(self.my_event_handler2)

        predicted_grid_list=[ value for key,value in self.items.items() if "_Predicted_TagButton" in key ]

        predicted_grid=widgets.GridBox(predicted_grid_list, layout=widgets.Layout(grid_template_columns="repeat(3, 200px)"))
        with self.items["predicted_TagsOutput"]:
            clear_output()
            display(self.predicted_explanation)
            display(predicted_grid)
