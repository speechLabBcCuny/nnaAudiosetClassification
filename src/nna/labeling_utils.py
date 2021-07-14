import subprocess
import ipywidgets as widgets
from IPython.display import Audio, display, HTML, clear_output
import random
from pathlib import Path

import time
import datetime
from ipywidgets import Output

from os import listdir
import os
import re
import csv

from .fileUtils import save_to_csv

import numpy as np
import matplotlib.pyplot as plt

# mp3_file_path=f[0]


def ffmpeg_split_mp3(
    mp3_file_path,
    ss,
    to,
    tmpfolder="./tmp/",
):
    import sys
    tmpfolder = Path(tmpfolder)
    tmpfolder.mkdir(parents=True, exist_ok=True)

    # TODO handle stderror
    # -y (global)
    #     Overwrite output files without asking
    # -i url (input)
    #     input file url
    # -c[:stream_specifier] copy (output only) to indicate that the stream is not to be re-encoded.
    # -map 0 map all streams from input to output.
    file_extension = str(Path(mp3_file_path).suffix)
    # print(file_extension)
    # TODO make this a global variable
    ffmpeg_path = '/scratch/enis/conda/envs/speechEnv/bin/ffmpeg'
    out_file_path = str(tmpfolder / ("output" + file_extension))
    command_list = [
        ffmpeg_path, '-y', '-i', "{}".format(str(mp3_file_path)), "-ss",
        str(ss), "-to",
        str(to), "-map", "0", "-c", "copy", out_file_path
    ]
    #     print(command_list)
    sp = subprocess.run(command_list, capture_output=True)
    # print(" ".join(command_list))
    #     mp3_segments=os.listdir(segments_folder)

    if sp.returncode != 0:
        print('Error: ' + sp.stderr.decode('ascii'))
        print('Output: ' + sp.stdout.decode('ascii'))

    sys.stdout.flush()
    sys.stderr.flush()


def mp3_split(mp3_path, start_second=10, end_second=20):
    sample_count = 0
    import audioread
    with audioread.audio_open(mp3_path) as f:
        # print(f.channels, f.samplerate, f.duration)
        #               sample_rate,channel, 2 bits
        bits_persecond = f.samplerate * f.channels * 2
        start_segment = start_second * bits_persecond
        end_segment = ((end_second - 1) * bits_persecond)
        mp3array = []
        # print(start_segment,end_segment)
        for buf in f:
            start_buf = sample_count
            end_buf = sample_count + len(buf)
            indexes = (max(start_buf,
                           start_segment), min(end_buf, end_segment))
            # buffer did not reach to segment
            if end_buf < start_segment:
                # print(end_buf,start_segment)
                pass
            # buffer passed segment stop
            elif end_segment < start_buf:
                break
            # print(indexes,sample_count)
            # print(indexes[0]-sample_count,indexes[1]-sample_count)
            len_buf = len(buf)
            start = indexes[0] - sample_count
            end = indexes[1] - sample_count
            start = 0 if start < 0 else start
            end = 0 if end < 0 else end
            buf = buf[start:end]
            sample_count += len_buf
            mp3array.append(buf)
    mp3_string = b"".join(mp3array)
    return mp3_string, f.channels, f.samplerate, f.duration


def stem_set(files):
    if type(files) == list:
        mp3files = files[:]
    else:
        files = Path(files)
        with open(files, "r") as mp3files:
            mp3files = mp3files.readlines()
            mp3files = [i.strip() for i in mp3files]

    ignored = []
    mp3filesset = list()
    mp3filestemset = []
    for file in mp3files:
        if Path(file).stem not in mp3filestemset:
            mp3filestemset.append(Path(file).stem)
            mp3filesset.append(file)
        else:
            ignored.append(file)
    mp3files = mp3filesset[:]

    return mp3files, ignored


def cut_random_file(input_mp3_file,
                    length=10,
                    split_folder="./splits",
                    total_minute=49 * 60,
                    depth=0,
                    backend="ffmpeg",
                    backend_path=''):

    start_minute = random.randint(0, total_minute)
    start_second = random.randint(0, 59)

    extra_minute = (start_second + 10) // 60

    end_minute = start_minute + extra_minute
    end_second = (start_second + length) % 60

    start_time = "{}.{}".format(start_minute, start_second)
    end_time = "{}.{}".format(end_minute, end_second)
    #     print(input_mp3_file,start_time,end_time)

    result = splitmp3(str(input_mp3_file),
                      split_folder,
                      start_time,
                      end_time,
                      backend=backend,
                      backend_path=backend_path)

    if result == 0 and depth < 3:
        print(input_mp3_file, start_time, end_time)
        # cut_random_file(mp3_files_path_list,length=length,split_folder=split_folder,
        # longest_minute=longest_minute,depth=depth+1)

    else:
        return result


def splitmp3(input_mp3_file,
             split_folder,
             start_time,
             end_time,
             depth=5,
             backend="ffmpeg",
             backend_path='',
             outputSuffix=None):
    # -f increases precision (ONLY mp3)
    # -t
    # -d folder
    # input
    # end time minute.seconds
    # start_time
    if backend_path=='':
        backend_path=backend
    
    if backend == "mp3splt":
        cmd = [
            backend_path, '-f', '-d', split_folder, input_mp3_file, start_time,
            end_time
        ]
    elif backend == "ffmpeg":
        start_minute, start_second = start_time.split(".")
        start_time = (int(start_minute) * 60) + int(start_second)
        end_minute, end_second = end_time.split(".")
        end_time = (int(end_minute) * 60) + int(end_second)
        wholepath = Path(input_mp3_file)
        if outputSuffix == None:
            outputSuffix = wholepath.suffix
        output_file = Path(split_folder) / (
            wholepath.stem + "_" + start_minute + "m_" + start_second + "s__" +
            end_minute + "m_" + end_second + "s" + outputSuffix)
        cmd = [
            # 'conda', 'run', '-n', 'speechEnv', 'ffmpeg', '-strict', '-2',
            backend_path, '-strict', '-2',
            '-ss',
            str(start_time), '-t',
            str(end_time - start_time), "-i",
            str(input_mp3_file),
            str(output_file)
        ]
    else:
        raise Exception(
            "{} is not supported as backend, available ones are mp3splt and ffmpeg"
            .format(backend))
        
    # custom name (@f_@n+@m:@s+@M:@S)
    # cmd+=["-o","temp"+str(file_index)]

    proc = subprocess.Popen(cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)

    o, e = proc.communicate()

    if proc.returncode != 0:
        print("---------")
        print(cmd)
        print('Output: ' + o.decode('ascii'))
        print('Error: ' + e.decode('ascii'))
        return 0
    else:
        #         print('Output: ' + o.decode('ascii'))
        if backend == "mp3splt":
            split_file = re.search('File "(.*)"', o.decode('ascii')).group(1)
            split_file = Path(split_file)
        elif backend == "ffmpeg":
            split_file = output_file
        else:
            split_file = None


#         print(split_file)
        return split_file


def play_html_modify(mp3file, items):
    out = items["mp3_output"]
    import librosa as librosa
    import librosa.display


    with out:
        clear_output()
        # displayed=display(HTML("<audio controls  loop autoplay><source src={} type='audio/{}'></audio>".format(mp3file,mp3file.suffix[1:])))
        displayed = display(
            HTML(
                "<audio autoplay controls loop src={} preload=\"auto\" width=100% style=\"width: 120%;\"> </audio>"
                .format(mp3file)))
        y, sr = librosa.load(str(mp3file))

        S = librosa.feature.melspectrogram(y, sr=sr)
        S_DB = librosa.power_to_db(S, ref=np.max)

        fig, ax = plt.subplots(figsize=(32, 8))
        img = librosa.display.specshow(S_DB,
                                       sr=sr,
                                       x_axis='time',
                                       y_axis='mel',
                                       ax=ax)
        fig.colorbar(img, format='%+2.0f dB', ax=ax)


def load_labels(csv_file="assets/class_labels_indices.csv"):
    import csv
    if os.path.exists(csv_file):
        csvfile = open(str(csv_file), newline='')
        csv_lines = csvfile.readlines()
        csvfile.close()
    else:
        import requests
        url = "https://raw.githubusercontent.com/qiuqiangkong/audioset_classification/master/metadata/class_labels_indices.csv"
        with requests.Session() as s:
            download = s.get(url)
            decoded_content = download.content.decode('utf-8')
            csv_lines = decoded_content.splitlines()
    labels = []
    reader = csv.reader(csv_lines, delimiter=',')
    headers = next(reader)
    for row in reader:
        labels.append(row[2])
    return labels


class labeling_UI:
    def __init__(self,
                 tags,
                 samples_dir="./",
                 username="",
                 RESULTS_DIR="./",
                 TEST_MODE=False,
                 model_tags={},
                 tag_threshold=0.1,
                 clippingFile=None):
        self.TEST_MODE = TEST_MODE
        self.tag_threshold = tag_threshold
        self.model_tags = model_tags
        self.samples_dir = samples_dir
        self.tags = tags
        self.RESULTS_DIR = RESULTS_DIR

        now = datetime.datetime.now()
        timestamp = now.strftime('%m-%d-%y_%H:%M:%S')
        self.csv_file_name = "nna_labels_" + username + "_" + timestamp

        self.mp3_splitted_files = listdir(str(self.samples_dir))

        self.mp3_splitted_files = [
            Path(f) for f in self.mp3_splitted_files
            if (".mp3" in f[-4:].lower() or ".flac" in f[-5:].lower())
        ]
        if self.model_tags:
            self.mp3_splitted_files = [
                f for f in self.mp3_splitted_files
                if model_tags.get(f.name, None) != None
            ]

        random.shuffle(self.mp3_splitted_files)

        self.current_mp3 = self.mp3_splitted_files.pop()
        self.labels = load_labels()

        if clippingFile != None:
            self.clippingDict = self.loadClippingData(clippingFile)
        else:
            self.clippingDict = None

        suggested_tags = self.get_AI_tags()
        # debug_view = widgets.Output(layout={'border': '1px solid black'})
        # display(debug_view)
        text = widgets.Text(value=None,
                            placeholder='Other tags (coma seperated)',
                            description='',
                            disabled=False)
        save_button = widgets.Button(
            value=False,
            description='Save',
            disabled=False,
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Description',
            icon='check')

        self.items = {
            name + "_static_TagButton":
            widgets.Button(value=False,
                           description=name,
                           disabled=False,
                           icon="square",
                           layout=widgets.Layout(width='auto'))
            for name in self.tags
        }

        for tag in suggested_tags:
            self.items[tag + "_Predicted_TagButton"] = widgets.Button(
                value=False,
                description=tag,
                disabled=False,
                icon="square",
                layout=widgets.Layout(width='auto'))

        self.items["extra_Text"] = text
        self.items["save_Button"] = save_button

        static_grid_list = [
            value for key, value in self.items.items()
            if "_static_TagButton" in key
        ]
        static_grid = widgets.GridBox(
            static_grid_list,
            layout=widgets.Layout(grid_template_columns="repeat(3, 200px)"))

        predicted_grid_list = [
            value for key, value in self.items.items()
            if "_Predicted_TagButton" in key
        ]
        predicted_grid = widgets.GridBox(
            predicted_grid_list,
            layout=widgets.Layout(grid_template_columns="repeat(3, 200px)"))
        self.predicted_explanation = widgets.HTML(
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

        self.items["question_answered"] = False

        play_html_modify(self.samples_dir / self.current_mp3, self.items)
#         print(self.samples_dir / self.current_mp3)
# @debug_view.capture(clear_output=True)

    def save_button_func(self, btn_object):
        csv_input = [str(self.current_mp3.name)]
        for checkbox in self.items.keys():
            if "TagButton" in checkbox:
                if self.items[checkbox].icon == "check-square":
                    csv_input.append(self.items[checkbox].description)
        if self.items["extra_Text"].value:
            extra_Text_values = self.items["extra_Text"].value.split(",")
            csv_input.extend(extra_Text_values)


#         csv_input=",".join(csv_input)
        for checkbox in self.items.keys():
            if "TagButton" in checkbox:
                self.items[checkbox].value = False
                self.items[checkbox].icon = "square"
        self.items["extra_Text"].value = ""
        self.items["save_Button"].icon = "square"
        previous = self.current_mp3
        self.current_mp3 = self.mp3_splitted_files.pop()
        suggested_tags = self.get_AI_tags()
        self.update_suggested_tags(suggested_tags)
        play_html_modify(self.samples_dir / self.current_mp3, self.items)

        # print(csv_input,self.current_mp3)
        if not self.TEST_MODE:
            #             Path(previous).unlink()
            save_to_csv(
                Path(self.RESULTS_DIR) / self.csv_file_name, [csv_input])

    def my_event_handler2(self, btn_object):
        if btn_object.icon == "check-square":
            btn_object.icon = "square"
        else:
            btn_object.icon = "check-square"

    # @debug_view.capture(clear_output=True)
    def get_AI_tags(self):
        # get AI generated labels
        original, audioop = self.model_tags.get(str(self.current_mp3.name),
                                                (None, None))
        if (original, audioop) == (None, None):
            return []
        suggested_tags_index, suggested_tags_prob = (original[0] + audioop[0],
                                                     original[1] + audioop[1])
        suggested_tags = set()
        for tag_index, prob in zip(suggested_tags_index, suggested_tags_prob):
            if prob > self.tag_threshold:
                tag = self.labels[tag_index]
                suggested_tags.add(tag)

        suggested_tags = sorted(list(suggested_tags - set(self.tags)))
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
    def update_suggested_tags(self, suggested_tags):
        if not suggested_tags:
            return None
        old_suggested_tags = [
            key for key, value in self.items.items()
            if "_Predicted_TagButton" in key
        ]
        for key in old_suggested_tags:
            del self.items[key]
        # print(suggested_tags)
        for tag in suggested_tags:
            self.items[tag + "_Predicted_TagButton"] = widgets.Button(
                value=False,
                description=tag,
                disabled=False,
                icon="square",
                #                                                         style={'description_width': 'initial'},
                layout=widgets.Layout(width='auto'))
        for checkbox in self.items.keys():
            if "_Predicted_TagButton" in checkbox:
                self.items[checkbox].on_click(self.my_event_handler2)

        predicted_grid_list = [
            value for key, value in self.items.items()
            if "_Predicted_TagButton" in key
        ]

        predicted_grid = widgets.GridBox(
            predicted_grid_list,
            layout=widgets.Layout(grid_template_columns="repeat(3, 200px)"))
        with self.items["predicted_TagsOutput"]:
            clear_output()
            display(self.predicted_explanation)
            display(predicted_grid)

    def loadClippingData(self, clippingFile):
        import numpy as np
        clippingDict = np.load(clippingFile)
        return clippingDict
