import subprocess
from xml.dom.minidom import TypeInfo
import ipywidgets as widgets
from IPython.display import Audio, display, HTML, clear_output

import random
from pathlib import Path
from typing import Dict
import time
import datetime
from ipywidgets import Output

from os import listdir
import os
import re
import csv

import numpy as np
import matplotlib.pyplot as plt

# mp3_file_path=f[0]

import librosa
import librosa.display


def ffmpeg_split_mp3(
    audio_file_path,
    start_time,
    end_time,
    tmpfolder="./tmp/",
):

    file_extension = str(Path(audio_file_path).suffix)

    out_file_path = str(tmpfolder / ("output" + file_extension))

    ffmpeg_split_audio(audio_file_path, start_time, end_time, out_file_path)

    return out_file_path


def ffmpeg_split_audio(
    audio_file_path,
    start_time,
    end_time,
    output_file,
    backend_path='/scratch/enis/conda/envs/speechEnv/bin/ffmpeg',
    dry_run=False,
    stereo2mono=False,
    overwrite=True,
    sampling_rate=None,
):
    '''
        

    '''
    import sys
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # -y (global)
    #     Overwrite output files without asking
    # -i url (input)
    #     input file url
    # -c[:stream_specifier] copy (output only) to indicate that the stream is not to be re-encoded.
    # -map 0 map all streams from input to output.

    cmd = [
        # 'conda', 'run', '-n', 'speechEnv', 'ffmpeg', '-strict', '-2',
        backend_path,
        '-strict',
        '-2',  # allow non standardized experimental things, #TODO
        '-ss',
        str(start_time),
        '-t',
        str(end_time - start_time),
        '-i',
        str(audio_file_path),
    ]
    if stereo2mono:
        cmd.extend(['-ac', '1'])
    if overwrite:
        cmd.extend(['-y'])
    if sampling_rate is not None:
        cmd.extend(['-ar', str(sampling_rate)])
    cmd.append(str(output_file))
    output, error, _ = run_cmd(cmd, dry_run=dry_run)
    return output, error


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
            indexes = (max(start_buf, start_segment), min(end_buf, end_segment))
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


def human_readable2ffmpeg_io(audio_file,
                             start_time,
                             end_time,
                             out_folder,
                             outputSuffix=None):
    start_minute, start_second = start_time.split(".")
    start_time = (int(start_minute) * 60) + int(start_second)
    end_minute, end_second = end_time.split(".")
    end_time = (int(end_minute) * 60) + int(end_second)
    wholepath = Path(audio_file)
    if outputSuffix == None:
        outputSuffix = wholepath.suffix
    output_file = Path(out_folder) / (wholepath.stem + "_" + start_minute +
                                      "m_" + start_second + "s__" + end_minute +
                                      "m_" + end_second + "s" + outputSuffix)
    return start_time, end_time, output_file


def mp3splt_backend(audio_file,
                    start_time,
                    end_time,
                    out_folder,
                    backend_path='mp3splt',
                    dry_run=False,
                    stereo2mono=False,
                    overwrite=True,
                    sampling_rate=None):
    if stereo2mono:
        raise NotImplementedError('stereo to mono not implemented for mp3splt')
    if overwrite:
        raise NotImplementedError('overwrite not implemented for mp3splt')
    if sampling_rate is not None:
        raise NotImplementedError('sampling rate not implemented for mp3splt')
    cmd = [
        backend_path, '-f', '-d', out_folder, audio_file, start_time, end_time
    ]
    output, error, _ = run_cmd(cmd, dry_run=dry_run)
    return output, error


def splitmp3(input_mp3_file,
             out_folder,
             start_time,
             end_time,
             backend="ffmpeg",
             backend_path='',
             outputSuffix=None,
             dry_run=False,
             stereo2mono=False,
             overwrite=True,
             sampling_rate=None):
    '''
    '''
    if backend_path == '':
        backend_path = backend

    if backend == "mp3splt":
        output, error = mp3splt_backend(input_mp3_file,
                                        start_time,
                                        end_time,
                                        out_folder,
                                        backend_path=backend_path,
                                        dry_run=dry_run,
                                        stereo2mono=stereo2mono,
                                        overwrite=overwrite,
                                        sampling_rate=sampling_rate)
        output_file = re.search('File "(.*)"', output).group(1)
        output_file = Path(output_file)

    elif backend == "ffmpeg":
        start_time, end_time, output_file = human_readable2ffmpeg_io(
            input_mp3_file,
            start_time,
            end_time,
            out_folder,
            outputSuffix=outputSuffix)

        output, error = ffmpeg_split_audio(input_mp3_file,
                                           start_time,
                                           end_time,
                                           output_file,
                                           backend_path=backend_path,
                                           dry_run=dry_run,
                                           stereo2mono=stereo2mono,
                                           overwrite=overwrite,
                                           sampling_rate=sampling_rate)

    else:
        raise Exception(
            "{} is not supported as backend, available ones are mp3splt and ffmpeg"
            .format(backend))

    # custom name (@f_@n+@m:@s+@M:@S)
    # cmd+=["-o","temp"+str(file_index)]

    return output_file


def run_cmd(cmd, dry_run=False):
    if dry_run:
        return ''.join(cmd), '\n cmd not run, dry_run is True', 'no run'
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    output, error = proc.communicate()
    output = output.decode('ascii')
    error = error.decode('ascii')
    if proc.returncode != 0:
        print('---------')
        print(cmd)
        print('Output: \n' + output)
        print('Error: \n' + error)

    return output, error, proc.returncode


def play_html_modify(audiofile,
                     audiofile_local,
                     items,
                     image_file=None,
                     img_width=800,
                     img_height=800):
    out = items["mp3_output"]
    if not audiofile_local:
        audiofile_local = audiofile
    with out:
        print('current file:', audiofile_local)
        clear_output()
        # displayed=display(HTML("<audio controls  loop autoplay><source src={} type='audio/{}'></audio>".format(mp3file,mp3file.suffix[1:])))
        displayed = display(
            HTML(
                "<audio autoplay controls loop src={} preload=\"auto\" width=100% style=\"width: 100%;\"> </audio>"
                .format(audiofile)))
        # displayed=display(Audio(audiofile)) # load a local WAV file

        if image_file is None:
            y, sr = librosa.load(str(audiofile_local))

            S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128, fmax=20000)
            S_DB = librosa.power_to_db(S, ref=np.max)

            fig, ax = plt.subplots(figsize=(32, 8))
            img = librosa.display.specshow(S_DB,
                                           sr=sr,
                                           x_axis='time',
                                           y_axis='mel',
                                           ax=ax)
            fig.colorbar(img, format='%+2.0f dB', ax=ax)
            plt.show()
        else:
            # display(HTML(f"<img src={image_file} height='200'  width=95% style='background-color:#e8e4c9'/>"))
            display(
                HTML(
                    f"<img src={image_file} width='{img_width}' height='{img_height}' viewBox='0 0 100 800'  style='background-color:#e8e4c9' />"
                ))


def read_csv(csv_file_path, fieldnames=None):
    with open(csv_file_path, encoding='utf-8') as csv_file:
        rows = csv.DictReader(csv_file, fieldnames=fieldnames)
        rows = list(rows)
    return rows


def write_csv(new_csv_file, rows_new, fieldnames=None):
    with open(new_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        if fieldnames is None:
            fieldnames = rows_new[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(rows_new)


def sort_samples(sample_rows,):
    # sort by label
    # sort by location
    # sory by clip_path
    return sample_rows


def find_file_loc(sample_row, samples_dir, s3=True):
    if s3:
        return 'https://crescent.sfo3.digitaloceanspaces.com' + sample_row[
            "Clip Path"], sample_row["Clip Path"]
    else:
        samples_dir = Path(samples_dir)
        return samples_dir / sample_row["Clip Path"], samples_dir / sample_row[
            "Clip Path"]


def find_image_loc(sample_row, samples_dir, s3=True):
    clip_path_modified = sample_row["Clip Path"].replace(
        'combined7-V1', 'combined7-V1_mel').replace(".wav", ".svg")
    if s3:
        return 'https://crescent.sfo3.digitaloceanspaces.com' + clip_path_modified, sample_row[
            "Clip Path"]
    else:
        return None, None
        samples_dir = Path(samples_dir)
        return samples_dir / clip_path_modified, samples_dir / sample_row[
            "Clip Path"].replace(".wav", ".svg")


debug_view = widgets.Output(layout={'border': '1px solid black'})
display(debug_view)


class SamplesDataset():

    def __init__(self, rows, locations=None, is_random=False):
        self.rows = rows
        if is_random:
            random.shuffle(self.rows)
        self.filter_locations = locations
        self.index = 0

    def get_index(self,):
        if self.index == len(self.rows) - 1:
            self.index = 0
        else:
            self.index += 1
        return self.index

    def __getitem__(self, idx):
        return self.rows[idx]

    def is_available(self, row):
        if row["Reviewed"].lower() == "false" and (not self.filter_locations or
                                                   row["Site ID"]
                                                   in self.filter_locations):
            return True
        else:
            return False

    def __len__(self):
        count = 0
        for row in self.rows():
            if self.is_available(row):
                count += 1
        return count

    def next(self, unfiltered=True) -> Dict:
        if unfiltered:
            return self.next_unreviewed()
        else:
            return self.rows(self.get_index())

    def next_unreviewed(self) -> Dict:
        for row in self.rows:
            if self.is_available(row):
                return row
        return None  # type: ignore

    def set_reviewed(self, row, username='True'):
        if username.lower() == 'false':
            raise ValueError('username cannot be false')
        row["Reviewed"] = username
        return row


class labeling_UI:
    '''
        This class is used to label the audio files.

        Picks a random file from the given folder and displays it in a web browser.

    '''

    def __init__(
        self,
        tags,
        samples_dir="./",
        samples_csv='',
        username="",
        RESULTS_DIR="./",
        TEST_MODE=False,
        tag_threshold=0.1,
        clippingFile=None,
        is_random=False,
        dataset_name="",
        locations=[],
        box_per_row=3,
        box_length=200,
        img_height=800,
        img_width=800,
    ):

        self.TEST_MODE = TEST_MODE
        self.tag_threshold = tag_threshold
        self.samples_dir = samples_dir
        self.tags = tags
        self.RESULTS_DIR = Path(RESULTS_DIR)
        self.username = username
        self.dataset_name = dataset_name
        self.box_per_row = box_per_row
        self.box_length = box_length
        self.end = False
        self.img_height = img_height
        self.img_width = img_width
        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        now = datetime.datetime.now()
        timestamp = now.strftime('%m-%d-%y_%H:%M:%S')
        self.csv_file_name = "nna_labels_" + self.username + "_" + timestamp

        sample_rows = read_csv(samples_csv)
        sample_rows = SamplesDataset(sample_rows,
                                     locations,
                                     is_random=is_random)
        self.sample_rows = sample_rows

        self.current_audio = self.sample_rows.next()
        if self.current_audio is None:
            print('!!! No samples to label, change location if possible !!!')
            return None
        self.sample_rows.set_reviewed(self.current_audio, self.username)

        if clippingFile != None:
            self.clippingDict = self.loadClippingData(clippingFile)
        else:
            self.clippingDict = None

        self.init_UI()
        self.update_UI_with_current_audio()

    def init_UI(self):

        self.items = {}

        text = widgets.Text(value=None,
                            placeholder='Other tags (coma seperated)',
                            description='',
                            disabled=False)
        self.items["extra_Text"] = text

        save_button = widgets.Button(
            value=False,
            description='Save',
            disabled=False,
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Description',
            icon='check')
        self.items["save_Button"] = save_button

        for name in self.tags:
            button_value = False
            button_icon = "square"

            self.items[name + "_static_TagButton"] = widgets.Button(
                value=button_value,
                description=name,
                disabled=False,
                icon=button_icon,
                layout=widgets.Layout(width='auto'))

        static_grid_list = [
            value for key, value in self.items.items()
            if "_static_TagButton" in key
        ]
        static_grid = widgets.GridBox(
            static_grid_list,
            layout=widgets.Layout(
                grid_template_columns=
                f"repeat({self.box_per_row}, {self.box_length}px)"))

        self.items["mp3_output"] = widgets.Output()
        self.items["static_TagsOutput"] = widgets.Output()

        with self.items["static_TagsOutput"]:
            clear_output()
            display(static_grid)
        display(self.items["static_TagsOutput"])
        display(self.items["extra_Text"])
        display(self.items["save_Button"])

        display(self.items["mp3_output"])

        self.items["save_Button"].on_click(self.save_button_func)
        for checkbox in self.items.keys():
            if "TagButton" in checkbox:
                self.items[checkbox].on_click(self.reverse_square_box)

    def update_row_with_UI_info(self):
        labels_from_UI = {}
        for checkbox in self.items.keys():
            if "TagButton" in checkbox:
                if self.items[checkbox].icon == "check-square":
                    labels_from_UI[self.items[checkbox].description] = '1'
                elif "square":
                    labels_from_UI[self.items[checkbox].description] = '0'
                else:
                    raise ValueError("Unknown icon")

        if self.current_audio['Reviewed'].lower() != 'false':
            print(
                f'{self.current_audio["Clip Path"]} has been reviewed before!\n'
            )
            print('last reviewed by:', self.current_audio['Reviewed'], '\n')
            print('overwriting...\n')

        for key, _ in self.current_audio.items():
            if '_r' in key:
                org_value = self.current_audio[key[:-2]]
                self.current_audio[key] = labels_from_UI.get(
                    key[:-2], org_value)

        self.current_audio['extra_tags'] = self.items["extra_Text"].value
        self.sample_rows.set_reviewed(self.current_audio, self.username)

    def write_rows2csv(self,):

        now = datetime.datetime.now()
        timestamp = now.strftime('%y-%m-%d_%H-%M-%S')
        self.csv_file_name = self.dataset_name + "_" + self.username + "_" + timestamp + ".csv"

        # print(csv_input,self.current_audio)
        if not self.TEST_MODE:
            #             Path(previous).unlink()
            write_csv(self.RESULTS_DIR / self.csv_file_name,
                      self.sample_rows.rows)

    @debug_view.capture(clear_output=True)
    def save_button_func(self, btn_object):

        # set current audio as reviewed
        self.update_row_with_UI_info()
        # save previous data
        self.write_rows2csv()
        # get new sample
        self.current_audio = self.sample_rows.next()

        self.update_UI_with_current_audio()

    def update_UI_with_current_audio(self):

        # create new UI with new sample
        # if sample labeled before, tick label boxes
        for checkbox in self.items.keys():
            if "TagButton" in checkbox:
                tag_name = checkbox.split('_')[0]
                button_value = bool(self.current_audio[tag_name] == '1')
                button_icon = "check-square" if button_value else "square"
                self.items[checkbox].value = button_value
                self.items[checkbox].icon = button_icon

        # clear extra tags
        self.items["extra_Text"].value = ""
        self.items["save_Button"].icon = "square"

        if self.current_audio is None:
            print('!!! No samples to label, change location if possible !!!')
            return None

        image_file, _ = find_image_loc(self.current_audio,
                                       self.samples_dir,
                                       s3=True)

        play_html_modify(*find_file_loc(self.current_audio,
                                        self.samples_dir,
                                        s3=True),
                         self.items,
                         image_file=image_file,
                         img_height=self.img_height,
                         img_width=self.img_width)

    def reverse_square_box(self, btn_object):
        if btn_object.icon == "check-square":
            btn_object.icon = "square"
        else:
            btn_object.icon = "check-square"

    def loadClippingData(self, clippingFile):
        clippingDict = np.load(clippingFile)
        return clippingDict
