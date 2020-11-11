"""Utilities for clipping detection in audio.

'Clipping is a form of waveform distortion that occurs when an amplifier is
overdriven and attempts to deliver an output voltage or current beyond its
maximum capability.'(wikipedia)
Clipped samples of the audio signal does not carry any information.
We assume clipping happens when sample's value is +1 or -1 (threshold).


    Typical usage example:
    Function run_task_save combines load_audio and get_clipping_percent.
    ```[python]
    # file info
    file_properties_df_path = "../nna/data/prudhoeAndAnwr4photoExp_dataV1.pkl"
    file_properties_df = pd.read_pickle(file_properties_df_path)
    # where to save results
    clipping_results_path="./clipping_results/"

    for i,area in enumerate(all_areas[:10]):
        print(area,i)
        area_filtered=file_properties_df[file_properties_df.site_id==area]
        run_task_save(area_filtered.index,area,clipping_results_path,
                        clipping_threshold)
    ```

"""

from typing import Union, Tuple, List
# from nna.fileUtils import list_files
from pydub import AudioSegment
# from IPython.display import display, Audio
from pathlib import Path
import librosa
import numpy as np
import pickle
# import IPython.display as ipd
# import librosa.display
# import matplotlib.pyplot as plt
# import pandas as pd
# import time


def load_audio(filepath: Union[Path, str],
               dtype: np.dtype = np.int16,
               backend: str = "pydub") -> Tuple[np.array, int]:
    """Load audio file as numpy array using given backend.

    Depending on audio reading library handles data type conversions.

    Args:
        filepath: path to the file
        dtype: Data type to store audio file.
        backend: Which backend to use load the audio.

    Returns:
        A tuple of array storing audio and sampling rate.

    """
    filepath = str(filepath)
    if backend == "librosa":
        sound_array, sr = librosa.load(filepath, mono=False, sr=None)
        # TODO: explain this part
        if dtype == np.int16:
            sound_array = sound_array * 32768
            sound_array = sound_array.astype(np.int16)
    elif backend == "pydub":
        sound_array = AudioSegment.from_file(filepath)
        sr = sound_array.frame_rate
        sound_array = np.frombuffer(
            sound_array._data,  # pylint: disable=W0212
            dtype=np.int16)
        sound_array = sound_array.reshape(-1, 2).T
        if dtype in (np.float32, np.float64):
            sound_array = sound_array.astype(np.float32)
            sound_array = (sound_array / 32768)
    else:
        print(f"no backend called {backend}")
    return sound_array, sr


def get_clipping_percent(sound_array: np.array,
                         threshold: float = 1.0) -> List[np.float64]:
    """Calculate clipping percentage comparing to (>=) threshold.

        Args:
            sound_array: a numpy array with shape of
                        (sample_count) or (2,sample_count)
            threshold: min and max values which samples are assumed to be
                      Clipped  0<=threshold<=1,

    """
    if sound_array.dtype == np.int16:
        minval = int(-32768 * threshold)
        maxval = int(32767 * threshold)
    else:
        minval = -threshold
        #librosa conversion from int to float causes missing precision
        # so we lover it sligtly
        maxval = threshold * 0.9999

    #mono
    if len(sound_array.shape) == 1:
        result = ((
            (np.sum(sound_array <= minval) + np.sum(sound_array >= maxval))) /
                  sound_array.size)
        results = [result]
    elif len(sound_array.shape) == 2:
        results = (np.sum(sound_array <= minval, axis=1) + np.sum(
            sound_array >= maxval, axis=1)) / sound_array.shape[-1]
        results = list(results)
    return results


def run_task_save(allfiles: List[str],
                  area_id: str,
                  results_folder: Union[str, Path],
                  clipping_threshold: float,
                  segment_len: int = 10,
                  audio_load_backend: str = "pydub",
                  save=True) -> Tuple[dict, list]:
    """Save clipping dict{Path:np.array} to file as f"{area_id}_{threshold}.pkl"
        Args:
            allfiles: List of files to calculate clipping.
            area_id: of the files coming from, will be used in file_name
            results_folder: where to save results if save==True
            clipping_threshold:
            segment_len: length of segments to calculate clipping per.
            audio_load_backend: backend for loading files
            save: Flag for saving results to a file.
        Returns:
            Tuple(all_results_dict ,files_w_errors)
                all_results_dict: Dict{a_file_path:np.array}
                files_w_errors: List[(index, a_file_path, exception),]

    """
    files_w_errors = []
    all_results_dict = {}
    # CALCULATE RESULTS
    for _, audio_file in enumerate(allfiles):
        # try:
        y, sr = load_audio(audio_file,
                           dtype=np.int16,
                           backend=audio_load_backend)

        assert sr == int(sr)
        sr = int(sr)
        results = []
        for clip_i in range(0, int(y.shape[-1] - segment_len),
                            int(segment_len * sr)):
            res = get_clipping_percent(y[:,
                                         clip_i:(clip_i + (segment_len * sr))],
                                       threshold=clipping_threshold)
            results.append(res)
        resultsnp = np.array(results)
        all_results_dict[audio_file] = resultsnp[:]
        # except Exception as e:  # pylint: disable=W0703
        # print(i, audio_file)
        # print(e)
        # files_w_errors.append((i, audio_file, e))
    # SAVE RESULTS
    clipping_threshold_str = str(clipping_threshold)
    clipping_threshold_str = clipping_threshold_str.replace(".", ",")
    filename = "{}_{}.pkl".format(area_id, clipping_threshold_str)
    error_filename = "{}_{}_error.pkl".format(area_id, clipping_threshold_str)
    results_folder = Path(results_folder)
    output_file_path = results_folder / filename
    error_file_path = results_folder / error_filename
    if save:
        with open(output_file_path, "wb") as f:
            np.save(f, all_results_dict)
        if files_w_errors:
            with open(error_file_path, "wb") as f:
                pickle.dump(files_w_errors,
                            f,
                            protocol=pickle.HIGHEST_PROTOCOL)

    return all_results_dict, files_w_errors
