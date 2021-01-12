'''Module handling importing data from external sources.

'''
from typing import Dict, Union, Optional, Type

from pathlib import Path
from collections.abc import MutableMapping

import numpy as np
import nna


class Audio():
    """Single audio sample within the dataset.
    """

    def __init__(
            self,
            file_path: Union[str, Path],
            length_seconds: float,
            taxo_code: Optional[str] = None,
            clipping=None,  # Optional[np.array] 
            # shape = (# of 10 seconds,number of channels)
    ):
        self.path = Path(file_path)
        self.name = self.path.name
        self.suffix = self.path.suffix
        self.length = length_seconds  # in seconds
        self.taxo_code = taxo_code
        self.clipping = clipping
        self.data = np.empty(0)  # suppose to be np.array
        self.sr: Optional[int] = None  # sampling rate

    def __str__(self,):
        return str(self.name)

    def __repr__(self,):
        return f'{self.path}, length:{self.length}'

    def pick_channel_by_clipping(self, excerpt_length):
        if len(self.data.shape) == 1:
            return None
        cleaner_channel_indexes = np.argmin(self.clipping, axis=1)
        new_data = np.empty(self.data.shape[-1], dtype=self.data.dtype)

        excpert_len_jump = self.sr * excerpt_length

        for ch_i, data_i in zip(cleaner_channel_indexes,
                                range(0, self.data.shape[-1],
                                      excpert_len_jump)):
            new_data[data_i:data_i +
                     excpert_len_jump] = self.data[ch_i, data_i:data_i +
                                                   excpert_len_jump]

        self.data = new_data[:]


class Dataset(MutableMapping):
    """A dictionary that holds data points."""

    def __init__(self,
                 dataset_name_v='',
                 excerpt_len=10,
                 dataset_folder='',
                 data_dict=None):
        self.store = dict()
        if data_dict is not None:
            self.update(dict(**data_dict))  # use the free update to set keys
        self.excerpt_length = excerpt_len  # in seconds
        self.name_v = dataset_name_v
        if dataset_folder == '':
            self.dataset_folder = ''
        else:
            self.dataset_folder = Path(dataset_folder)

    def __getitem__(self, key):
        return self.store[self._keytransform(key)]

    def __setitem__(self, key, value):
        self.store[self._keytransform(key)] = value

    def __delitem__(self, key):
        del self.store[self._keytransform(key)]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def _keytransform(self, key):
        return key

    def dataset_clipping_percentage(self, output_folder='') -> tuple:
        """for given dataset_name_version, calculate clipping info

            check if there is already a previous calculation and load that
            ex format "output/megan_1,0.pkl"

        """
        if output_folder == '':
            output_folder = self.dataset_folder

        dict_output_path = Path(output_folder) / (self.name_v + '_1,0.pkl')
        clipping_error_path = Path(output_folder) / (self.name_v +
                                                     '_1,0_error.pkl.pkl')

        if dict_output_path.exists():
            clipping_results = np.load(dict_output_path, allow_pickle=True)[()]
            if clipping_error_path.exists():
                clipping_errors = np.load(clipping_error_path,
                                          allow_pickle=True)[()]
            else:
                clipping_errors = []
            return clipping_results, clipping_errors
        else:
            msg = ((f'Could not find clipping info at {dict_output_path} ' +
                    'calculating.'))
            print(msg)
            path_list = []
            for key in self.store:
                path_list.append(self.store[key].path)

            all_results_dict, files_w_errors = nna.clippingutils.run_task_save(  # type: ignore
                path_list, self.name_v, output_folder, 1.0)
            return all_results_dict, files_w_errors

    def update_samples_w_clipping_info(self, output_folder=''):

        all_results_dict, files_w_errors = self.dataset_clipping_percentage(
            output_folder)
        del files_w_errors
        for key in self.store:
            clipping = all_results_dict.get(str(self.store[key].path), None)
            if clipping is not None:
                self.store[key].clipping = clipping

    def load_audio_files(
        self,
        cached_dict_path=None,
        dtype=np.int16,
    ):
        if cached_dict_path is not None:
            print("loading from cache at {}".format(cached_dict_path))
            cached_dict = np.load(cached_dict_path, allow_pickle=True)[()]
        else:
            print('no cache found, loading original files')
            cached_dict = {}
        for key, value in self.store.items():
            data = cached_dict.get(str(value.path), None)
            if data is None:
                sound_array, sr = nna.clippingutils.load_audio(value.path,
                                                               dtype=dtype,
                                                               backend='pydub')
            else:
                sound_array, sr = data
            self.store[key].data = sound_array
            self.store[key].sr = sr

    def pick_channel_by_clipping(self):
        for _, v in self.store.items():
            v.pick_channel_by_clipping(self.excerpt_length)

    def create_cache_pkl(self, output_file_path):
        '''save data files of samples as pkl.
        '''
        data_dict = {}
        if Path(output_file_path).exists():
            raise ValueError(f'{output_file_path} already exists')
        for value in self.store.values():
            data_dict[str(value.path)] = value.data, value.sr

        with open(output_file_path, 'wb') as f:
            np.save(f, data_dict)


# taxonomy YAML have an issue that leafes has a different structure then previous
# orders, I should change that.
class Taxonomy(MutableMapping):
    """A dictionary that holds taxonomy structure.

    transforms edge keys from x.y.z to just last bit z 
    
    """

    def __init__(self, *args, **kwargs):
        self._init_end = False
        self._store = dict()
        self.update(dict(*args, **kwargs))  # use the free update to set keys
        self._edges = self.flatten(self._store)
        self.shorten_edge_keys(self._store)
        self._init_end = True

    @property
    def edges(self,):
        """Property of _edges."""
        return self._edges

    @edges.setter
    def edges(self, value):
        if self._init_end:
            raise NotImplementedError('Edges and taxonomy are Immutable')
        else:
            self._edges = value

    @edges.getter
    def edges(self,):
        return self._edges

    def __getitem__(self, key):
        key = self._keytransform(key)
        if isinstance(key, list):
            data = self._store
            for k in key:
                data = data[k]
            return data
        return self._store[key]

        # trying to implement general access by single key or multiple with dot
        # current_order = self._store[key[0]]
        # if len(key)==1:
        #     return current_order
        # keys = self._store[self._keytransform(key)]
        # for k in keys[:-1]:
        #     current_order = current_order[k]

        # return current_order[key]

    def __setitem__(self, key, value):
        if self._init_end:
            raise NotImplementedError('You cannot update after initilization.')
        else:
            self._store[key] = value

    def __delitem__(self, key):
        if self._init_end:
            raise NotImplementedError('You cannot update after initilization.')
        else:
            del self._store[key]
        # del self._store[self._keytransform(key)]
        # self.edges = self.flatten(self._store)

    def __iter__(self):
        return iter(self._store)

    def __len__(self):
        return len(self._store)

    def _keytransform(self, key):
        if isinstance(key, str):
            return key.split('.')
        elif isinstance(key, list):
            return key
        return key

    def flatten(self, d):
        out = {}
        for key, val in d.items():
            if isinstance(val, dict):
                deeper = self.flatten(val).items()
                out.update(deeper)
            else:
                out[key] = val
        return out

    def shorten_edge_keys(self, d):
        for key, val in list(d.items()):
            del d[key]
            if isinstance(val, dict):
                d[key.split('.')[-1]] = self.shorten_edge_keys(val)
            else:
                d[key.split('.')[-1]] = val
        return d


# tax={'0': {'0.0': {'0.0.0': ['other-anthrophony']},
#   '0.1': {'0.1.0': ['other-car'],
#    '0.1.1': ['truck'],
#    '0.1.X': ['unknown-car']},
#   '0.2': {'0.2.0': ['other-aircraft'],
#    '0.2.1': ['propeller-plane'],
#    '0.2.2': ['helicopter'],
#    '0.2.3': ['commercial-plane'],
#    '0.2.X': ['unknown-aircraft']},
#   '0.3': {'0.3.0': ['other-machinery'], '0.3.X': ['unknown-machinery']}},
#  '1': {'1.0': {'1.0.0': ['other-biophony']},
#   '1.1': {'1.1.0': ['other-bird'],
#    '1.1.1': ['hummingbirds'],
#    '1.1.2': ['shorebirds'],
#    '1.1.3': ['loons'],
#    '1.1.4': ['raptors-falcons'],
#    '1.1.5': ['seabirds'],
#    '1.1.6': ['owls'],
#    '1.1.7': ['duck-goose-swan'],
#    '1.1.8': ['grouse-ptarmigan'],
#    '1.1.9': ['woodpecker'],
#    '1.1.10': ['songbirds'],
#    '1.1.11': ['cranes'],
#    '1.1.X': ['unknown-bird']},
#   '1.2': {'1.2.0': ['other-mammal'],
#    '1.2.1': ['rodents'],
#    '1.2.2': ['ursids'],
#    '1.2.3': ['cervids'],
#    '1.2.4': ['canids'],
#    '1.2.5': ['mustelids'],
#    '1.2.6': ['felids'],
#    '1.2.7': ['lagomorphs'],
#    '1.2.8': ['shrews'],
#    '1.2.X': ['unknown-mammal']},
#   '1.3': {'1.3.0': ['other-insect'],
#    '1.3.1': ['mosquito'],
#    '1.3.2': ['fly'],
#    '1.3.3': ['bee'],
#    '1.3.X': ['unknown-insect']},
#   '1.X': {'1.X.X': ['unknown-biophony']}},
#  '2': {'2.0': {'2.0.0': ['other-geology']},
#   '2.1': {'2.1.0': ['other-rain'], '2.1.X': ['unknown-rain']},
#   '2.2': {'2.2.0': ['other-water'], '2.2.X': ['unknown-water']},
#   '2.X': {'2.X.X': ['unknown-geology']}},
#  'X': {'X.X': {'X.X.X': ['unknown-sound']}}}

# from pprint import pprint
# t = Taxonomy(tax)
# pprint(len(t.edges))
# pprint(t.edges)
# pprint(list(t.items()))


def megan_excell_row2yaml_code(row: Dict, excell_names2code: Dict = None):
    '''Megan style labels to nna yaml topology V1.

    Row is a mapping, with 3 topology levels, function starts from most specific
    category and goes to most general one, when a mapping is found, returns
    corresponding code such as 0.2.0 for plane.

    Args:
        row = dictinary with following keys
                'Anthro/Bio','Category','Specific Category'
        excell_names2code = mapping from names to topology code

    '''
    if excell_names2code is None:
        excell_names2code = {
            'anth': '0.0.0',
            'auto': '0.1.0',
            'bio': '1.0.0',
            'bird': '1.1.0',
            'bug': '1.3.0',
            'dgs': '1.1.7',
            'flare': '0.4.0',
            'fox': '1.2.4',
            'geo': '2.0.0',
            'grouse': '1.1.8',
            'loon': '1.1.3',
            'mam': '1.2.0',
            'plane': '0.2.0',
            'ptarm': '1.1.8',
            'rain': '2.1.0',
            'seab': '1.1.5',
            'silence': '3.0.0',
            'songbird': '1.1.10',
            'unknown': 'X.X.X',
            'water': '2.2.0',
            'x': 'X.X.X',
        }
    if row['Specific Category'] in ['Songb', 'SongB']:
        row['Specific Category'] = 'Songbird'

    if row['Category'] in ['Mamm']:
        row['Category'] = 'Mam'
    # 'S4A10288_20190729_033000_unknown.wav', # 'Anthro/Bio': 'Uknown', no other data

    if row['Anthro/Bio'] in ['Uknown', 'Unknown']:
        row['Anthro/Bio'] = ''

    code = [row['Anthro/Bio'], row['Category'], row['Specific Category']]

    # place X for unknown topology
    # '0' is reserved for 'other'
    code = [i if i != '' else 'X' for i in code]

    # place X for unknown topology
    # '0' is reserved for 'other'
    # print(code)
    code = [i if i != '' else 'X' for i in code]
    for c in code:
        if '/' in c:
            raise NotImplementedError(
                f"row has wrong info about categories, '/' found: {row}")

    if code == ['X', 'X', 'X']:
        yaml_code = 'X.X.X'
    elif code[2] != 'X':
        yaml_code = excell_names2code[code[2].lower()]
    elif code[1] != 'X':
        yaml_code = excell_names2code[code[1].lower()]
    elif code[0] != 'X':
        yaml_code = excell_names2code[code[0].lower()]
    else:
        print(code)
        raise ValueError(f'row does not belong to any toplogy: {row}')
    return yaml_code
