'''Module handling importing data from external sources.

'''
from typing import Dict, Union

from pathlib import Path
from collections.abc import MutableMapping


class Audio():
    """Single audio sample within the dataset.
    """
    def __init__(
        self,
        file_path: Union[str, Path],
        length_seconds: float,
    ):
        self.path = Path(file_path)
        self.name = self.path.name
        self.suffix = self.path.suffix
        self.length = length_seconds  # in seconds
        self.taxo_code = None

    def __str__(self,):
        return str(self.name)

    def __repr__(self,):
        return f'{self.path}, length:{self.length}'


class Dataset(MutableMapping):
    """A dictionary that holds data points."""

    def __init__(self,excerpt_len=10, *args, **kwargs):
        self.store = dict()
        self.update(dict(*args, **kwargs))  # use the free update to set keys
        self.excerpt_length = excerpt_len # in seconds

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
    def edges(self,value):
        if self._init_end:
            raise NotImplementedError('Edges and taxonomy are Immutable')
        else:
            self._edges = value
    @edges.getter
    def edges(self,):
        return self._edges

    def __getitem__(self, key):
        key = self._keytransform(key)
        if isinstance(key,list):
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
        if isinstance(key,str):
            return key.split('.')
        elif isinstance(key,list):
            return key
        return key

    def flatten(self,d):
        out = {}
        for key, val in d.items():
            if isinstance(val, dict):
                deeper = self.flatten(val).items()
                out.update(deeper)
            else:
                out[key] = val
        return out

    def shorten_edge_keys(self,d):
        for key, val in list(d.items()):
            del d[key]
            if isinstance(val, dict):
                d[key.split('.')[-1]] = self.shorten_edge_keys(val)
            else:
                d[key.split('.')[-1]]=val
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
            'geo': 'X.X.X',
            'grouse': '1.1.8',
            'loon': '1.1.3',
            'mam': '1.2.0',
            'plane': '0.2.0',
            'ptarm': '1.1.8',
            'rain': 'X.X.X',
            'seab': '1.1.5',
            'silence': 'X.X.X',
            'songbird': '1.1.10',
            'unknown': 'X.X.X',
            'water': 'X.X.X',
            'x': 'X.X.X',
        }
    if row['Specific Category'] in ['Songb', 'SongB']:
        row['Specific Category'] = 'Songbird'

    if row['Category'] in ['Mamm']:
        row['Category'] = 'Mam'

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

    if code[2] != 'X':
        yaml_code = excell_names2code[code[2].lower()]
    elif code[1] != 'X':
        yaml_code = excell_names2code[code[1].lower()]
    elif code[0] != 'X':
        yaml_code = excell_names2code[code[0].lower()]
    else:
        print(code)
        raise ValueError(f'row does not belong to any toplogy: {row}')

    return yaml_code


