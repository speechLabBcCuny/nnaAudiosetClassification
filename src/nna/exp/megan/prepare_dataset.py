"""Process excell from megan and taxonomy.yaml to create dataset.

Exported from prepare_dataset notebook.
"""

# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from pathlib import Path
import yaml
import csv
from pprint import pprint
from copy import deepcopy
from collections import Counter

from nna import dataimport

# %%
#Parameters
sample_count_limit = 25
sample_length_limit = 10

# %%
# read path and length of megan labeled files from meganLabeledFiles_wlenV1.txt
# store them in a dataimport.dataset, keys are gonna be file_name
src_path = '/scratch/enis/data/nna/labeling/megan/AudioSamplesPerSite/'
ffp = src_path + 'meganLabeledFiles_wlenV1.txt'

audio_dataset = dataimport.Dataset()
with open(ffp, "r") as f:
    lines = f.readlines()
    lines = [i.strip().split(",") for i in lines]
    for i in lines:
        audio_dataset[i[0]] = dataimport.Audio(i[0], float(i[1]))

# path has to be inque but is file names are unique ?
assert len(set([i.name for i in audio_dataset.values()
               ])) == len(list(audio_dataset.keys()))
# change keys to file names
# TODO fix requirement for **
audio_dataset = dataimport.Dataset(
    **{i.name: i for i in audio_dataset.values()})

# %%
# Read labeled info from spreat sheet
# and remove samples with no audio file, also files given in ignore_files
resources_folder = ('/scratch/enis/archive/' +
                    'forks/cramer2020icassp/resources/')
sheetOrg = (resources_folder + 'Sheet1.csv')
sheetMine = (resources_folder + 'Sheet1(1).csv')

ignore_files = set([
    'S4A10268_20190610_103000_bio_anth.wav',  # has two topology bird/plane
])

with open(sheetOrg) as csvfile:
    reader = csv.DictReader(csvfile)
    reader = list(reader)
    reader_strip = []
    for row in reader:
        row = {r: row[r].strip() for r in row}
        reader_strip.append(row)
    reader = reader_strip.copy()

missingAudioFiles = []
for row in reader:
    if audio_dataset.get(row['File Name'], None) is None:
        missingAudioFiles.append(row['File Name'])

missingAudioFiles = set(missingAudioFiles)
print(
    f'{len(missingAudioFiles)} audio files are missing corresponding to excell.'
)

megan_data_sheet = []
for row in reader:
    if row['File Name'] not in ignore_files:
        if row['File Name'] not in missingAudioFiles:
            megan_data_sheet.append(row)
deleted_count = len(reader) - len(megan_data_sheet)
pprint(f'{deleted_count} number of samples are DELETED')

# %%
# reader[0]

# %%
# dataset properties
audio_dataset.excerpt_length = 10
audio_dataset.excell_names2code = {
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

# %%
# Store taxonomy information in the dataset.
# Taxonomy file
taxonomy_file_path = Path(
    '/home/enis/projects/nna/src/nna/assets/taxonomy/taxonomy.yaml')

with open(taxonomy_file_path) as f:
    taxonomy = yaml.load(f, Loader=yaml.FullLoader)

t = dataimport.Taxonomy(deepcopy(taxonomy))
audio_dataset.taxonomy = t

# %%
# Go through rows of the excell and
category_code_counter = Counter()
for row in megan_data_sheet:
    taxonomy_code = dataimport.megan_excell_row2yaml_code(row)
    audio_sample = audio_dataset[row['File Name']]
    audio_sample.taxo_code = taxonomy_code
    if audio_sample.length >= sample_length_limit:
        sample_count = audio_sample.length // audio_dataset.excerpt_length
        category_code_counter.update({audio_sample.taxo_code: sample_count})

# %%
# find taxonomies with not enough data
taxonomy_no_enough_data = []
print('classes that do not have enough data\nwill be REMOVED!')
for k, v in category_code_counter.items():
    if v < sample_count_limit:
        print(audio_dataset.taxonomy.edges[k], v)
        taxonomy_no_enough_data.append(k)

print('classes that have enough data\n')
for k, v in category_code_counter.items():
    if v >= sample_count_limit:
        print(audio_dataset.taxonomy.edges[k], v)

#  DELETE taxonomy with not enough data
samples_2_delete = []
for k, v in audio_dataset.items():
    if v.taxo_code in taxonomy_no_enough_data:
        samples_2_delete.append(k)

for k in samples_2_delete:
    del audio_dataset[k]

pprint(f'{len(samples_2_delete)} number of samples are deleted because ' +
       'their taxonomy category does not have enough data.')
