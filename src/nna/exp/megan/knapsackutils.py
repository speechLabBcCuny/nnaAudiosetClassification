# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from __future__ import print_function
from ortools.linear_solver import pywraplp

from scipy.stats import entropy as kl_div
from numpy.linalg import norm
import numpy as np

# %%
import csv
from collections import Counter

# sheetOrg='/Users/berk/Desktop/NNA/downloads/Sheet1.csv'
# sheetMine='/Users/berk/Desktop/NNA/downloads/Sheet1(1).csv'

resources_folder = ('/scratch/enis/archive/' +
                    'forks/cramer2020icassp/resources/')
src_path = '/scratch/enis/data/nna/labeling/megan/AudioSamplesPerSite/'
megan_labeled_files_info_path = src_path + 'meganLabeledFiles_wlenV1.txt'

# csv4megan_excell = (resources_folder + 'Sheet1.csv')
csv4megan_excell_clenaed = (resources_folder + 'Sheet1(1).csv')
csv4megan_excell = (resources_folder + 'Sheet1.csv')

with open(csv4megan_excell_clenaed) as csvfile:
    reader = csv.DictReader(csvfile)
    reader = list(reader)

# %%
# reader[0]

# from typing import Dict, Union, Optional, Type
from nna import dataimport


# %%
def add_taxo_code2dataset(megan_data_sheet):
    '''Create Counter for each taxonomy from excell sheet

        Returns:
            {'1.1.1':Counter('22':5,'11':10...),
            '0.1.1':Counter('5':5,'6':10...)}

        todo remove try except
    '''
    codest_dict = {}
    for row in megan_data_sheet:
        try:
            taxonomy_code = dataimport.megan_excell_row2yaml_code(row, None)
            site_id = row['Site ID'].strip()
            codest_dict.setdefault(taxonomy_code, Counter({}))

            codest_dict[taxonomy_code] = codest_dict[taxonomy_code] + Counter(
                {site_id: 1})
        except:
            print(row)
    return codest_dict


# %%
# [0.6,0.2,0.2]
# [0.7,0.15,0.15]
# [0.8,0.1,0.1]
def approx_split_combinations(total):
    '''calculate posssible combinations that sums to  ~total (aproximate). 

        Distribution assumptions:
            test~=valid, 
            0.8~>train~>0.6
            0.4~>test,valid~>0.2

    example:
        total=110
        print(getCombinations(total))
            {(70, 42.0, 42.0), (71, 20, 20), (88, 15.0, 15.0), ...}
    '''
    combinations = set()
    for i in range(100, 201, 1):
        test_val_dist = i / 1000
        train_dist = 1 - (test_val_dist * 2)
        dist = np.array([train_dist, test_val_dist, test_val_dist])
        bin_capacities = tuple(np.ceil(total * dist).astype('int')) # type: ignore
        combinations.add(bin_capacities)

    #add some combinations that are test and valid
    # are bigger so that small number of elements can be handled
    combinations2 = combinations.copy()
    for comb in combinations2:
        for rate in [1.2, 1.4, 1.6, 1.8, 2.0]:
            newComb = (comb[0], np.ceil(comb[1] * rate),
                       np.ceil(comb[1] * rate))
            combinations.add(newComb)

    return combinations


# test
# print(getCombinations(total))

# %%


def JSD(P, Q):
    '''

        todo: add reference and docs
    '''
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (kl_div(_P, _M) + kl_div(_Q, _M))


def create_data_model(weights, values, bin_capacities):
    '''Create variables for knapsack problem setup.

    '''
    data = {}
    data['weights'] = weights
    data['values'] = values
    data['items'] = list(range(len(weights)))
    data['num_items'] = len(weights)
    data['bins'] = list(range(len(bin_capacities)))
    data['bin_capacities'] = bin_capacities

    return data


# %%


def multiple_knapsack_solve(codes_dict):
    '''Solves a knapsack problems for each key of codes_dict.

        codes_dict's keys are names and values are Counters of samples on
            each location. Items of a counter are items in the knapsack problem.
            We try to place each location into one of the train,test,valid bins.

            Expected distribution is calculated by approx_split_combinations.

        Returns: a dict solution_per_taxonomy
            keys are same with codes_dict and values are possible solutions to
            the knapsack problem. 

    '''
    total = 0
    solution_per_taxonomy = {}
    for k in codes_dict.keys():

        weights = list(codes_dict[k].values())
        values = list(codes_dict[k].values())
        if sum(weights) < 10:
            print(k)
            print('Error, too small', weights)
            continue
        if len(weights) < 3:
            print(k)
            print('Error, number of elements less than 3', weights)
            continue

        total = sum(weights)

        combinations = approx_split_combinations(total)

        #         solutionPerCombination=[]
        solution_per_taxonomy.setdefault(k, [])
        for bin_capacities in combinations:

            data = create_data_model(weights, values, bin_capacities)
            # old version of ortools
            # Create the mip solver with the CBC backend.
            #             solver = pywraplp.Solver.CreateSolver('multiple_knapsack_mip', 'CBC')
            # new version of ortools=>8.1
            solver = pywraplp.Solver.CreateSolver('SCIP') # type: ignore

            # Variables
            # x[i, j] = 1 if item i is packed in bin j.
            x = {}
            for i in data['items']:
                for j in data['bins']:
                    x[(i, j)] = solver.IntVar(0, 1, 'x_%i_%i' % (i, j))

            # Constraints
            # Each item can be in at most one bin.
            for i in data['items']:
                solver.Add(sum(x[i, j] for j in data['bins']) <= 1)
            # The amount packed in each bin cannot exceed its capacity.
            for j in data['bins']:
                solver.Add(
                    sum(x[(i, j)] * data['weights'][i]
                        for i in data['items']) <= data['bin_capacities'][j])

            # Objective
            objective = solver.Objective()

            for i in data['items']:
                for j in data['bins']:
                    objective.SetCoefficient(x[(i, j)], data['values'][i])
            objective.SetMaximization()

            status = solver.Solve()

            if status == pywraplp.Solver.OPTIMAL:
                #                 if objective.Value()/sum(data['bin_capacities'])>0.90:
                #                     continue

                total += sum(data['weights'])

                #                 print(codesDict[k])
                #                 print('------------',k,'--------------')
                #                 print('Total packed value:', objective.Value(),'/',sum(data['bin_capacities']))

                #             print()
                total_weight = 0
                solution = [list() for i in range(len(data['bins']))]
                for binIndex, j in enumerate(data['bins']):
                    bin_weight = 0
                    bin_value = 0
                    #                     print('Bin ', j, '\n')
                    for i in data['items']:
                        if x[i, j].solution_value() > 0:
                            solution[j].append(data['weights'][i])
                            #                             print('Item', i, '- weight:', data['weights'][i], ' value:',
                            #                                   data['values'][i])
                            bin_weight += data['weights'][i]
                            bin_value += data['values'][i]

#                     print('Packed bin weight:', bin_weight,'/',data['bin_capacities'][binIndex])
#                 print('bin capacity:',)
#                 print('Packed bin value:', bin_value)
                    total_weight += bin_weight
#                 print('Total packed weight:', total_weight)
                solution_per_taxonomy[k].append((bin_capacities, solution[:]))
            else:
                print('The problem does not have an optimal solution.')
#             print('total',total)
    return solution_per_taxonomy


# %%


# %%
def find_best_solution(codes_dict):
    '''

    Return:
        {'1.0.0': [0.0026258626328006605,
            [29, 9, 9],
            [[6, 11, 5, 7], [5, 3, 1], [1, 2, 1, 1, 1, 1, 1, 1]]],
            '3.0.0': [0.01975404831728412,
            [27, 10, 9],
            [[4, 4, 3, 5, 2, 5, 2, 2], [5, 2, 3], [5, 4]]], ...}

    '''
    expectedDist = [0.6, 0.2, 0.2]
    results = []
    best_solution_per_taxonomy = {}

    for taxoKey in solutionPerTaxonomy:
        found = False
        total = sum(codes_dict[taxoKey].values())

        for a_solution in solutionPerTaxonomy[taxoKey]:
            if total == sum([sum(m) for m in a_solution[1]]):
                found = True
        if found is False:
            print(codes_dict[taxoKey].values())
            print(total)

    #         for i in solutionPerTaxonomy[taxoKey]:
    #             print(i[0],,sum([sum(m) for m in i[1]]))

        smallest_cost = 999999
        best_comb = None
        best_dist = None
        for a_solution in solutionPerTaxonomy[taxoKey]:
            #         print(dist=[sum(m) for m in i[1]])
            dist = [sum(m) for m in a_solution[1]]
            cost = JSD(expectedDist, dist)
            if cost < smallest_cost and total - sum(dist) == 0:
                smallest_cost = cost
                best_comb = a_solution[1]
                best_dist = [sum(m) for m in a_solution[1]]

        # sort best_dist and best_comb
        combined_sorted = sorted(list(zip(best_dist, best_comb)), reverse=True) # type: ignore
        a, b = [], []
        for m in combined_sorted:
            a.append(m[0])
            b.append(m[1])
        best_dist, best_comb = a, b

        results.append([smallest_cost, best_dist, best_comb])

        best_solution_per_taxonomy[taxoKey] = [
            smallest_cost, best_dist, best_comb
        ]
    return results, best_solution_per_taxonomy


# %%
def knapsack_index2location_name(codesDict, best_solution_per_taxonomy):
    '''Replace indexes of locations with location names. 
        
        returns:
            {'1.0.0': [['44', '46', '17', '14'],
                ['11', '34', '27'],
                ['31', '50', '18', '12', '30', '39', '48', '45']],
                '3.0.0': [['40', '20', '14', '17', '13', '36', '25', '33'],
                ['18', '38', '39'],
                ['32', '45']],
                ...}
    '''
    solReverse = {i: {} for i in codesDict.keys()}
    for taxo, counter in codesDict.items():
        counter = dict(counter)
        for x, y in counter.items():
            solReverse[taxo].setdefault(y, []).append(x)

    best_solution_per_taxonomy_by_location = {
        i: None for i in best_solution_per_taxonomy.keys()
    }
    for taxo, data in best_solution_per_taxonomy.items():
        #     print(taxo,data)
        comb = data[2]
        train, test, val = comb[:]
        combLocation = [[] for i in range(len(comb))]
        for i, dataSet in enumerate(comb):
            for v in dataSet:
                location = solReverse[taxo][v].pop()
                combLocation[i].append(location)
        best_solution_per_taxonomy_by_location[taxo] = combLocation # type: ignore
    return best_solution_per_taxonomy_by_location

# %%
# excellNames2code

# %%


def results2file_names_like_cramer(
    BestSolutionPerTaxonomyLocation,
    excellNames2code=None,
):
    # train test valid
    # BestSolutionPerTaxonomyLocation
    if excellNames2code is None:
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
    for yamlCode, data in BestSolutionPerTaxonomyLocation.items():
        #     print(yamlCode)
        fileCode = yamlCode.replace('.', '-')
        for dataSet in data:
            for loc in dataSet:
                fileName = ('_'.join(
                    ['site-' + str(loc), fileCode, 'original.h5']))
                pathFile = './resources/myDatasets/megan/' + fileName


#                 print(pathFile)

# %%
# birdvox-cls-test
# birdvox-cls-train
# birdvox-cls-valid
# load files with librosa, sample to

# %%

codest_dict2 = add_taxo_code2dataset(reader)
total2 = 110
dist2 = np.array([0.6, 0.2, 0.2])
#test
# np.ceil(total*dist).astype('int')
solutionPerTaxonomy = multiple_knapsack_solve(codest_dict2)
results2, BestSolutionPerTaxonomy2 = find_best_solution(codest_dict2)

# %%

results2 = sorted(results2, reverse=True)
[i[1] for i in results2]
# len(results),len(codesDict.keys())

BestSolutionPerTaxonomyLocation2 = knapsack_index2location_name(codest_dict2, BestSolutionPerTaxonomy2)
# BestSolutionPerTaxonomyLocation
results2file_names_like_cramer(BestSolutionPerTaxonomyLocation2)

# %%

# %%
