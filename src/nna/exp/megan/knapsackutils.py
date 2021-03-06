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
    taxo2loc_dict = {}
    for row in megan_data_sheet:
        try:
            taxonomy_code = dataimport.megan_excell_row2yaml_code(row, None)
            site_id = row['Site ID'].strip()
            taxo2loc_dict.setdefault(taxonomy_code, Counter({}))

            taxo2loc_dict[taxonomy_code] = taxo2loc_dict[
                taxonomy_code] + Counter({site_id: 1})
        except:
            print(row)
    return taxo2loc_dict


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
        bin_capacities = tuple(np.ceil(total *
                                       dist).astype('int'))  # type: ignore
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

    from https://github.com/BirdVox/cramer2020icassp/blob/master/00_create_splits.py
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

        bin_capacities_w_solutions = solve_knapsack4combinations(
            weights, values)

        solution_per_taxonomy[k] = (bin_capacities_w_solutions)

    return solution_per_taxonomy


def solve_knapsack4combinations(weights, values):

    total = sum(weights)
    combinations = approx_split_combinations(total)
    bin_capacities_w_solutions = []
    for bin_capacities in combinations:
        bin_capacities, solution = solve_knapsack(weights, values,
                                                  bin_capacities)
        if solution is not None:
            bin_capacities_w_solutions.append((bin_capacities, solution))

    return bin_capacities_w_solutions


def solve_knapsack(weights, values, bin_capacities):

    data = create_data_model(weights, values, bin_capacities)

    # SETUP knapsack
    # old version of ortools
    # Create the mip solver with the CBC backend.
    #             solver = pywraplp.Solver.CreateSolver('multiple_knapsack_mip', 'CBC')
    # new version of ortools=>8.1
    solver = pywraplp.Solver.CreateSolver('SCIP')  # type: ignore

    # Variables
    # x[i, j] = 1 if item i is packed in bin j.
    knapsack_coefficients = {}
    for i in data['items']:
        for j in data['bins']:
            knapsack_coefficients[(i,
                                   j)] = solver.IntVar(0, 1, 'x_%i_%i' % (i, j))

    # Constraints
    # Each item can be in at most one bin.
    for i in data['items']:
        solver.Add(sum(knapsack_coefficients[i, j] for j in data['bins']) <= 1)
    # The amount packed in each bin cannot exceed its capacity.
    for j in data['bins']:
        solver.Add(
            sum(knapsack_coefficients[(i, j)] * data['weights'][i]
                for i in data['items']) <= data['bin_capacities'][j])

    # Objective
    objective = solver.Objective()

    for i in data['items']:
        for j in data['bins']:
            objective.SetCoefficient(knapsack_coefficients[(i, j)],
                                     data['values'][i])
    objective.SetMaximization()
    status = solver.Solve()

    return read_knapsack_solution(status, data, bin_capacities,
                                  knapsack_coefficients)


def read_knapsack_solution(status, data, bin_capacities, knapsack_coefficients):
    if status == pywraplp.Solver.OPTIMAL:
        total_weight = 0
        solution = [list() for i in range(len(data['bins']))]
        for binIndex, j in enumerate(data['bins']):
            del binIndex
            bin_weight = 0
            bin_value = 0
            for i in data['items']:
                solution_value = knapsack_coefficients[i, j].solution_value()
                if solution_value > 0:
                    solution[j].append(data['weights'][i])
                    bin_weight += data['weights'][i]
                    bin_value += data['values'][i]

            total_weight += bin_weight
        return (bin_capacities, solution[:])
    else:
        print('The problem does not have an optimal solution.')
        return None, None


# %%
def find_best_solution_per_taxo(item_weights_by_taxo,
                                solution_per_taxonomy,
                                expected_dist=None):
    '''
    
    Args:
        item_weights_by_taxo: dict, keys are taxonomies, values are
                             dict such as item:weight.
        solution_per_taxonomy: a dict, keys are taxonomies, values are
            list of solutions. Each solution is a list with two items which are:
            list of bin size dist,list of item weights in each bin. 
                {'1.0.0':[((35, 7, 7), [[1, 2, 1, 1, 1, 1, 5, 1, 11, 1, 5, 3, 1], [7], [6]]),
                        ((36, 8.0, 8.0), [[1, 2, 1, 1, 1, 1, 5, 1, 11, 1, 5, 3, 1], [7], [6]]),
                        ((30, 11.0, 11.0), [[1, 2, 1, 1, 1, 1, 5, 1, 6, 11], [1, 5, 3, 1], [7]]),
                        ...
                }

    Return:
        {'1.0.0': [0.0026258626328006605,
            [29, 9, 9],
            [[6, 11, 5, 7], [5, 3, 1], [1, 2, 1, 1, 1, 1, 1, 1]]],
            '3.0.0': [0.01975404831728412,
            [27, 10, 9],
            [[4, 4, 3, 5, 2, 5, 2, 2], [5, 2, 3], [5, 4]]], ...}

    '''
    if expected_dist is None:
        expected_dist = [0.6, 0.2, 0.2]

    results = []
    best_solution_per_taxonomy = {}

    for taxo_key in solution_per_taxonomy:
        item_weights = item_weights_by_taxo[taxo_key]
        solutions = [m[1] for m in solution_per_taxonomy[taxo_key]]

        smallest_cost, best_dist, best_comb = pick_solution_by_bin_sum_distr(
            item_weights, solutions, expected_dist)

        results.append([smallest_cost, best_dist, best_comb])

        best_solution_per_taxonomy[taxo_key] = [
            smallest_cost, best_dist, best_comb
        ]
    return results, best_solution_per_taxonomy


def pick_solution_by_bin_sum_distr(item_weights, solutions, expected_dist):
    '''Given a distribution of total weight in each bin, find closest solution.
        
        We want to divide data into bins/knapsacks but we are not strict on
        the size of the bins, but we want to pick the closest one. 
        Here we use JSD metric to find closest distribution.

        Args:
            item_weights: dict from items/location to corresponding weights 
            solutions: weights of possible solutions in each bin
                        [[[1, 2, 1, 1, 1, 1, 5, 1, 11, 1, 5, 3, 1], [7], [6]],
                        [[1, 2, 1, 1, 1, 1, 5, 1, 6, 11], [1, 5, 3, 1], [7]]...]

    '''
    found = False
    total = sum(item_weights.values())

    for a_solution in solutions:
        if total == sum([sum(m) for m in a_solution]):
            found = True
            break
    if found is False:
        print(item_weights.values())
        print(total)

    smallest_cost = float('Inf')
    best_comb = None
    best_dist = None
    for a_solution in solutions:
        #         print(dist=[sum(m) for m in i[1]])
        dist = [sum(m) for m in a_solution]
        cost = JSD(expected_dist, dist)
        if cost < smallest_cost and total - sum(dist) == 0:
            smallest_cost = cost
            best_comb = a_solution
            best_dist = [sum(m) for m in a_solution]

    # sort best_dist and best_comb
    combined_sorted = sorted(
        list(zip(best_dist, best_comb)),  # type: ignore
        reverse=True)
    best_dist, best_comb = list(zip(*combined_sorted))

    return smallest_cost, best_dist, best_comb


def pick_solution_by_item_distr(item_weights, solutions, location2taxo_dict,
                                target_dist):
    '''Pick best solution has closest item dist between bins.
    '''
    best_sol = None
    lowest_cost = float('Inf')
    target_array = np.array(list(target_dist.values()))
    target_array = target_array/np.sum(target_array)

    for sol in solutions:
        #
        bins_as_taxo_dist = map_weight_sol2taxo_dist_sol(
            item_weights, sol, location2taxo_dict)
        # turn Counters into array with same order of target_dist
        bins_as_taxo_array = []
        for a_bin in bins_as_taxo_dist:
            bin_array = [a_bin.get(k, 0) for k in target_dist.keys()]
            bins_as_taxo_array.append(bin_array)
        
        cost, _ = calculate_item_distr_distance(bins_as_taxo_array,
                                                target_array)
        if lowest_cost > cost:
            best_sol = sol
            lowest_cost = cost

    return best_sol, lowest_cost


def map_weight_sol2taxo_dist_sol(item_weights, solution, location2taxo_dict):

    solution_by_location_name = map_weights2indexes(item_weights, solution)
    # go from items/locations to taxonomy Counter
    # bins_w_taxo_dist = [list() for m in range(len(solution_by_location_name))]
    bins_as_taxo_dist = []
    for a_bin in (solution_by_location_name):
        tmp_c = Counter()
        for loc_name in a_bin:
            tmp_c += location2taxo_dict[loc_name]
        bins_as_taxo_dist.append(tmp_c)
    return bins_as_taxo_dist


def reverse_taxo2loc_dict(taxo2loc_dict):
    '''reverse dict with keys of taxonomy and values with counter of locations.

    Args:
        taxo2loc_dict: {'1.0.0': Counter({'45': 1,
                                        '50': 2,
                                        '48': 1,}),
                        '3.0.0': Counter({'45': 4,
                                        '20': 4,})..}
    returns:
        {'45': Counter({'1.0.0': 1, '3.0.0': 4, 'X.X.X': 4}),
        '50': Counter({'1.0.0': 2, '1.1.10': 5, '1.1.0': 48, '1.3.0': 17}),}

    '''
    location2taxo_dict = {}
    for code, source_counter in taxo2loc_dict.items():
        for source, count in source_counter.items():
            location2taxo_dict.setdefault(source, Counter())
            location2taxo_dict[source] += Counter({code: count})
    return location2taxo_dict


def calculate_item_distr_distance(bins, target_distr):
    '''calculate largest distance between bins and target_distr

        Args:
            bins: list of lists, [[item1_count,...],[item1_count,...],[item1_count,...]..]
                 inner list contains count of each item in the bin. Bins has to 
                 have same length with target_distr and each index should 
                 correspond to same item.

    '''

    for a_bin in bins:
        assert len(a_bin) == len(target_distr)

    all_costs = []
    for a_bin in bins:
        cost = JSD(a_bin, target_distr)
        all_costs.append(cost)

    return max(all_costs), all_costs


# %%


def knapsack_index2location_name(codesDict, best_solution_per_taxonomy):
    '''Replace indexes of locations with location names. 

        Important thing to keep in mind is that knapsack solver does not return
        indexes of possible items, it returns weights and there can be multiple
        items with same weight. So we just assign first one with that weight
        to one of the bins.

        Args:
            codesDict:{'1.0.0': Counter({'45': 1,
                                        '50': 2,
                                        '48': 1,
                                        '39': 1,
                                        '30': 1,}
                        '3.0.0': Counter({'45': 4,}
                                                ...}
            best_solution_per_taxonomy:{taxo:[cost,bins_dist,[bin1,bin2,[item_weights]]}
                    {'1.0.0': [0.00015203111926031642,
                    [29, 9, 9],
                    [[6, 11, 5, 7], [5, 3, 1], [1, 2, 1, 1, 1, 1, 1, 1]]],}
        
        returns: {taxo:[[train],[test],[valid]]}
            {'1.0.0': [['44', '46', '17', '14'],
                ['11', '34', '27'],
                ['31', '50', '18', '12', '30', '39', '48', '45']],
                '3.0.0': [['40', '20', '14', '17', '13', '36', '25', '33'],
                ['18', '38', '39'],
                ['32', '45']],
                ...}
    '''
    best_solution_per_taxonomy_by_location = {}
    for taxo in best_solution_per_taxonomy.keys():
        item_weights = codesDict[taxo]
        knapsack_sol_weights = best_solution_per_taxonomy[taxo][2]
        knapsack_sol_indexes = map_weights2indexes(item_weights,
                                                   knapsack_sol_weights)
        best_solution_per_taxonomy_by_location[taxo] = knapsack_sol_indexes
    return best_solution_per_taxonomy_by_location


def map_weights2indexes(item_weights, knapsack_sol_weights):
    '''Weights to index of items. 

        args:
            item_weights: item IDs to weights
                            {'45': 1,
                                '50': 2,
                                '48': 1},
            knapsack_solution: [[item_weights],[item_weights] ... for each bin]
        returns:[[item indexes],[item indexes] ... for each bin] 
    '''
    counter = dict(item_weights)
    weight2indexes = {}
    for x, y in counter.items():
        weight2indexes.setdefault(y, []).append(x)

    knapsack_sol_indexes = [[] for i in range(len(knapsack_sol_weights))]
    for i, a_set_of_weights in enumerate(knapsack_sol_weights):
        for a_weight in a_set_of_weights:
            item_index = weight2indexes[a_weight].pop()
            knapsack_sol_indexes[i].append(item_index)

    return knapsack_sol_indexes


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
def main(version="split_by_distr"):
    if version == "split_per_taxo":
        taxo2loc_dict = add_taxo_code2dataset(reader)
        total2 = 110
        dist2 = np.array([0.6, 0.2, 0.2])
        #test
        # np.ceil(total*dist).astype('int')
        solution_per_taxonomy = multiple_knapsack_solve(taxo2loc_dict)
        results2, best_solution_per_taxo = find_best_solution_per_taxo(
            taxo2loc_dict, solution_per_taxonomy)

        # %%

        results2 = sorted(results2, reverse=True)

        best_solution_per_taxo_w_loc_name = knapsack_index2location_name(
            taxo2loc_dict, best_solution_per_taxo)
        # BestSolutionPerTaxonomyLocation
        results2file_names_like_cramer(best_solution_per_taxo_w_loc_name)

    # %%
    ###############
    # TODO pick_solution_by_item_distr is calling map_weight_sol2taxo_dist_sol 
            # first improve that
    # TODO improve map_weight_sol2taxo_dist_sol so it returns all possible combinations

    elif version =="split_by_distr":
        #load excell
        taxo2loc_dict = add_taxo_code2dataset(reader)

        #merge taxo counters
        a=list(taxo2loc_dict.values())
        all_taxo=Counter()
        for m in a:
            all_taxo+=m

        # weights and values for knapsack
        # counts of taxonomies are weights
        # bins are train,validation,test
        weights = list(all_taxo.values())
        values = weights

        #solve knapsack possible distributions around ~(60,20,20)
        bin_capacities_w_solutions = solve_knapsack4combinations(weights,values)

        # remove bin_capacity info from solutions
        solutions = [solution for bin_capacity,solution in bin_capacities_w_solutions]

        # switch dict keys with key of value dicts
        item_weights=all_taxo
        location2taxo_dict = reverse_taxo2loc_dict(taxo2loc_dict)

        dataset_taxo_distr = {k:sum(v.values()) for k,v in  taxo2loc_dict.items()}

        target_dist = dataset_taxo_distr

        best_sol, lowest_cost = pick_solution_by_item_distr(item_weights, solutions, location2taxo_dict, target_dist)

        solution_by_location_name = map_weights2indexes(item_weights, best_sol)

        # print solutions
        solution_by_taxo_dist=[]
        for sol_bin in solution_by_location_name:
            taxo_counter = [location2taxo_dict[loc] for loc in sol_bin]
            tc=Counter()
            for c in taxo_counter:
                tc+=c
            solution_by_taxo_dist.append(tc)
            

        t_dist = {}
        for taxo in taxo2loc_dict.keys():
            train=solution_by_taxo_dist[0][taxo]
            val=solution_by_taxo_dist[1][taxo]
            test = solution_by_taxo_dist[2][taxo]
            total = (train+val+test)
            t_dist[taxo]=(train/total,val/total,test/total)
            print(f"{taxo} : {train/total:1.2}, {val/total:1.2}, {test/total:1.2}, {total}")
        return solution_by_location_name
    ###############