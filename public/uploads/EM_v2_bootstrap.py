# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 15:17:47 2019

@author: Philip F

EM Algorithm v2
This is the Revise version of EMCore written by Prof. Chang.
This algorithm is divided into the following steps:

Preprocessing
-------------
    1. Reading CSV File (csv)
    2. Creating OneHot Lot Encoding of Machines [1]_
    3. Preprocessing: Calculating LOT Information
    4. Preprocessing: Calculating Machine Information

the initial yieldrate is calculated using Least Square and L-BFGS-B [2]_, and is
included id step 4 above.



EM Algorithm with Bootstrap
------------
    Introducing miss rate on E and M steps
    5.1 Running EM Algorithm with random sampling data n times
    5.2 Running EM Algorithm 100% data

Report
------
    6. Saving reports to CSV

Optimization
------------
some of the code is optimize based on Python Speed Performance Tips recommended
by python.org [3]. In this code, it could be seen commented as
``Optimizing:...``. One of them are avoiding dots (e.g. avoid multiple calls
of ``df.my_column``)


References
----------
.. [1] Jezrael. 2019. Answer on "Create dummy variable of multiple columns with
   python". One hot encoding with pandas "get_dummies" and different column and
   same values is seen as one values. `link <https://stackoverflow.com/
   questions/55182909/create-dummy-variable-of-multiple-columns-with-python>`_

.. [2] The SciPy community. 2019. Limited-memory BFGS B. see: `scipy.optimize.
   fmin_l_bfgs_b <https://docs.scipy.org/doc/scipy/reference/generated/
   scipy.optimize.fmin_l_bfgs_b.html>`_

.. [3] Python.org. 2016. Python Speed Performance Tips. `link <https://wiki.
   python.org/moin/PythonSpeed/PerformanceTips>`_

"""

import time
import numpy as np
import pandas as pd
import scipy.optimize
from scipy.stats import bootstrap

import multiprocessing as mp
from joblib import Parallel, delayed

import custom_utility as cu
from collections import Counter
#np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)     
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)            

#############################|##############################
#                  CODE CONFIGURATIONS                     #
############################################################
# using 'raw_real_data.csv' or 'test_case_1a.csv'
RAWDATA_CSV_PATH_FILENAME = 'uploads/test_case_1.csv'
REPORT_PATH_FILENAME = "uploads/test_1_report.csv"
USE_MULTIPROCESS = False
MAX_RETRY = 5 # bootstrap max retry (if failed)
MAX_EMA_ITERATIONS = 60 # see the 1st paper result, 60-100 is enough
BOOTSTRAP_FREQ = 30 # the number of times doing sampling and estimating yield rate
BOOTSTRAP_SRS_FRAC = 0.8 # the fraction of random sampling, e.g. 80% of data
EPSILON = 1.e-99  # a small positive infinitesimal quantity
OLD_COLUMNS = {"lot name": 'Lot',
               "process prefix": 'Process',
               "process suffix": '_MachineNo',
               "lot step": 'Final',
               "processed": 'CurrentTotalNum',
               "bad pieces": 'BadNum',
               "total lot pieces": 'TotalNum'
               }
NEW_COLUMNS = {"lot name": 'lot_name',
               "process prefix": 'lot_machine_step',
               "lot step": 'lot_step',
               "processed": 'processed',
               "bad pieces": 'bad_pieces',
               }


#############################|##############################
#                      PREPROCESSING                       #
############################################################
def read_csv_files(filename):
    """
    read the preprocessed data with given path and filename:

    Parameter:
    ----------
    filename : str
        given path and filename such as ``"folder/yourfilename.csv"``

    Return:
    ----------
    raw_csv : DataFrame or TextParser (2D array)
        full data from your csv.

    """
    # read the chinese character with encoding big5
    # raw_csv = pd.read_csv(filename, encoding='big5', low_memory=False)
    raw_csv = pd.read_csv(filename, encoding='big5',
                          iterator=True, chunksize=10000)
    raw_csv = pd.concat(raw_csv, ignore_index=True)

    # renaming columns if lot name columns appear as old name
    if OLD_COLUMNS["lot name"] in raw_csv.columns:
        # make sure the conversion of process column
        _col_1 = len(set(raw_csv.CurrentTotalNum))
        _col_2 = len(set(raw_csv.TotalNum))
        process_col = 'CurrentTotalNum'
        if _col_1 < _col_2:
            process_col = 'TotalNum'

        # rename default name
        raw_csv.rename(columns={
            OLD_COLUMNS["lot name"]: NEW_COLUMNS["lot name"],
            OLD_COLUMNS["lot step"]: NEW_COLUMNS["lot step"],
            process_col: NEW_COLUMNS["processed"],
            OLD_COLUMNS["bad pieces"]: NEW_COLUMNS["bad pieces"]
        }, inplace=True)

        # remove old process column suffix
        raw_csv.columns = [col.replace(OLD_COLUMNS["process suffix"], '')
                           for col in raw_csv.columns]
        # rename all prefix of machine process column
        raw_csv.columns = [col.replace(OLD_COLUMNS["process prefix"],
                                       NEW_COLUMNS["process prefix"])
                           for col in raw_csv.columns]

    # fix unconsistent bad pieces:
    # when processed pieces is reduced in next row [shift(-1)],
    # then re-calculate all bad pieces in current row
    # except for the last step (-1)
    raw_csv['all_bad_pieces'] = raw_csv.bad_pieces
    raw_csv.loc[raw_csv['lot_step'] != -1, 'all_bad_pieces'] \
        = raw_csv.processed - raw_csv.processed.shift(-1)
    return raw_csv


def get_onehot_machine(source_data):
    """
    preprocess the machine name in each step with one hot encoding, then return
    hotencoding including machine name and updated source data.

    Parameter:
    ----------
    source_data : DataFrame (2D array)
        insert full source data from your csv.

    Return:
    ----------
    onehot_last_step : DataFrame (2D array)
        an each machine step feature in one hot encoding format, with
        machine name as its columns name
    machine_name : DataFrame (2D array)
        a machine information dataframe which only consist of machine name.
    source_data : DataFrame (2D array)
        full data source from your csv, but added a new column called
        ``lot_step_machines`` which consist of machines name that used in
        each LOT's step

    """
    # select the machine step columns (backslash to break, avoid long lines)
    # select column "ProcessXXX_MachineNo"
    machine_step_data = source_data.loc[:, source_data.columns.str. \
                                               startswith("lot_machine_step")]

    # one hot encoding with pandas "get_dummies" and
    # different column and same values is seen as one values
    onehot_machine_step = pd.get_dummies(machine_step_data,
                                         prefix='',
                                         prefix_sep='',
                                         dtype=int
                                         ).groupby(level=0, axis=1).max()
                                         #).max(level=0, axis=1) # old version of pandas (1.4.4)

    # get last step index of each LOT (from raw data)
    last_step_index = source_data.lot_step[source_data.lot_step == -1] \
        .index \
        .tolist()

    # filter the index based on last step raw data for one hot encoding
    onehot_last_step = onehot_machine_step.iloc[last_step_index]
    # machine name as dataframe
    machine_name = pd.DataFrame({'machine_name': list(onehot_last_step)})
    # machine_name['index'] = machine_name.index

    # get the machine used for every row, based on machine config
    lot_step_machines = machine_step_data.copy()

    # change dataframe to tuple in 4 steps
    # Step 1: get non NaN/Null value each row using "stack" data structure
    lot_step_machines = lot_step_machines.stack()
    # Step 2: group by dataframe row / index value (level=0)
    lot_step_machines = lot_step_machines.groupby(level=0)
    # Step 3: change to list for each row using apply depends on group by
    lot_step_machines = lot_step_machines.apply(list)
    # Step 4: change to tuple of list
    lot_step_machines = tuple(lot_step_machines)  # change to tuple of list

    # modify the raw data to add new column
    source_data['lot_step_machines'] = lot_step_machines

    return onehot_last_step, machine_name, source_data


def get_lot_information(source_data):
    """
    read the statistic of each lot in given source dataframe.

    Parameter:
    ----------
    source_data : DataFrame (2D array)
        insert full data from your csv.

    Return:
    ----------
    lot_info : DataFrame or TextParser (2D array)
        A statistical dataframe consist of 'lot_name', 'machine_used',
        'lot_total_pieces', 'lot_good_pieces', 'lot_yield_rate',
        and 'ln_lot_yield_rate' for each LOT.

    """

    # function to avoid ZeroDivisionError, and replace result to near zero
    def safe_division(x, y):
        if x == 0 or y == 0:
            return EPSILON
        return x / y

    # initializing dataframe
    _columns = ['lot_name', 'machine_used', 'detected_bad_pieces', 'lot_total_pieces',
                'lot_good_pieces', 'lot_bad_pieces', 'lot_yield_rate',
                'ln_lot_yield_rate']
    lot_info = pd.DataFrame(data=None, columns=_columns)

    # groping the raw data based on LOT Name
    lot_group = source_data.groupby('lot_name')

    # calculating all columns value in lot_info for each row / each LOT
    for _lot_name, _lot_data in lot_group:
        # get row where lot_step = -1
        last_step = _lot_data.loc[_lot_data.lot_step == -1]

        # fix unconsistent bad pieces:
        # find delta between all bad pieces and micro crack bad pieces
        lot_all_defect = _lot_data.all_bad_pieces.sum()
        lot_all_microcrack = _lot_data.bad_pieces.sum()
        lot_badpieces_delta = lot_all_defect - lot_all_microcrack

        # get data for lot_info
        # use .item() to get the value of groupby cells -- deprecated, use iat[0]
        machine_used = last_step.lot_step_machines.iat[0]  # .item()

        # process 'machine_bad_pieces' column
        detected_bad_pieces = list(_lot_data.bad_pieces)

        # syncronized with only micro crack defect
        lot_total_pieces = _lot_data.processed.max() - lot_badpieces_delta
        lot_good_pieces = last_step.processed.iat[0] - last_step.bad_pieces.iat[0]
        lot_bad_pieces = lot_total_pieces - lot_good_pieces

        # use function to safely process the division
        lot_yield_rate = safe_division(lot_good_pieces, lot_total_pieces)
        ln_lot_yield_rate = np.log(lot_yield_rate)

        # append to the list based on _columns name
        current_lot_info = [_lot_name, machine_used, detected_bad_pieces, lot_total_pieces,
                            lot_good_pieces, lot_bad_pieces, lot_yield_rate,
                            ln_lot_yield_rate]

        # append to statistic dataframe
        lot_info.loc[len(lot_info)] = current_lot_info

    return lot_info


def get_machine_information(onehot_lot_data, machine_info, lot_info):
    """
    calculate the statistic of each machine from several related sources.

    Parameter:
    ----------
    onehot_lot_data : DataFrame (2D array)
        Dataframe consist of each machine step feature in one hot encoding
        format, with machine name as its columns name
    machine_info : DataFrame (2D array)
        A machine information dataframe which only consist of machine name.
    lot_info : DataFrame (2D array)
        An information dataframe for each LOT.

    Return:
    ----------
    machine_info : DataFrame or TextParser (2D array)
        A machine information dataframe consist of 'machine_name',
        'good_produced', 'init_yieldrate', and 'em_yieldrate' for each machine.

    """

    # Step 1: Calculating the number of bad pieces detected at machine m for each machine
    # Step 2: Calculating Good Pieces Produced for each machine
    # Step 3: Calculating the number of initial not detected bad pieces for each machine
    # initializing dataframe
    machine_info['machine_bad_pieces'] = 0
    machine_info['good_produced'] = 0

    # for faster indexing, first, change dataframe to dictionary
    # dict(zip(keys, values)) --> to change 2 lists to single dictionary
    machine_bp = dict(zip(machine_info.machine_name,
                          machine_info.machine_bad_pieces))
    machine_good = dict(zip(machine_info.machine_name,
                            machine_info.good_produced))
    machine_init_not_detected = dict.fromkeys(machine_info.machine_name, 0)

    # calculate good produced for each machine based on LOT's machine usage
    for _, lot_step in lot_info.iterrows():

        # Optimizing: caching variable for faster get value

        machine_bp_lot = list(zip(lot_step.machine_used, lot_step.detected_bad_pieces))
        # Step 1 (Calculate the Bm)
        _temp_bp_counter = Counter()

        # Sum all the bp with same machine name in this LOT
        for _mch_name, _bp_found in machine_bp_lot:  
            _temp_bp_counter[_mch_name] += _bp_found
        for _mch_name in _temp_bp_counter:
            machine_bp[_mch_name] += _temp_bp_counter[_mch_name]

        # Step 2 (Calculate the Good Produced)
        machine_used = lot_step.machine_used
        good_pieces = lot_step.lot_good_pieces

        for machine in machine_used:
            # add or sum good pieces to previous value
            machine_good[machine] += good_pieces

        # Step 3 (Calculate the init_not_detected)
        remove_repeat_machines = list(
            dict.fromkeys(lot_step.machine_used))  # remove duplicates in list with maintaining order
        for mach in remove_repeat_machines:
            for index, value in enumerate(machine_bp_lot):
                if index > machine_used.index(mach): # and value[0] != mach:  # machine_used.index(mach) get the first match index of mach in machine_used
                    machine_init_not_detected[mach] += value[1]

    # insert to dataframe
    machine_info['machine_bad_pieces'] = machine_bp.values()
    machine_info['good_produced'] = machine_good.values()

    dict_init_miss_rate = dict.fromkeys(machine_info.machine_name, 0)
    for q in machine_init_not_detected:
        if (machine_init_not_detected[q] + machine_bp[q]) > 0:
            dict_init_miss_rate[q] = machine_init_not_detected[q] / (machine_init_not_detected[q] + machine_bp[q])
        else:
            dict_init_miss_rate[q] = 0  # Avoid divided by 0

    machine_info['init_miss_rate'] = dict_init_miss_rate.values()

    # Step 4: Calculating initial yield rate for each machine
    machine_info['init_yieldrate'] = init_yieldrate_est(onehot_lot_data, lot_info)

    return machine_info

def init_yieldrate_est(onehot_data, lot_info):
    """
    calculating each machine's initial yieldrate using Least Square and
    L-BFGS-B (Limited-memory BFGS).
    see: `scipy.optimize.fmin_l_bfgs_b <https://docs.scipy.org/doc/scipy/
    reference/generated/scipy.optimize.fmin_l_bfgs_b.html>`_

    Parameter:
    ----------
    onehot_lot_data : dataframe (2D array)
        insert the onehot encoding data of the LOT's Last Step
    lot_info : dataframe (2D array)
        insert a dataframe which consist of statistic details of each lot

    Return:
    ----------
    est_yieldrate : list or 1D array
        the yield rate probability using least square algorithm, and minimized
        by a function of L-BFGS-B algorithm.

    """

    # initializing yield rate estimation with all zeros
    max_machine_step = len(onehot_data.columns)
    log_est_yieldrate = np.zeros((max_machine_step,))

    # lot onehot where step == -1
    onehot_data = onehot_data.values  # convert to array --> as X on formula

    # get initial yield rate as array
    lot_info = lot_info["ln_lot_yield_rate"].values
    # reshape array with any rows but with 1 column
    lot_yieldrate = lot_info.reshape(-1, 1)  # --> as Y on formula

    # calculate dot product for config and yield rate
    config_dot_config = onehot_data.T.dot(onehot_data)  # --> X^2
    config_dot_yield = onehot_data.T.dot(lot_yieldrate)  # --> X.Y

    # least Square
    def perform_least_square(params, cfg, i_yr, c_dot_c, c_dot_y):
        """
        using Limited-memory BFGS
        see: `scipy.optimize.fmin_l_bfgs_b <https://docs.scipy.org/doc/scipy/
        reference/generated/scipy.optimize.fmin_l_bfgs_b.html>`_
        """
        yieldrate = (c_dot_c.dot(params.reshape(-1, 1)) - c_dot_y) / 2
        val = cfg.dot(params.reshape(-1, 1)) - i_yr
        val *= val
        return val.sum() / 2, yieldrate

    # Minimize a function using the L-BFGS-B algorithm.
    log_est_yieldrate = scipy.optimize.fmin_l_bfgs_b \
        (perform_least_square,
         log_est_yieldrate,
         None, (onehot_data,
                lot_yieldrate,
                config_dot_config,
                config_dot_yield
                ),
         False,
         [(None, 0) for _ in range(max_machine_step)])

    # restore from np.log in initial yield rate (remove min and info)
    est_yieldrate = np.exp(log_est_yieldrate[0])

    return est_yieldrate


#############################|##############################
#                      EM ALGORITHM                        #
############################################################

def e_step(source_data, machine_info):
    """
    The E-STEP of Expectation–Maximization (EM) algorithm.
    This step is used to calculate the hidden variable or Expectation of
    bad pieces in each machine, hence can provide the estimation of the number
    of bad pieces and the number of good pieces produced by each machine.

    Parameter:
    ----------
    source_data : dataframe (2D array)
        full data from your csv, including the 'lot_step_machines' column.
    machine_info : dataframe (2D array)
        A machine information dataframe for each machine.

    Return:
    ----------
    machine_info : dataframe (2D array)
        A machine information dataframe for each machine that already updated
        with adding some columns including 'exp_num_bad_pieces',
        'exp_pieces_going_bad', and 'exp_num_good_pieces'.

    """

    # initializing or reset the curent value to recalculate later
    machine_info['exp_num_bad_pieces'] = 0  # reset
    machine_info['exp_pieces_going_bad'] = 0  # reset
    machine_info['not_detected_bad_pieces'] = 0  # reset
    dict_exp_bad = dict(zip(machine_info.machine_name,
                            machine_info.exp_num_bad_pieces))
    dict_going_bad = dict(zip(machine_info.machine_name,
                              machine_info.exp_pieces_going_bad))
    dict_not_detected = dict(zip(machine_info.machine_name,
                                 machine_info.not_detected_bad_pieces))

    # creating dictionary for faster lookup in dataframe
    # dict(zip(keys, values)) --> to change 2 lists to single dictionary
    cur_yieldrate = dict(zip(machine_info.machine_name,
                             machine_info.em_yieldrate))

    cur_log_yieldrate = dict(zip(machine_info.machine_name,
                                 machine_info.ln_yieldrate))

    ##
    cur_missrate = dict(zip(machine_info.machine_name,
                            machine_info.em_missrate))

    cur_log_missrate = dict(zip(machine_info.machine_name,
                                machine_info.ln_missrate))

    # Calculation is going through all LOT's steps
    for _, lot_step in source_data.iterrows():

        # Optimizing: caching bad_pieces (for faster get value)
        obs_bad_pieces = lot_step.bad_pieces

        # E-Step: Ignore Zero bad Pieces
        if obs_bad_pieces > 0:

            # Get j (Lot step number (observed bad pieces))
            obs_step = lot_step.lot_step
            if obs_step == -1:  obs_step = len(lot_step.lot_step_machines)

            # Optimizing: caching lot_step_machines (for faster get value)
            lot_step_machines = lot_step.lot_step_machines

            ### BEGIN Calculate Number of Expected Bad Pieces ###
            exp_bad_pieces = []

            for step_number, machine in enumerate(lot_step_machines):

                # Calculating E-Step point B1
                bad_estimation_b1 = 1 - cur_yieldrate[machine]

                # Calculating E-Step point B2
                # callculating in Log version
                bad_estimation_b2 = 0
                previous_machines = lot_step_machines[0:step_number]
                for _machine in previous_machines:
                    bad_estimation_b2 += cur_log_yieldrate[_machine]
                # leaving log version of yield rate
                bad_estimation_b2 = np.exp(bad_estimation_b2)

                # Calculating E-Step point B3
                bad_estimation_b3 = 0
                previous_missrate_mach = lot_step_machines[step_number:obs_step - 1]
                for mach in previous_missrate_mach:
                    bad_estimation_b3 += cur_log_missrate[mach]
                # leaving log version of miss rate
                bad_estimation_b3 = np.exp(bad_estimation_b3)

                inspec_mach = lot_step_machines[obs_step - 1]
                bad_estimation_b3 = bad_estimation_b3 * (1 - cur_missrate[inspec_mach])

                # E-Step: B1 * B2 * B3
                # + EPSILON is to avoid true_divide (sum value is zero)
                bad_estimation = bad_estimation_b1 * bad_estimation_b2 * bad_estimation_b3 + EPSILON

                exp_bad_pieces.append(bad_estimation)

            # calculating % bad pieces for each machine in each LOT row
            exp_percent_bad_pieces = (exp_bad_pieces / sum(exp_bad_pieces))

            # calculating number of bad pieces for each machine in each LOT row
            exp_num_bad_pieces = (obs_bad_pieces * exp_percent_bad_pieces)

            ### END OF Calculate Percentage of Number Bad Pieces ###

            ### BEGIN Calculate Expected Pieces Going Bad ###
            # start with all zero
            exp_pieces_going_bad = np.zeros((len(exp_num_bad_pieces),))

            # reversed loop, calculate from the last index
            # https://stackoverflow.com/questions/529424/
            # traverse-a-list-in-reverse-order-in-python
            for step_number, exp_bad_pieces \
                    in reversed(list(enumerate(exp_num_bad_pieces))):

                # stop before index = 0
                if step_number > 0:
                    exp_pieces_going_bad[step_number - 1] \
                        = exp_bad_pieces \
                          + exp_pieces_going_bad[step_number]

            ### END OF Calculate Expected Pieces Going Bad ###

            ### BEGIN Calculate Am ###
            remove_last = lot_step_machines[:-1]  # Only calculate until j-1
            for step, mach in enumerate(remove_last):
                for n in range(0, step + 1):
                    dict_not_detected[mach] += exp_num_bad_pieces[n]

            ###  END OF Calculate Am ###

            # merged bad and going_bad pieces to the appropriate machine
            for step_number, machine in enumerate(lot_step_machines):
                dict_exp_bad[machine] += exp_num_bad_pieces[step_number]
                dict_going_bad[machine] += exp_pieces_going_bad[step_number]

    # after LOT's steps iteration
    # merge dictionaries to main machine information dataframe
    machine_info['exp_num_bad_pieces'] = machine_info['machine_name'].map(dict_exp_bad)
    machine_info['exp_pieces_going_bad'] = machine_info['machine_name'].map(dict_going_bad)

    machine_info['not_detected_bad_pieces'] = machine_info['machine_name'].map(dict_not_detected)

    # calculating Number of Expected Good Pieces for each machine
    machine_info['exp_num_good_pieces'] = machine_info['exp_pieces_going_bad'] \
                                          + machine_info['good_produced']

    return machine_info


def m_step(machine_info):
    """
    The M-STEP of Expectation–Maximization (EM) algorithm.
    This step is used to RE-calculate the yield rate (parameter) of each machine
    based on the estimated number of bad pieces and the estimated number of
    good pieces calculated in E-STEP of EM Algorithm.

    Parameter:
    ----------
    machine_info : dataframe (2D array)
        A machine information dataframe for each machine that already updated
        with adding some columns including 'exp_num_bad_pieces' and
        'exp_num_good_pieces'.

    Return:
    ----------
    machine_info : dataframe (2D array)
        An updated machine information dataframe for each machine which the
        'em_yieldrate' column is already recalculate based on the E-STEP
        estimation.

    """
    # total expected processed pieces (exp bad + exp good)
    machine_info['sum_exp'] = machine_info['exp_num_bad_pieces'] \
                              + machine_info['exp_num_good_pieces']

    machine_info['sum_miss'] = machine_info['not_detected_bad_pieces'] \
                               + machine_info['machine_bad_pieces']

    # RE-calculate the yield rate miss rate of each machine,
    # change NaN value with zeros
    machine_info['em_yieldrate'] = (machine_info['exp_num_good_pieces'] \
                                    / machine_info['sum_exp'] \
                                    ).replace(np.nan, 0)
    machine_info['em_missrate'] = (machine_info['not_detected_bad_pieces'] \
                                   / machine_info['sum_miss'] \
                                   ).replace(np.nan, 0)

    return machine_info


def em_algorithm(source_data, mach_info, lot_info):
    """
    Expectation–Maximization (EM) algorithm. Return most optimal or converged
    yieldrate.

    Parameter:
    ----------
    source_data : dataframe (2D array)
        full data from your csv, including the 'lot_step_machines' column.
    mach_info : dataframe (2D array)
        A machine information dataframe for each machine.

    Return:
    ----------
    machine_info : dataframe (2D array)
        An updated machine information dataframe for each machine. The column of
        'em_yieldrate' is already calculated by EM algorithm, and can be a
        benchmark for each machine's yield rate.

    """
    # initializing variables
    mach_info['em_yieldrate'] = mach_info['init_yieldrate']
    mach_info['em_missrate'] = mach_info['init_miss_rate']
    # record for EM iterations
    em_update = pd.DataFrame(data=None, columns=['last', 'current', 'diff'])

    # Start EM Iteration until Max Iteration
    for iters in range(MAX_EMA_ITERATIONS):
        # initializing
        # If some element of yieldrate <= 0 then replaces with a very small number
        mach_info['em_yieldrate'] = np.where(mach_info['em_yieldrate'] <= 0,
                                             EPSILON,
                                             mach_info['em_yieldrate'])
        mach_info['em_missrate'] = np.where(mach_info['em_missrate'] <= 0,
                                            EPSILON,
                                            mach_info['em_missrate'])
        mach_info['ln_yieldrate'] = np.log(mach_info['em_yieldrate'])
        mach_info['ln_missrate'] = np.log(mach_info['em_missrate'])
        mach_info['prev_yieldrate'] = mach_info['em_yieldrate']

        # E-STEP ###
        mach_info = e_step(source_data, mach_info)

        # M-STEP ###
        mach_info = m_step(mach_info)

        # show accuracy each iteration
        # _, accuracy, lot_bp_acc = em_accuracy(mach_info, lot_info)

        # Calculating Stop Condition
        tolerance = 1.e-3
        # it could save the update history for each iteration
        prev_yr = mach_info['prev_yieldrate'].values.copy()
        cur_yr = mach_info['em_yieldrate'].values.copy()
        diff = np.linalg.norm(prev_yr - cur_yr)

        # save to update record
        em_update.loc[len(em_update)] = [prev_yr, cur_yr, diff]

        """
        #show current progress / iteration
        cu.show_progress(iters+1, MAX_EMA_ITERATIONS,
                         messages="iteration: %d/%d;" 
                         %(iters+1, MAX_EMA_ITERATIONS),
                         force_show=True)
        """
        # stop condition
        if diff <= tolerance:
            break

    # print()

    return mach_info, em_update, iters + 1


def em_accuracy(machine_info, lot_info):
    """
    Calculating the global accuracy of the EM Algorithm result. The accuracy can
    be calculated using markov chain formula, which the multiplication of all
    the used machines in a LOT should be near its LOT's yield rate or better
    have same value (100% accuracy).

    Parameter:
    ----------
    machine_info : dataframe (2D array)
        A machine information dataframe for each machine including the updated
        'em_yieldrate'.
    lot_info : dataframe (2D array)
        insert a dataframe which consist of statistic details of each lot

    Return:
    ----------
    lot_info : dataframe (2D array)
        A dataframe which consist of statistic details of each lot including
        the accuracy of each LOT.
    acc : float
        The accuracy of the EM ALgorithm result in percentage range (0-100).

    """
    # calculating accuracy
    cur_yieldrate = dict(zip(machine_info.machine_name,
                             machine_info.em_yieldrate))

    lot_info["mch_all_yieldrate"] = 0
    lot_info["lot_accuracy"] = 0
    for row_index, lot_step in lot_info.iterrows():
        yieldrate = 1  # initial yield rate
        machine_used = lot_step.machine_used  # cache

        for machine in machine_used:
            yieldrate *= cur_yieldrate[machine]

        lot_accuracy = (100 - abs((yieldrate - lot_step.lot_yield_rate) * 100))

        lot_info.loc[lot_info.index == row_index, "mch_all_yieldrate"] = yieldrate
        lot_info.loc[lot_info.index == row_index, "lot_accuracy"] = lot_accuracy

    acc = lot_info["lot_accuracy"].mean()

    total_lot_bp = lot_info["lot_bad_pieces"].sum()
    total_machine_bp = machine_info["exp_num_bad_pieces"].sum()
    _lot_bp_accuracy = ((total_lot_bp \
                         - abs(total_lot_bp - total_machine_bp)) \
                        / total_lot_bp) * 100

    # print(total_lot_bp, total_machine_bp)

    return lot_info, acc, _lot_bp_accuracy

def mch_conf_factors(raw, mach_info, lot_info, _multi_process=False):
    """
    Generate factors for confidence:
    1. bootstrap for confidence interval
    2. # of batches using machine m
    3. # of pieces processed by machine m
    """    
    def j_mach_name (_raw_csv):
        """
        show the current j machine in every row of raw data
        """
        _raw_csv_w_j_mach = _raw_csv.copy()
        
        # load columns with machine name only
        lot_mchs = _raw_csv_w_j_mach.loc[:, _raw_csv_w_j_mach.columns.str. \
                                            startswith("lot_machine_step")]

        # get current jth machine (last non-null column)
        # https://stackoverflow.com/questions/40583482/getting-last-non-na-value-across-rows-in-a-pandas-dataframe
        _raw_csv_w_j_mach['j_mch'] = lot_mchs.ffill(axis=1).iloc[:, -1]
        
        return _raw_csv_w_j_mach
    
    def ci_bootstrap(data_df):
        # calculate confidence interval
        _data_init = data_df.tail(BOOTSTRAP_FREQ).values.tolist()
        _data_avg = np.mean(_data_init)
        
        if (_data_avg < 1) and (np.std(_data_init) > 1.e-15):
            _data_init.sort()
            _data = (_data_init,)
            
            for retry in range(MAX_RETRY):
                # using retry because sometimes bootstrap showing 
                # unknown error such as "Percentiles must be in the range [0, 100]"
                try:
                    res = bootstrap(_data, np.mean, confidence_level=0.95, n_resamples=10000)
                  
                    _width = res.confidence_interval.high - res.confidence_interval.low
                    _average = res.confidence_interval.high - (_width / 2)

                    data_df['ci_lo'] = res.confidence_interval.low
                    data_df['ci_hi'] = res.confidence_interval.high
                    data_df['ci_avg'] = _average
                    data_df['ci_width'] = _width
                except:
                    print("bootstrap failed, retrying %d/%d times. data: %s" %(retry, MAX_RETRY,
                                                                               _data_init))
                    continue #retry if failed
                break #break if succeed              
            
        elif(np.std(_data_init) <= 1.e-15):
            data_df['ci_lo'] = _data_avg
            data_df['ci_hi'] = _data_avg
            data_df['ci_avg'] = _data_avg
            data_df['ci_width'] = 0
        else:
            data_df['ci_lo'] = 1
            data_df['ci_hi'] = 1
            data_df['ci_avg'] = 1
            data_df['ci_width'] = 0
            
        return data_df
        
    bs_mach_info = mach_info['machine_name']    
    
    def sample_collection (_raw, _mach_info, _lot_info, idx=0):
        # random sampling using 80% of total batches
        drop_lots = _lot_info.sample(frac=1-BOOTSTRAP_SRS_FRAC)
        drop_lots_list = drop_lots.lot_name.to_list()
        
        raw_sample = _raw[~_raw.lot_name.isin(drop_lots_list)].copy()        
        bs_mach_yr, _, _ = em_algorithm(raw_sample, _mach_info, _lot_info)
        
        bs_mach_yr = bs_mach_yr[['em_yieldrate', 'machine_name']]
        bs_mach_yr = bs_mach_yr.rename(columns={'em_yieldrate': "bs_yr_%d" %idx})
        
        return bs_mach_yr
    
    bs_mach_info = mach_info['machine_name']  
    
    if _multi_process:
        ### BEGIN Multiprocessing
        cores = mp.cpu_count() 
        # less cpu for a better stability
        used_cores = (cores - 1) if (cores >= 2) else cores
        #used_cores = 1
        print("Number of cpu %d, used %d" % (cores, used_cores))
        
        results = Parallel(n_jobs=used_cores,)\
                    (delayed(sample_collection)
                                 (raw, mach_info, lot_info, i)
                            for i in range(BOOTSTRAP_FREQ) 
                            )
        # append result into single dataframe
        for i in range(len(results)):
            bs_mach_info = pd.merge(bs_mach_info, results[i], 
                                           on='machine_name')
        
    else:    
        for i in range(BOOTSTRAP_FREQ):
            #print("bootstrap est %d" %i)
            sample_i = sample_collection(raw, mach_info, lot_info, i)
            
            # append result into single dataframe
            bs_mach_info = pd.merge(bs_mach_info, sample_i, 
                                           on='machine_name')
    
    # apply CI
    bs_mach_info = bs_mach_info.apply(ci_bootstrap, axis=1)
    mach_info['ci_lo'] = bs_mach_info['ci_lo']
    mach_info['ci_hi'] = bs_mach_info['ci_hi']  
    mach_info['ci_avg'] = bs_mach_info['ci_avg']  
    mach_info['ci_width'] = bs_mach_info['ci_width']
    
    batches_lite = raw.loc[raw['lot_step'].values == -1].copy()
    batches_lite = batches_lite.iloc[:,:-5]
    # add new column with current j machine name
    emraw_csv = j_mach_name (raw)
    # analyze number of batches and pieces 
    for idx, row in mach_info.iterrows():
        # g_truth col --> mach_name, yieldrate, mach_insp, freq_used
        mch_name = row['machine_name']    
             
        # analysis: size of inspection block        
        batch_found = emraw_csv.loc[emraw_csv['j_mch'] == mch_name].copy()
        # sum processed pieces
        mach_info.loc[idx, 'proc_pieces']  = batch_found['processed'].sum(axis=0)
        
        # find batch that use mch_name
        _batch_sel = batches_lite.loc[(batches_lite == mch_name).any(axis=1)]
        mach_info.loc[idx, 'batches']  = len(_batch_sel)
    
    return mach_info

#############################|##############################
#                         REPORT                           #
############################################################
def create_report(machine_info, em_iter_info, target_filename):
    """
    Saving report to csv file. this will save only the yield rate and bad pieces
    per machine step.

    Parameter:
    ----------
    machine_statistic : DataFrame (2D array)
        a machine statistic dataframe which already processed by EM algorithm
        and have bad pieces information
    target_filename : str
        given path and filename such as ``"folder/yourfilename.csv"``

    """
    # sort by yieldrate from the smallest and get some report summary
    machine_info = machine_info.sort_values(['ci_avg', 'ci_width'])
    report = machine_info.copy()
    
    # renaming columns to match to the dictionary documents    
    report = report.rename(columns={'em_yieldrate': 'YieldRate',
                                    'em_missrate': 'Missrate',
                                    'exp_num_bad_pieces': 'BadPiece',
                                    'machine_name': 'Machine',
                                    'proc_pieces': 'ProcessedPieces',
                                    'batches': 'NumBatches',
                                    'ci_lo': 'CIlow', 'ci_hi': 'CIhigh',
                                    'ci_avg': 'CIavg', 'ci_width': 'CIwidth' })

    report = report.reset_index(drop=True)

    # changing display options to set print all column
    pd.set_option('display.max_columns', None)
    report = report.round(5)  # cleaning some decimal
    # print(report)

    # saving report summaries
    report.to_csv(target_filename, sep=',', index=False)
    
    # saving #of iterations and MSE
    #iter_csv_fname = "%s%s.csv" %(target_filename[:-4], "_iter") #remove.csv from target filename
    #em_iter_info.to_csv(iter_csv_fname, sep=',', index=False)

    # check file
    check = pd.read_csv(target_filename, index_col=0)
    print("Saving %s Success" % (target_filename)
          if check.size == check.size \
              else "Saving Report %s failed" % (target_filename))

    # reset display options
    pd.reset_option('^display.', silent=True)
    
    
#############################|##############################
#                     MAIN FUNCTIONS                       #
############################################################
def main(source_filename, report_filename, multiprocess=False):
    """
    Main procedure for running the EM Algorithm

    Parameter:
    ----------
    source_filename, report_filename : str
        given path and filename such as ``"folder/yourfilename.csv"``

    """
    start = time.time()
    #print('Process Start: ', source_filename)

    # STEP 1: get raw data
    #print('Step 1: Reading CSV')
    raw_data = read_csv_files(source_filename)
    # cu.check_time(start, time.time())

    # start_split = time.time()
    #print('Step 2: One Hot Encoding of Machine Steps')
    # STEP 2: create one hot encoding for each machine step
    # and add lot machine usage to raw_data
    onehot_lot_last_step, machine_information, \
    raw_data = get_onehot_machine(raw_data)

    # cu.check_time(start_split, time.time())

    # start_split = time.time()
    #print('Step 3: Preprocesing: Lot Information')
    # STEP 3: calculate each LOT's statistics
    lot_information = get_lot_information(raw_data)
    # cu.check_time(start_split, time.time())

    # start_split = time.time()
    #print('Step 4: Preprocesing: Machine Information')
    # STEP 4: calculate each machine's statistics
    machine_information = get_machine_information(onehot_lot_last_step,
                                                  machine_information,
                                                  lot_information)
    # cu.check_time(start_split, time.time())

    # start_split = time.time()
    #print('Step 5: Running EM Algorithm with bootstrap')
    # STEP 5.1: calculate machine's yield rate using EM Algorithm
    machine_information = mch_conf_factors(raw_data, machine_information,
                                                  lot_information, multiprocess)
    # STEP 5.2: calculate machine's yield rate using EM Algorithm (100% data)
    machine_information, em_iter_res, iters = em_algorithm(raw_data, machine_information,
                                                           lot_information)

    # get EM accuracy
    lot_information, accuracy, lot_bp_accuracy = em_accuracy(machine_information,
                                                             lot_information)
    #print("accuracy = ", accuracy)
    # cu.check_time(start_split, time.time())

    # STEP 6: creating reports
    #print('Step 6: Saving Report')
    create_report(machine_information, em_iter_res, report_filename)

    # print the processing time
    # print('Process End')
    #print('Process End: ', source_filename)
    cu.check_time(start, time.time())
    spend_time = time.time() - start
    # print()

    return raw_data, onehot_lot_last_step, machine_information, lot_information, \
           machine_information, em_iter_res, iters, accuracy, lot_bp_accuracy, spend_time


if __name__ == "__main__":
    raw_data, onehot_lot_last_step, machine_information, lot_information, \
    machine_information, em_iter_res, iters, accuracy, lot_bp_accuracy, spend_time = main(
        RAWDATA_CSV_PATH_FILENAME, REPORT_PATH_FILENAME, USE_MULTIPROCESS)
