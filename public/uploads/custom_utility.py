# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 11:50:43 2019

@author: Faster

Importing:
    first, this file should be in the same folder as the importer, then

    >>> import custom_utility as cu

    or using Python's Relative imports, Every leading dot is another higher
    level in the hierarchy beginning with the current directory.

    >>> import ...app.folder.file as cu

Features:
    show_progress:
        Print the current progress in percentage
    check_time:
        Calculate and print time passes from ``time_start`` to ``time_end``.
"""

import sys

def show_progress(current_num, max_num, step_print=5, messages='',
                  force_show=False):
    """
    Print the current progress in percentage

    Parameter:
    ----------
    current_num, max_num: int
        The current number and the maximum number of the progress or loop
        (if used)
    step_print: int (optional)
        (unit is in percent) How often the message should be printed, the
        default is every 5 percent of ``max_num``.
    message: string (optional)
        Insert message in the beginning of printing percentage
    force_show: bool (optional)
        Printing for each progress no matter how small the progress is. If set
        to ``True``, It will disabled or ignore the step_print option.

    This function does not return anything, It prints as function is called.

    >>> show_progress(3, 100, messages='Current progress', force_show=True)
    Current progress 3%
    """

    #calculating real progress
    progress = (current_num/max_num*100)

    # change to int, avoid too many printing, except is forced to
    progress = int(progress) if (force_show or (progress % 1) >= 0.93
                                 or (progress % 1) <= 0.07) else progress

    #print progress
    if (progress % step_print) == 0 or force_show:
        msg = "\r%s %d%% " %(messages, progress)
        sys.stdout.write(msg)
        sys.stdout.flush()

def check_time(time_start, time_end):
    """
    calculate time passed from ``time_start`` to ``time_end``.

    Parameter:
    ----------
    time_start, time_end: time float format
        the argument should be in ``float`` numbers. It can use import time
        and use time.time() as input

    this function does not return anything, It prints as function is called.

    >>> check_time(11890.88, 21192.99) #sample time
    Time elapsed: 2 H : 35 M : 2.11 Sec
    """

    t_sec = round(time_end - time_start, 3)
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    print('Time elapsed: {} H : {} M : {} Sec'. \
          format(round(t_hour), round(t_min), round(t_sec, 3)))
