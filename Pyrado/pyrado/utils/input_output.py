# Copyright (c) 2020, Fabio Muratore, Honda Research Institute Europe GmbH, and
# Technical University of Darmstadt.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of Fabio Muratore, Honda Research Institute Europe GmbH,
#    or Technical University of Darmstadt, nor the names of its contributors may
#    be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL FABIO MURATORE, HONDA RESEARCH INSTITUTE EUROPE GMBH,
# OR TECHNICAL UNIVERSITY OF DARMSTADT BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import numpy as np
from colorama import Fore, Style
from typing import Sequence, Iterable

import pyrado
from pyrado.sampling.step_sequence import StepSequence


def insert_newlines(string: str, every: int) -> str:
    """
    Inserts multiple line breaks.

    :param string: input string to be broken into multiple lines
    :param every: gap in number of characters between every line break
    :return: the input sting with line breaks
    """
    return '\n'.join(string[i:i + every] for i in range(0, len(string), every))


def print_cbt(msg: str, color: str = '', bright: bool = False, tag: str = ''):
    """
    Print a colored (and bright) message with a tag in the beginning.

    :param msg: string to print
    :param color: color to print in, default `''` is the IDE's/system's default
    :param bright: flag if the message should be printed bright
    :param tag: tag to be printed in brackets in front of the message
    """
    brgt = Style.BRIGHT if bright else ''

    if not isinstance(tag, str):
        raise pyrado.TypeErr(given=tag, expected_type=str)
    else:
        if tag != '':
            tag = f'[{tag}] '

    color = color.lower()
    if color in ['', 'w', 'white']:
        print(brgt + tag + msg + Style.RESET_ALL)
    elif color in ['y', 'yellow']:
        print(Fore.YELLOW + brgt + tag + msg + Style.RESET_ALL)
    elif color in ['b', 'blue']:
        print(Fore.BLUE + brgt + tag + msg + Style.RESET_ALL)
    elif color in ['g', 'green']:
        print(Fore.GREEN + brgt + tag + msg + Style.RESET_ALL)
    elif color in ['r', 'red']:
        print(Fore.RED + brgt + tag + msg + Style.RESET_ALL)
    elif color in ['c', 'cyan']:
        print(Fore.CYAN + brgt + tag + msg + Style.RESET_ALL)
    else:
        raise pyrado.ValueErr(given=color, eq_constraint="'y', 'b', 'g', 'r', or 'c'")


def select_query(items,
                 max_display=10,
                 fallback=None,
                 item_formatter=str,
                 header: str = 'Available options:',
                 footer: str = 'Please enter the number of the option to use.'):
    """ Ask the user to select an item out of the given list. """

    # Truncate if needed
    print(header)
    if max_display is not None and len(items) > max_display:
        items = items[:max_display]
        print(f'(showing the latest {max_display})')

    # Display list
    for i, exp in enumerate(items):
        print('  ', i, ': ', item_formatter(exp))

    print(footer)

    # Repeat query on errors
    while True:
        sel = input()

        # Check if sel is a number, if so use it.
        if sel == '':
            # first item is default
            return items[0]
        elif sel.isdigit():
            # Parse index
            sel_idx = int(sel)
            if sel_idx < len(items):
                return items[sel_idx]
            # Error
            print('Please enter a number between 0 and ', len(items) - 1, '.')
        elif fallback is not None:
            # Use fallback if any
            fres = fallback(sel)
            if fres is not None:
                return fres
            # The fallback should report it's own errors
        else:
            print('Please enter a number.')


def ensure_math_mode(inp: [str, Sequence[str]], no_subscript: bool = False) -> [str, list]:
    """
    Naive way to ensure that a sting is compatible with LaTeX math mode for printing.

    :param inp: input string
    :param no_subscript: force no subscript (sometimes there might be problems due to double subscript
    :return s: sting in math mode
    """
    if isinstance(inp, str):
        if inp.count('$') == 0:
            # There are no $ symbols yet
            if not inp[0] == '$':
                inp = '$' + inp
            if not inp[-1] == '$':
                inp = inp + '$'
        elif inp.count('$')%2 == 0:
            # There is an even number of $ symbols, so we assume they are correct and do nothing
            pass
        else:
            raise pyrado.ValueErr(msg=f"The string {inp} must contain an even number of '$' symbols!")
        if no_subscript:
            inp = ensure_no_subscript(inp)

    elif inp is None:
        return None  # in case there a Space has 1 one dimension but no labels

    elif isinstance(inp, Iterable):
        # Do it recursively
        return [ensure_math_mode(s, no_subscript) if s is not None else None for s in inp]  # skip None entries

    else:
        raise pyrado.TypeErr(given=inp, expected_type=[str, list])

    return inp


def ensure_no_subscript(inp: [str, Sequence[str]]) -> [str, list]:
    """
    Naive way to ensure that a sting is compatible with LaTeX for printing by removing the math mode symbols.

    :param inp: input string or iterable of stings
    :return: sting without subscript
    """
    if isinstance(inp, str):
        return inp.replace('_', '\_')  # just hoping that there is no \_ which we are replacing with \\_

    elif inp is None:
        return None  # in case there a Space has 1 one dimension but no labels

    elif isinstance(inp, Iterable):
        return [s.replace('_', '\_') if s is not None else None for s in inp]  # skip None entries

    else:
        raise pyrado.TypeErr(given=inp, expected_type=[str, list])


def num_iter_from_rollouts(ros: [Sequence[StepSequence], None],
                           concat_ros: [StepSequence, None],
                           batch_size: int) -> int:
    """
    Get the number of iterations from the given rollout data.

    :param ros: multiple rollouts
    :param concat_ros: concatenated rollouts
    :param batch_size: number of samples per batch
    :return: number of iterations (e.g. used for the progress bar)
    """
    if ros is None:
        assert concat_ros is not None
        return (concat_ros.length + batch_size - 1)//batch_size
    else:
        return (sum(ro.length for ro in ros) + batch_size - 1)//batch_size


def color_validity(data: np.ndarray, valids: np.ndarray) -> list:
    """
    Color the entries of an data array red or green depending on the if the entries are valid.

    :param data: ndarray containing the data to print
    :param valids: ndarray containing boolian integer values deciding gor the color (1 --> green, 0 --> red)
    :return: list of stings
    """

    def color_validity(v: int) -> str:
        return Fore.GREEN if v == 1 else Fore.RED

    return [color_validity(v) + str(ele) + Style.RESET_ALL for (ele, v) in zip(data, valids)]