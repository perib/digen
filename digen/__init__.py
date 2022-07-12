# -*- coding: utf-8 -*-

"""
Copyright (c) 2020 Patryk Orzechowski | Epistasis Lab | University of Pennsylvania

DIGEN was developed at the University of Pennsylvania by Patryk Orzechowski (patryk.orzechowski@gmail.com).

Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import pickle
try:
    import importlib.resources as importlib_resources

except ImportError:
    # In PY<3.7 fall-back to backported `importlib_resources`.
    import importlib_resources

def initialize():
    return pickle.loads(importlib_resources.read_binary(__name__, 'chartsdata.pkl'))

def load_datasets():
    return importlib_resources.read_text(__name__, 'stats.csv')

from .benchmark import (
    Benchmark
)

from .dataset import (
    Dataset
)

from . import defaults

from .benchmark import __version__