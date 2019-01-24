import sys
import os
import pandas as pd

from bokeh.palettes import Category10
import matplotlib.pyplot as plt

from file_cache.utils.util_pandas import *
from file_cache.cache import file_cache


#Adjust the working folder
file_folder = globals()['_dh'][0]
wk_dir = os.path.dirname(file_folder)
os.chdir(wk_dir)
from core.feature import *
logging.getLogger().setLevel(logging.DEBUG)