import pandas as pd
import os 
import glob
import math
import numpy as np
import pyarrow.parquet as pq
from pandas import read_csv
#import matplotlib.pyplot as plt

import sys
from os import path
# to be able to access sister folders
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

import Algos.Util as Util
import Algos.MetricsCalculator as Metrics
from Algos.Metadata import Metadata 

campaign_ids = [597165, 837817, 722100] # 809153
#x.iloc[1] = dict(x=9, y=99)
meta = Metadata(0)
input = open("{0}/multi_hindsight_sorted_time_impressions.csv".format(meta.path), "r")
header = input.readline()

output = open("{0}/multi_hindsight_sorted_time_impressions_2.csv".format(meta.path), "w")
output.write(header)

for line in input:
	new_line = line.replace(",,",",")
	output.write(new_line)

output.close()
input.close()