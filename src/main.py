# General:
import os
import sys
import numpy as np
import pandas as pd
import timeit
from itertools import chain

# Spatial:
import geopandas as gpd
from shapely.geometry import MultiLineString, LineString, Point, Polygon, GeometryCollection

# Visualization:
import matplotlib as mpl
import synthesizer as syn

os.chdir('H:/My Drive/TranEpiSim/')
# 1. Road file
road = gpd.read_file('data/road/roads.shp')

# 2. Demographic profile
dp = gpd.read_file('data/dp/dp.shp').set_index('GEOID10')
dp['portion'] = dp.apply(lambda tract: tract.geometry.area / tract.Shape_Area, axis=1)

# 3. Schools and daycares
school = gpd.read_file('data/education/school.shp')
daycare = gpd.read_file('data/education/day_care.shp')

# 4. Number of establishments per county per size
cbp = pd.read_csv('data/cbp/cbp10co.zip')
cbp = cbp[(cbp.naics.str.startswith('-'))] #All types of establishments included
cbp['fips'] = cbp.fipstate.map("{:02}".format) + cbp.fipscty.map("{:03}".format)
cbp = cbp.set_index('fips')

# 5. Origin (home) - destination (job) at census-tract level
od = pd.read_csv('data/od/tract-od15Cook.csv', dtype={i:(str if i<2 else int) for i in range(6)})

#Add workplace counts and sizes to dp
dp['WP_CNT'] = syn.number_of_wp(dp,od,cbp)
dp['WP_PROBA'] = dp.WP_CNT.map(syn.wp_proba)

# Create a unified file for education
school = syn.clean_schools(school,daycare)

population = []
errors = []
wps = []

dp.apply(lambda t: syn.synthesize(t,od,road,school,errors, population, wps, dp),axis=1)

