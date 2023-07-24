# General:
import sys
from multiprocessing import *
import numpy as np
import pandas as pd
import datetime as dt
import ast

# Visualization:
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

# Graph analyses:
from graph_tool.all import *

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

def preprocess_micromobility(divvy_data, Divvy_Stations, divvy_data_distances, start_time, end_time):

    divvy_data['start_time'] = pd.to_datetime(divvy_data['start_time'])
    divvy_data['end_time'] = pd.to_datetime(divvy_data['end_time'])
    divvy_data['bike_id'] = (divvy_data['bike_id']).astype(int)
    divvy_data['start_station_id'] = (divvy_data['start_station_id']).astype(int)
    divvy_data['end_station_id'] = (divvy_data['end_station_id']).astype(int)
    divvy_data['duration'] = divvy_data['duration'].str.replace(r',', '')
    divvy_data['duration'] = (divvy_data['duration']).astype(float).astype(int)

    divvy_data.loc[divvy_data['member_gender'] == 'Male', 'member_gender'] = 1
    divvy_data.loc[divvy_data['member_gender'] == 'Female', 'member_gender'] = 0

    divvy_data['age'] = (pd.to_datetime("today").year - divvy_data['member_birth_day'])

    divvy_data.loc[divvy_data['user_type'] == 'Subscriber', 'user_type'] = 1
    divvy_data.loc[divvy_data['user_type'] == 'Customer', 'user_type'] = 2

    divvy_data['start_day_of_week'] = divvy_data['start_time'].dt.strftime('%A')
    divvy_data['end_day_of_week'] = divvy_data['end_time'].dt.strftime('%A')
    divvy_data = divvy_data[(divvy_data['start_station_id'].isin(Divvy_Stations['ID'])) &
            (divvy_data['end_station_id'].isin(Divvy_Stations['ID']))]


    divvy_data = divvy_data[(divvy_data['start_time']>=start_time) & (divvy_data['end_time']<=end_time)]

    ## correcting data
    # Single out trips with unrealistic attributes
    #  Age <= 80 or null
    divvy_data['age'].hist()
    print('# of trips with age higher than 80 or null: ', divvy_data[divvy_data['age']>80].shape)
    divvy_data = divvy_data[(divvy_data['age']<=80)|(divvy_data['age'].isna())]
    # A large number of ages are null
    # Let's use the distribution of existing records to impute ages for the reocrds without age value

    np.random.seed()
    pot = np.random.choice(len(list(divvy_data[~divvy_data['age'].isna()].groupby('age').count()['trip_id'].index.astype(int))),
                        divvy_data[divvy_data['age'].isna()].shape[0],
                        p=list(divvy_data[~divvy_data['age'].isna()].groupby('age').count()['trip_id'] / \
                                divvy_data[~divvy_data['age'].isna()].groupby('age').count()['trip_id'].sum()))
    aux = 0
    for idx, row in divvy_data.iterrows():
        if row['age'] == None:
            divvy_data.loc[idx, 'age'] = list(divvy_data[~divvy_data['age'].isna()].groupby('age').count()['trip_id'].index.astype(int))[pot(aux)]
            aux = aux + 1

    # A trip more than 5 hours makes less sense
    # The longest trip is 35km in 2.5 hour
    #  300 <= duration : 522189 records
    #  5*3600 <= duration : 2605  records
    print('# of trips with duration larger than 5 hours: ', divvy_data[divvy_data['duration']>5*3600].shape)
    divvy_data = divvy_data[divvy_data['duration']<=5*3600]

    divvy_data = divvy_data.merge(divvy_data_distances, left_on=divvy_data['trip_id'], right_on=divvy_data_distances.index, how='left')
    divvy_data.drop('key_0', inplace=True, axis=1)
    # Speed
    divvy_data['speed'] = divvy_data['distance'] / divvy_data['duration']
    # Speed is less likely to get more than 25 m/h (~12m/s)
    print('# of trips with speed higher than 25 m/h: ', divvy_data[divvy_data['speed']>12].shape)
    divvy_data = divvy_data[divvy_data['speed']<=12]

    return divvy_data

def impute_bikes_and_trips (precentage_new, divvy_data, NPROCESSOR):
    # Impute new trips
    # New bikes
    all_samples = pd.DataFrame(columns=divvy_data.columns)
    divvy_data_plus = divvy_data
    bikes_plus = round(divvy_data_plus['bike_id'].unique().shape[0] * precentage_new / 100)
    max_bikes_id = divvy_data_plus['bike_id'].max()
    uniq_bikes = pd.DataFrame(divvy_data_plus['bike_id'].unique(), columns=['bike_id']).sort_values('bike_id')
    aux = pd.DataFrame(range(max_bikes_id + 1, max_bikes_id + bikes_plus + 1), columns=['bike_id'])

    cores = partition_bikes(NPROCESSOR, aux)

    processes = []
    workers = []
    qq = []

    for process in range(cores.shape[0]):
        q = Queue()
        p = Process(
            target=impute_bikes_and_trips_parallel,
            args=(aux[aux['bike_id'].isin(ast.literal_eval(cores.iloc[process, 0]))], divvy_data, uniq_bikes, q,))
        workers.append(p)
        p.start()
        processes.append(p)
        qq.append(q)

    for item in qq:
        s = item.get()
        all_samples = pd.concat([all_samples, s], axis=0)

    for process in processes:
        process.join()

    return all_samples

def impute_bikes_and_trips_parallel (bikes, divvy_data, uniq_bikes, q):
    # Sampling from bike-trip distribution
    bike_distribution = divvy_data.groupby('bike_id').count()['trip_id']
    bike_distribution.rename({'trip_id': 'trips'}, inplace=True, axis=1)
    weights = bike_distribution / divvy_data.shape[0]

    i = 0
    all_samples = pd.DataFrame(columns=divvy_data.columns)

    for idx, row in bikes.iterrows():

        print('Processing bike:', row['bike_id'])

        # Sample from unique bikes by number of trips
        smpls = uniq_bikes.sample(n=1, replace=True, weights=weights)
        # Extract sample trips associated with the sample bike
        smpls_trips = divvy_data[divvy_data['bike_id'] == int(smpls['bike_id'])]
        smpls_trips = smpls_trips.reset_index()
        try:
            smpls_trips.drop('level_0', axis=1, inplace=True)
        except:
            pass
        # Change the trip id and bike id
        smpls_trips['trip_id'] = pd.DataFrame(
            range(divvy_data['trip_id'].max() + 1, divvy_data['trip_id'].max() + smpls_trips.shape[0] + 1))
        smpls_trips['bike_id'] = bikes.loc[idx, 'bike_id']
        smpls_trips.drop('index', inplace=True, axis=1)
        # update trip table
        all_samples = pd.concat([all_samples, smpls_trips], axis=0)

    q.put(all_samples)

def partition_bikes(NPROCESSOR, aux):
    cores = pd.DataFrame(index = range(NPROCESSOR), columns=['bike_id'])
    cores['bike_id'] = '[]'
    core = 0

    while aux.shape[0] !=0:
        print(aux.shape[0])
        if core < NPROCESSOR:
            cores.loc[core, 'bike_id'] = str(ast.literal_eval(cores.loc[core, 'bike_id']) + [aux.iloc[0, 0]])
            aux = aux.drop(aux.index[0], axis=0)
            core+=1
        else:
            core = 0
            cores.loc[core, 'bike_id'] = str(ast.literal_eval(cores.loc[core, 'bike_id']) + [aux.iloc[0, 0]])
            aux = aux.drop(aux.index[0], axis=0)
            core+=1

    return cores

def choose_bikes(bikes, start_time, divvy_data, INITIAL_CONTAMINATED_BIKES, SURVIVAL_TIME):
    deltatime = dt.timedelta(minutes=SURVIVAL_TIME)
    potential_bikes = divvy_data[(divvy_data['start_time']<start_time) & (divvy_data['end_time']>start_time)]['bike_id'].unique()
    contaminated_bikes = pd.DataFrame(np.random.choice(potential_bikes, INITIAL_CONTAMINATED_BIKES), columns=['bike_id'])
    contaminated_bikes['start_of_contamination'] = start_time
    contaminated_bikes['end_of_contamination'] = start_time + deltatime

    return contaminated_bikes