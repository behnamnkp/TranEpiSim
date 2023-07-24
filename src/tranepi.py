# General:
import sys
from multiprocessing import *
import numpy as np
import pandas as pd
import datetime as dt

# Spatial:
import geopandas as gpd

# Visualization:
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

# Graph analyses:
from graph_tool.all import graph_tool as gt
from graph_tool.all import *
import networkx as nx
#import cairo

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

def preprocess_population(people, dp, wps2, school2, Divvy_Stations_Thiessen):

    people = gpd.sjoin(people, dp.loc[:, ['GEOID10', 'geometry']], how="left", op='within')
    people.drop('index_right', inplace=True, axis=1)
    people.rename({'GEOID10': 'GEOID10_hhold'}, inplace=True, axis=1)

    people = people.merge(wps2.loc[:, ['GEOID10_wps', 'geometry_wps']], left_on=people['wp'], right_on=wps2.index,
                        how='left')
    people.drop('key_0', inplace=True, axis=1)

    people = people.merge(school2.loc[:, ['GEOID10_sch', 'geometry_sch']], left_on=people['wp'], right_on=school2.index,
                        how='left')
    people.drop('key_0', axis=1, inplace=True)

    people['geometry_outside'] = people['geometry_wps']
    mask = people['geometry_outside'].isnull()
    people.loc[mask, 'geometry_outside'] = people.loc[mask, 'geometry_sch']
    mask = people['geometry_outside'].isnull()
    people.loc[mask, 'geometry_outside'] = people.loc[mask, 'geometry']

    people = gpd.sjoin(people, Divvy_Stations_Thiessen.loc[:, ['ID', 'geometry']], how="left", op='within')
    people.drop('index_right', inplace=True, axis=1)
    people.rename({'ID': 'station_hhold'}, inplace=True, axis=1)
    people = people.drop_duplicates('index')

    people.rename({'geometry': 'geometry_hhold', 'geometry_outside': 'geometry'}, inplace=True, axis=1)
    people = gpd.sjoin(people, Divvy_Stations_Thiessen.loc[:, ['ID', 'geometry']], how="left", op='within')
    people.drop('index_right', inplace=True, axis=1)
    people.rename({'ID': 'station_outside'}, inplace=True, axis=1)
    people.rename({'geometry': 'geometry_outside', 'geometry_hhold': 'geometry'}, inplace=True, axis=1)
    people = people.drop_duplicates('index')

    return people

def preprocess_micromobility(divvy_data, start_time, end_time):

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

def spread(cores, G, state, beta, q):

    new = np.array([])
    for v in cores:
        # Filter infectious individuals in population
        # Loop over the current neighbors to calculate probability of infection for them
        w = G.get_out_neighbors(v, vprops=[state])
        if w.shape[0]>0:
            w = w[w[:, 1]==1]
            for i in range(w.shape[0]):
                n = G.get_out_neighbors(w[i,0], vprops=[state])
                neighbors = G.get_out_degrees([w[i,0]])
                sicks = n[n[:, 1]==2].shape[0]
                # Estiamte risk of infection for individual w with at least one infectious neighbor
                if neighbors!=0:
                    risk = (sicks * beta) / neighbors
                else:
                    risk = 0
                # Probability of infection for individual w
                pi = 1 - np.exp(-risk)
                # Status of individual w at time step t
                if pi[0] != 0:
                    np.random.seed()
                    infected = np.random.choice(np.arange(0, 2), p=[1 - pi[0], pi[0]])
                else:
                    infected = 0
                # if infected, add the individual to the list of new cases for time step t
                if infected == 1:
                    new = np.append(np.array(w[i,0]), new)
    print('new', new.shape)
    q.put([new])

def trip_partitioning(NPROCESSOR, trips):
    cores = pd.DataFrame(index = range(NPROCESSOR), columns=['stations', 'trips'])
    freq = trips.groupby('start_station_id').count()['trip_id'].sort_values(ascending=False)
    cores['stations'] = '[]'
    cores['trips'] = 0
    core = 0

    while freq.shape[0] !=0:
        if core != NPROCESSOR:
            cores.loc[core, 'stations'] = str(ast.literal_eval(cores.loc[core, 'stations']) + [freq.index[0]])
            # str(cores.loc[core, 'stations']) + ', ' + str(freq.index[0])
            cores.loc[core, 'trips'] = cores.loc[core, 'trips'] + freq.iloc[0]
            freq = freq.drop(freq.index[0], axis=0)
            core+=1
        else:
            core = 0
            freq = freq.sort_values(ascending=True)
            cores.loc[core, 'stations'] = str(ast.literal_eval(cores.loc[core, 'stations']) + [freq.index[0]])
            cores.loc[core, 'trips'] = cores.loc[core, 'trips'] + freq.iloc[0]
            freq = freq.drop(freq.index[0], axis=0)
            core+=1

    cores = cores[cores['trips']!=0]

    return cores

def assign_cyclist (data, people, current_station, q):
    people_pot = people
    people_pot = people_pot.rename({'index': 'people_id'}, axis=1)
    aux = pd.DataFrame(columns=['trip_id', 'cyclist'])
    aux2 = data['start_station_id'].unique()
    if people_pot.shape[0] != 0:
        for item in aux2:
            mask = data['start_station_id'] == item
            counts = mask[mask == True].shape[0]
            cyclists = people_pot[people_pot[current_station] == item]
            if cyclists.shape[0] != 0 and cyclists.shape[0] >= counts:
                cyclist_sample = cyclists.sample(n=counts)
            elif cyclists.shape[0] != 0 and cyclists.shape[0] < counts:
                cyclist_sample1 = cyclists.sample(n=cyclists.shape[0])
                cyclist_sample2 = people_pot.sample(n=counts - cyclists.shape[0])
                cyclist_sample = pd.concat([cyclist_sample1, cyclist_sample2], axis=0)
            else:
                cyclist_sample = people_pot.sample(n=counts)
            aux3 = data.loc[mask, 'trip_id'].reset_index()['trip_id']
            aux4 = cyclist_sample.reset_index()['index']
            aux5 = pd.concat([aux3, aux4], axis=1)
            aux5 = aux5.rename({'index': 'cyclist'}, axis=1)
            aux = pd.concat([aux, aux5], axis=0)

        q.put(aux)

    else:
        print('*************** these stations have no population! ******************', data[:, 2])
        print(people_pot.shape)

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

def choose_bikes(bikes, start_time, divvy_data, INITIAL_CONTAMINATED_BIKES):
    deltatime = dt.timedelta(minutes=SURVIVAL_TIME)
    potential_bikes = divvy_data[(divvy_data['start_time']<start_time) & (divvy_data['end_time']>start_time)]['bike_id'].unique()
    contaminated_bikes = pd.DataFrame(np.random.choice(potential_bikes, INITIAL_CONTAMINATED_BIKES), columns=['bike_id'])
    contaminated_bikes['start_of_contamination'] = start_time
    contaminated_bikes['end_of_contamination'] = start_time + deltatime

    return contaminated_bikes

def bike_to_human_infection(c, beta):
    contaminated_bikes = c[c['health_status_bikes']==1]['count']
    clean_bikes = c[c['health_status_bikes']==0]['count']
    if contaminated_bikes.shape[0]==0 and clean_bikes.shape[0]!=0:
        contaminated_bikes=0
        m = contaminated_bikes + list(clean_bikes)[0]
        risk = (contaminated_bikes * beta) / m
    elif contaminated_bikes.shape[0]!=0 and clean_bikes.shape[0]==0:
        clean_bikes=0
        m = list(contaminated_bikes)[0] + clean_bikes
        risk = (list(contaminated_bikes)[0] * beta) / m
    elif contaminated_bikes.shape[0]==0 and clean_bikes.shape[0]==0:
        clean_bikes=0
        contaminated_bikes=0
        m = 0
        risk = 0
    else:
        m = list(contaminated_bikes)[0] + list(clean_bikes)[0]
        risk = (list(contaminated_bikes)[0] * beta) / m
    # Probability of infection for human
    pi = 1 - np.exp(-risk)
    # Status of individual
    np.random.seed()
    infected = np.random.choice(np.arange(0, 2), p=[1 - pi, pi])

    return infected

def human_to_bike_contamination(b, beta):
    infected_people = b[b['health_status_human']==2]['count']
    healthy_people = b[b['health_status_human']==1]['count']
    if infected_people.shape[0]==0 and healthy_people.shape[0]!=0:
        infected_people=0
        n = infected_people + list(healthy_people)[0]
        risk = (infected_people * beta) / n
    elif infected_people.shape[0]!=0 and healthy_people.shape[0]==0:
        healthy_people=0
        n = list(infected_people)[0] + healthy_people
        risk = (list(infected_people)[0] * beta) / n
    elif infected_people.shape[0]==0 and healthy_people.shape[0]==0:
        infected_people=0
        healthy_people=0
        n=0
        risk=0
    else:
        n = list(infected_people)[0] + list(healthy_people)[0]
        risk = (list(infected_people)[0] * beta) / n
    # Probability of contamination for bike
    pi = 1 - np.exp(-risk)
    # Status of bike at time step t
    np.random.seed()
    contaminated = np.random.choice(np.arange(0, 2), p=[1 - pi, pi])

    return contaminated

def human_bike_interaction(data, contaminated_bikes, beta, start_time, period, q):
    data = data.copy()
    # Bike-to-human infection
    infected_human = pd.DataFrame(columns=['cyclist', 'start_of_infection', 'end_of_infection'])
    data.loc[:, ['health_status_bikes']] = np.nan
    for idx, trip in data.iterrows():
        if trip['bike_id'] in contaminated_bikes['bike_id'].unique():
            cont_s = list(contaminated_bikes[contaminated_bikes['bike_id']==trip['bike_id']]['start_of_contamination'])[0]
            cont_e = list(contaminated_bikes[contaminated_bikes['bike_id']==trip['bike_id']]['end_of_contamination'])[0]
            if (cont_s >= trip['start_time'] and cont_s <= trip['end_time']) or (cont_e >= trip['start_time'] and cont_e <= trip['end_time']) or (trip['start_time'] >= cont_s  and trip['end_time'] > cont_s and trip['start_time'] < cont_e  and trip['end_time'] <= cont_e):
                data.loc[idx, 'health_status_bikes'] = 1
        else:
            data.loc[idx, 'health_status_bikes'] = 0

    cycling_frequency =  pd.DataFrame(data.groupby(['cyclist', 'health_status_bikes']).size())
    cycling_frequency = cycling_frequency.rename({0:'count'}, axis=1)
    cycling_frequency = cycling_frequency.reset_index()
    for clc in data['cyclist'].unique():
        c = cycling_frequency[cycling_frequency['cyclist'] == clc]
        infc = bike_to_human_infection(c, beta/2)
        if infc == 1:
            aux_cyc = [clc, start_time, start_time + dt.timedelta(minutes=period*24*60)]
            aux_cyc = pd.DataFrame(aux_cyc).T
            aux_cyc.columns=infected_human.columns
            infected_human = pd.concat([infected_human, aux_cyc], axis=0)

    q.put(infected_human)


def bike_human_interaction (data, contaminated_bikes, beta, state, start_time, SURVIVAL_TIME, q):
    data = data.copy()
    # Human-to-bike contamination
    contam_bikes = pd.DataFrame(columns=contaminated_bikes.columns)
    data.loc[:, ['health_status_human']] = np.nan
    for idx, trip in data.iterrows():
        data.loc[idx, 'health_status_human'] = state[trip['cyclist']]

    bike_usage = pd.DataFrame(data.groupby(['bike_id', 'health_status_human']).size())
    bike_usage = bike_usage.rename({0: 'count'}, axis=1)
    bike_usage = bike_usage.reset_index()
    for bike in data['bike_id'].unique():
        b = bike_usage[bike_usage['bike_id'] == bike]
        contaminated = human_to_bike_contamination(b, beta / 2)
        if contaminated == 1:
            aux_bike = [bike, start_time, start_time + dt.timedelta(minutes=SURVIVAL_TIME)]
            aux_bike = pd.DataFrame(aux_bike).T
            aux_bike.columns = contaminated_bikes.columns
            contam_bikes = pd.concat([contam_bikes, aux_bike], axis=0)

    q.put(contam_bikes)

def people_partitioning (NPROCESSOR, G, state):
    g = gt.util.find_vertex(G, state, 2)
    g = np.array([int(i) for i in g])
    people_cores = pd.DataFrame(index=g, columns=['degree', 'core'])
    people_cores['degree'] = G.get_out_degrees(g)
    people_cores = people_cores.sort_values(by='degree', ascending=False)
    quotient = people_cores.shape[0] // NPROCESSOR
    remainder = people_cores.shape[0] % NPROCESSOR
    core = np.array(quotient * [i for i in range(NPROCESSOR)])
    core = np.append(core, np.array(remainder * [(NPROCESSOR - 1)]))
    people_cores['core'] = core
    cores = pd.DataFrame(index=range(NPROCESSOR), columns=['people', 'contacts'])
    for c in cores.index:
        cores.loc[c, 'people'] = list(people_cores[people_cores['core'] == c].index)
        cores.loc[c, 'contacts'] = c

    return cores