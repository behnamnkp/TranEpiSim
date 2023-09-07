# General:
import sys
import numpy as np
import pandas as pd
import datetime as dt
import ast

# Spatial:
import geopandas as gpd

# Visualization:
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

# Graph analyses:
from graph_tool.all import graph_tool as gt
from graph_tool.all import *

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

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