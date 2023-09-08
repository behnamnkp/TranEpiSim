from dateutil.parser import parse
import datetime as dt
import tran
import epi
import pandas as pd
import timeit
from graph_tool.all import graph_tool as gt
from graph_tool.all import *
import ast
from multiprocessing import *
import numpy as np
import utils
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

def simulate(config):

    ROOT = config['file_paths']['ROOT']
    INPUT = ROOT + config['file_paths']['INPUT']
    OUTPUT = ROOT + config['file_paths']['OUTPUT']
    DOCS = ROOT + config['file_paths']['DOCS']

    # Options + initial conditions. Update config for desired  settings
    NPROCESSOR = config['flags']['NPROCESSOR']
    START_TIME = config['flags']['START_TIME']
    END_TIME = config['flags']['END_TIME']
    DELTA_TIME = config['flags']['DELTA_TIME']
    WORKING_HOURS_START = config['flags']['WORKING_HOURS_START']
    WORKING_HOURS_END = config['flags']['WORKING_HOURS_END']
    SURVIVAL_TIME = config['flags']['SURVIVAL_TIME']
    BIKES_CLEANED_A_DAY = config['flags']['BIKES_CLEANED_A_DAY']
    PERIOD = config['flags']['PERIOD']
    R0 = config['flags']['R0']
    RHO = config['flags']['RHO']
    INITIAL_CONTAMINATED_BIKES = config['flags']['INITIAL_CONTAMINATED_BIKES']
    INFECTED_TRACTS_PERCENTAGE = config['flags']['INFECTED_TRACTS_PERCENTAGE']
    NEW_BIKES_PERCENTAGE = config['flags']['NEW_BIKES_PERCENTAGE']
    IMPUTE_NEW_TRIPS = config['flags']['IMPUTE_NEW_TRIPS']
    CALCULATE_STATION_DISTANCES = config['flags']['CALCULATE_STATION_DISTANCES']

    print("Reading data...")
    # Census tracts
    dp = utils.read_census(INPUT)
    # Central business district
    cbd = utils.read_cbd(INPUT)
    # Workplaces
    wps = utils.read_workplaces(INPUT, dp)
    # Roads
    road = utils.read_roads(INPUT)

    # Schools
    schools = utils.read_schools(INPUT, dp)
    # Micromobility stations
    Divvy_Stations, Divvy_Stations_Thiessen = utils.read_mic_stations(INPUT, dp, cbd)

    # Synthetic population
    people = utils.read_syn_pop(OUTPUT)

    # Contact network
    G = utils.read_network(OUTPUT, people, dp, wps, schools, Divvy_Stations_Thiessen)

    # Micromobility trips
    divvy_data, bikes = utils.read_mic_trips(INPUT, Divvy_Stations,
                                             IMPUTE_NEW_TRIPS, NEW_BIKES_PERCENTAGE,
                                             CALCULATE_STATION_DISTANCES, NPROCESSOR)

    deltatime = dt.timedelta(minutes=DELTA_TIME) # Minutes
    beta = R0 * (1 / PERIOD)

    # Initial contaminated bikes
    contaminated_bikes = tran.choose_bikes(bikes, parse('2018-07-05 08:00:00'), divvy_data,
                                           INITIAL_CONTAMINATED_BIKES, SURVIVAL_TIME)
    print('Initial contaminated bikes: ', list(contaminated_bikes['bike_id']))

    # Create a property to track health status of individuals
    state = G.new_vertex_property('int16_t')
    state.get_array()[:] = 1
    G.vertex_properties["state"] = state

    # Create a property to track infectious PERIOD of individuals
    infectious_PERIOD = G.new_vertex_property('int16_t')
    infectious_PERIOD.get_array()[:] = 0
    G.vertex_properties["infectious_PERIOD"] = infectious_PERIOD

    # Track cyclists
    cyclist = pd.DataFrame(columns=['trip_id', 'cyclist'])
    dailycases_rep = pd.DataFrame(columns=['people_id', 'time'])
    recoveries_rep = pd.DataFrame(columns=['people_id', 'time'])
    micro_infections_rep = pd.DataFrame(
        columns=['cyclist', 'start_of_infection', 'end_of_infection', 'replication', 'R0'])
    contaminated_bikes_rep = pd.DataFrame(
        columns=['bike_id', 'start_of_contamination', 'end_of_contamination', 'replication', 'R0'])

    strt_m = timeit.default_timer()

    while (START_TIME <= END_TIME) and (contaminated_bikes.shape[0] != 0 or len(gt.util.find_vertex(G, state, 2)) != 0):
        print('** Calculating micromobility:' + str(START_TIME))

        # Trips that start during this time step
        mask = (divvy_data['START_TIME'] < START_TIME) & (divvy_data['START_TIME'] >= START_TIME - deltatime)
        trips = divvy_data.loc[mask, ['trip_id', 'START_TIME', 'END_TIME', 'duration', 'bike_id',
                                      'start_station_id', 'end_station_id',
                                      'user_type', 'member_gender', 'age']]

        trips_n = trips.shape[0]
        print('   Number of  of trips: ' + str(trips.shape[0]))

        if trips_n != 0:
            if ((START_TIME.hour >= 8) and (START_TIME.hour <= 16)) and (
                    START_TIME.strftime('%A') != 'Saturday' or START_TIME.strftime(
                    '%A') != 'Sunday' or START_TIME.strftime('%Y-%m-%d') == parse('2018-06-01').strftime('%Y-%m-%d')):

                # Partition trips on cores
                cores = tran.trip_partitioning(NPROCESSOR, trips)
                # Assign trips to cyclists
                processes = []
                workers = []
                qq = []

                for idx, row in cores.iterrows():
                    aux = trips[trips['start_station_id'].isin(ast.literal_eval(row['stations']))]

                    q = Queue()
                    p = Process(target=tran.assign_cyclist, args=(
                    aux, people[people['station_outside'].isin(ast.literal_eval(row['stations']))], 'station_outside', q))
                    workers.append(p)
                    p.start()
                    processes.append(p)
                    qq.append(q)

                for process in processes:
                    process.join()

                for item in qq:
                    cyclist = pd.concat([cyclist, item.get()], axis=0, ignore_index=True)

                for process in processes:
                    process.terminate()

            else:

                # Partition trips on cores
                cores = tran.trip_partitioning(NPROCESSOR, trips)
                # Assign trips to cyclists
                processes = []
                workers = []
                qq = []
                for idx, row in cores.iterrows():
                    aux = trips[trips['start_station_id'].isin(ast.literal_eval(row['stations']))]

                    q = Queue()
                    p = Process(target=tran.assign_cyclist, args=(
                    aux, people[people['station_hhold'].isin(ast.literal_eval(row['stations']))], 'station_hhold', q,))
                    workers.append(p)
                    p.start()
                    processes.append(p)
                    qq.append(q)

                for process in processes:
                    process.join()

                for item in qq:
                    cyclist = pd.concat([cyclist, item.get()], axis=0, ignore_index=True)

                for process in processes:
                    process.terminate()

        else:
            # print('no trips in this step!')
            pass

        if START_TIME.hour == 23 and START_TIME.minute == 00:

            print('   Micromobility trips are added in:', timeit.default_timer() - strt_m)

            if (START_TIME.strftime('%A') == 'Saturday' or START_TIME.strftime('%A') == 'Sunday' or START_TIME.strftime(
                    '%Y-%m-%d') == parse('2018-07-04').strftime('%Y-%m-%d')):
                G.set_edge_filter(G.home)

            print('** Calculating spread: R0' + str(R0) + str(START_TIME))
            strt2 = timeit.default_timer()
            print('   Micromobility spread...')

            infected_human = pd.DataFrame(columns=['cyclist', 'start_of_infection', 'end_of_infection'])
            aux_bike = pd.DataFrame(columns=contaminated_bikes.columns)

            divvy_data2 = divvy_data.merge(cyclist, left_on=divvy_data['trip_id'], right_on=cyclist['trip_id'],
                                           how='left')
            divvy_data2 = divvy_data2.drop(['key_0', 'trip_id_x'], axis=1)
            divvy_data2 = divvy_data2.rename({'trip_id_y': 'trip_id'}, axis=1)
            divvy_data2 = divvy_data2[~divvy_data2['cyclist'].isna()]

            # Partition trips on cores
            cores = tran.trip_partitioning(NPROCESSOR, divvy_data2)
            # Assign trips to cyclists

            processes = []
            workers = []
            qq = []

            for idx, row in cores.iterrows():
                aux = divvy_data2[divvy_data2['start_station_id'].isin(ast.literal_eval(row['stations']))]

                q = Queue()
                p = Process(target=epi.human_bike_interaction, args=(aux, contaminated_bikes, beta / RHO, START_TIME, PERIOD, q,))
                workers.append(p)
                p.start()
                processes.append(p)
                qq.append(q)

            for process in processes:
                process.join()

            for item in qq:
                while item.empty() is False:
                    s = item.get()
                    if s.shape[0] != 0:
                        infected_human = pd.concat([infected_human, s], axis=0, ignore_index=True)

            for process in processes:
                process.terminate()

            print('   New cases through micromobility: ', infected_human.shape[0])
            micro_infections = infected_human
            micro_infections['R0'] = str(R0)
            micro_infections_rep = pd.concat([micro_infections_rep, micro_infections], axis=0, ignore_index=True)

            processes = []
            workers = []
            qq = []

            for idx, row in cores.iterrows():
                aux = divvy_data2[divvy_data2['start_station_id'].isin(ast.literal_eval(row['stations']))]

                q = Queue()
                p = Process(target=epi.bike_human_interaction, args=(aux, contaminated_bikes, beta / RHO, state, START_TIME, SURVIVAL_TIME, q,))
                workers.append(p)
                p.start()
                processes.append(p)
                qq.append(q)

            for process in processes:
                process.join()

            for item in qq:
                while item.empty() is False:
                    s = item.get()
                    if s.shape[0] != 0:
                        aux_bike = pd.concat([aux_bike, s], axis=0, ignore_index=True)

            for process in processes:
                process.terminate()

            contaminated_bikes = pd.concat([contaminated_bikes, aux_bike], axis=0, ignore_index=True)
            print('   Currently contaminated bikes: ', contaminated_bikes.shape[0])
            aux_bike['R0'] = str(R0)

            contaminated_bikes_rep = pd.concat([contaminated_bikes_rep, aux_bike], axis=0, ignore_index=True)

            # Update status of the individuals in the contact network
            # New micromobility infections
            for idx, case in infected_human.iterrows():
                if state[case['cyclist']] != 3:
                    state[case['cyclist']] = 2

            print('   Micromobility spread is added in:', timeit.default_timer() - strt2)

            # Human-to-human infection (general population)
            strt3 = timeit.default_timer()

            processes = []
            qq = []

            # Create partitions:
            print('   Creating network partitions...')
            cores = epi.people_partitioning(NPROCESSOR, G, state)

            print('   Calculating the spread...')
            # cores = pd.DataFrame(columns=['people_id'])

            new = []
            for prcss in list(cores.index):
                q = Queue()
                p = Process(target=epi.spread, args=(cores.loc[prcss, 'people'], G, state, beta, q))
                processes.append(p)
                qq.append(q)

            for pr in processes:
                pr.start()

            for item in qq:
                s = list(item.get()[0])
                new = new + s

            for pr in processes:
                pr.join()

            for process in processes:
                process.terminate()

            new = list(np.unique(new))

            new = pd.DataFrame(new, columns=['people_id'])
            new['time'] = START_TIME

            G.set_edge_filter(None)

            # Update status of the individuals in the contact network
            # New infections
            print('   New infections in general population: ', new.shape[0])
            if new.shape[0] != 0:
                for idx, case in new.iterrows():
                    if state[case['people_id']] != 3:
                        state[case['people_id']] = 2

                new['R0'] = str(R0)
                dailycases_rep = pd.concat([dailycases_rep, new], axis=0, ignore_index=True)

            print('   Currently infectious population: ', len(gt.util.find_vertex(G, state, 2)))
            # Recoveries
            recovery = []
            for vv in gt.util.find_vertex(G, state, 2):
                if infectious_PERIOD[vv] >= PERIOD:
                    infectious_PERIOD[vv] = 0
                    state[vv] = 3
                    recovery.append(int(vv))
                else:
                    infectious_PERIOD[vv] += 1

            print('   New recoveries: ', len(recovery))
            recovery = pd.DataFrame(recovery, columns=['people_id'])
            recovery['time'] = START_TIME
            recovery['R0'] = str(R0)
            recoveries_rep = pd.concat((recoveries_rep, recovery), axis=0, ignore_index=True)

            for idx, row in contaminated_bikes.iterrows():
                if row['end_of_contamination'] < START_TIME:
                    contaminated_bikes = contaminated_bikes.drop(idx, axis=0)

            cyclist = pd.DataFrame(columns=['trip_id', 'cyclist'])

            print('** Spread finished in: ', timeit.default_timer() - strt3)
            strt_m = timeit.default_timer()

        START_TIME += deltatime

    dailycases_rep.to_csv(OUTPUT + '/dailycases.csv')
    recoveries_rep.to_csv(OUTPUT + '/recoveries.csv')
    micro_infections_rep.to_csv(OUTPUT + '/micro_infections.csv')
    contaminated_bikes_rep.to_csv(OUTPUT + '/contaminated_bikes.csv')

