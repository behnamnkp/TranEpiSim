import pandas as pd
import pickle
import geopandas as gpd
from graph_tool.all import graph_tool as gt
from graph_tool.all import *
import synthesizer as syn
import os
import networkx as nx
import numpy as np
import tran
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

def create_population(config):

    ROOT = config['file_paths']['ROOT']
    INPUT = ROOT + config['file_paths']['INPUT']
    OUTPUT = ROOT + config['file_paths']['OUTPUT']
    DOCS = ROOT + config['file_paths']['DOCS']

    # 1. Road file
    road = gpd.read_file(INPUT + 'road/roads.shp')

    # 2. Demographic profile
    dp = gpd.read_file(INPUT + 'dp/dp.shp').set_index('GEOID10')
    dp['portion'] = dp.apply(lambda tract: tract.geometry.area / tract.Shape_Area, axis=1)

    # 3. Schools and daycares
    school = gpd.read_file(INPUT + 'education/school.shp')
    daycare = gpd.read_file(INPUT + 'education/day_care.shp')

    # 4. Number of establishments per county per size
    cbp = pd.read_csv(INPUT + 'cbp/cbp10co.zip')
    cbp = cbp[(cbp.naics.str.startswith('-'))] #All types of establishments included
    cbp['fips'] = cbp.fipstate.map("{:02}".format) + cbp.fipscty.map("{:03}".format)
    cbp = cbp.set_index('fips')

    # 5. Origin (home) - destination (job) at census-tract level
    od = pd.read_csv(INPUT + 'od/tract-od15Cook.csv', dtype={i:(str if i<2 else int) for i in range(6)})

    #Add workplace counts and sizes to dp
    dp['WP_CNT'] = syn.number_of_wp(dp,od,cbp)
    dp['WP_PROBA'] = dp.WP_CNT.map(syn.wp_proba)

    # Create a unified file for education
    school = syn.clean_schools(school,daycare)

    population = []
    errors = []
    wps = []

    dp.apply(lambda t: syn.synthesize(t,od,road,school,errors, population, wps, dp),axis=1)

    # Save the results
    with open(OUTPUT + 'errors.pkl', 'wb') as f:
        pickle.dump(errors, f)
    with open(OUTPUT + 'population.pkl', 'wb') as f:
        pickle.dump(population, f)
    with open(OUTPUT + 'wps.pkl', 'wb') as f:
        pickle.dump(wps, f)

def create_network(config):

    ROOT = config['file_paths']['ROOT']
    INPUT = ROOT + config['file_paths']['INPUT']
    OUTPUT = ROOT + config['file_paths']['OUTPUT']
    DOCS = ROOT + config['file_paths']['DOCS']

    # Read synthesized population
    with open(OUTPUT + 'population.pkl','rb') as f:
        people = pd.concat(pickle.load(f))

    # Create and save the networks
    g = syn.create_networks(people,k=31,p=.3)
    nx.write_gml(g, OUTPUT + 'contact_network.gml')

    # Create networks by contact types
    for etype in ['hhold','work','school']:
        sg = nx.Graph([(u,v) for u,v,d in g.edges(data=True) if d['etype']==etype])
        nx.write_gml(sg, OUTPUT + f'{etype}_contact_network.gml')

    # Create a network for work-school contacts
    work_school = nx.Graph([(u,v) for u,v,d in g.edges(data=True) if d['etype'] in ['work','school']])
    nx.write_gml(work_school,'work_school_contact_network.gml')

def read_census(INPUT):
    print('Census tracts....')
    dp = gpd.read_file(INPUT + 'dp/cook_tracts_prj.shp').set_index('GEOID10')
    return dp

def read_cbd(INPUT):
    cbd = gpd.read_file(INPUT + 'cbd/geo_export_f6a32edf-05fb-4a16-8de3-84a051947cb9.shp')
    cbd.to_crs("EPSG:3857", inplace=True)
    return cbd

def read_workplaces(INPUT, dp):
    print('Work places...')
    with open(INPUT + 'cbp/wps2.pkl', 'rb') as f:
        wps = pd.concat(pickle.load(f))
    wps = gpd.GeoDataFrame(wps)
    wps.rename({0: 'geometry'}, inplace=True, axis=1)
    wps.set_geometry("geometry", inplace=True)
    wps.set_crs(epsg=3857, inplace=True)
    wps2 = gpd.sjoin(wps, dp.loc[:, ['GEOID10', 'geometry']], how="left", op='intersects')
    wps2.drop('index_right', inplace=True, axis=1)
    wps2.rename({'GEOID10': 'GEOID10_wps', 'geometry': 'geometry_wps'}, inplace=True, axis=1)
    return wps

def read_roads(INPUT):
    print('Roads...')
    road = gpd.read_file(INPUT + 'road/roads_prj.shp')
    return road

def read_schools(INPUT, dp):
    print('school....')
    schools = gpd.read_file(INPUT + 'education/education.shp')
    schools.set_crs(epsg=3857, inplace=True)
    schools2 = gpd.sjoin(schools, dp.loc[:, ['GEOID10', 'geometry']], how="left", op='within')
    schools2.rename({'GEOID10': 'GEOID10_sch', 'geometry': 'geometry_sch'}, inplace=True, axis=1)

    return schools
def read_mic_stations(INPUT, dp, cbd):
    print('bike stations....')
    Divvy_Stations = gpd.read_file(INPUT + 'mic_stations/DivvyStations.shp')
    Divvy_Stations_Thiessen = gpd.read_file(INPUT + 'mic_stations/station_thiessen_network_final.shp')
    print('Micromobility...')
    Divvy_Stations.to_crs("EPSG:3857", inplace=True)
    Divvy_Stations_Thiessen.set_crs(epsg=3857, inplace=True)
    Divvy_Stations = gpd.sjoin(Divvy_Stations, dp.loc[:, ['GEOID10', 'geometry']], how="left", op='within')
    Divvy_Stations.drop(['index_right'], inplace=True, axis=1)
    Divvy_Stations['dist_to_cbd'] = 0
    for idx, row in Divvy_Stations.iterrows():
        Divvy_Stations.loc[idx, ['dist_to_cbd']] = cbd.distance(row.geometry)[0]

    return [Divvy_Stations, Divvy_Stations_Thiessen]

def read_syn_pop(OUTPUT):
    print("Synthetic population...")
    with open(OUTPUT + 'population.pkl', 'rb') as f:
        people = pd.concat(pickle.load(f))

    people.reset_index(inplace=True)
    mask = people['sex'] == 'm'
    people.loc[mask, ['sex']] = 1
    mask = people['sex'] == 'f'
    people.loc[mask, ['sex']] = 0

    people_joined = pd.read_csv(OUTPUT + 'people_joined.txt')
    people = people.join(people_joined, lsuffix='_caller', rsuffix='_other')
    people = people.drop(['index_other', 'TARGET_FID', 'Join_Count', 'OBJECTID'], axis=1)
    people = people.rename({'index_caller': 'index', 'GEOID10': 'GEOID10_hhold'}, axis=1)

    return people

def read_network(OUTPUT, people, dp, wps2, schools2, Divvy_Stations_Thiessen):
    print('Contact Network...')
    G = gt.load_graph(OUTPUT + "contact_network.gml")
    G.set_directed(False)

    dp.reset_index(inplace=True)
    print('People...')
    people = gpd.GeoDataFrame(people)
    people.set_crs(epsg=3857, inplace=True)
    people = preprocess_population(people, dp, wps2, schools2, Divvy_Stations_Thiessen)
    dp.set_index('GEOID10', inplace=True)

    print('Contact network...')
    label = G.vp['label']

    code = G.new_vertex_property('int16_t')
    code.get_array()[:] = people.code
    G.vertex_properties["code"] = code

    age = G.new_vertex_property('int16_t')
    age.get_array()[:] = people.age
    G.vertex_properties["age"] = age

    # male=1;female=0
    sex = G.new_vertex_property('int16_t')
    sex.get_array()[:] = people.sex
    G.vertex_properties["sex"] = sex

    htype = G.new_vertex_property('int16_t')
    htype.get_array()[:] = people.htype
    G.vertex_properties["htype"] = htype

    station_hhold = G.new_vertex_property('int16_t')
    station_hhold.get_array()[:] = people.station_hhold
    G.vertex_properties["station_hhold"] = station_hhold

    station_outside = G.new_vertex_property('int16_t')
    station_outside.get_array()[:] = people.station_outside
    G.vertex_properties["station_outside"] = station_outside

    infectious_period = G.new_vertex_property('int16_t')
    infectious_period.get_array()[:] = 0
    G.vertex_properties["infectious_period"] = infectious_period

    conntype = G.new_edge_property('int16_t')
    conntype.get_array()[:] = 0
    G.edge_properties["conntype"] = conntype

    # S = 1, I = 2, R = 3
    etype = G.ep["etype"]
    G.edge_properties["etype"] = etype

    conntype = G.new_edge_property('int16_t')
    conntype.get_array()[:] = 0
    G.edge_properties["conntype"] = conntype

    school = G.new_edge_property('bool')
    for i in range(G.ep["etype"].get_array()[:].shape[0]):
        if G.ep["etype"].get_array()[i] == 3:
            school.get_array()[i] = 1
        else:
            school.get_array()[i] = 0
    G.edge_properties["school"] = school

    work = G.new_edge_property('bool')
    for i in range(G.ep["etype"].get_array()[:].shape[0]):
        if G.ep["etype"].get_array()[i] == 2:
            work.get_array()[i] = 1
        else:
            work.get_array()[i] = 0
    G.edge_properties["work"] = work

    home = G.new_edge_property('bool')
    for i in range(G.ep["etype"].get_array()[:].shape[0]):
        if G.ep["etype"].get_array()[i] == 1:
            home.get_array()[i] = 1
        else:
            home.get_array()[i] = 0
    G.edge_properties["home"] = home

    return G

def read_mic_trips(INPUT, Divvy_Stations, IMPUTE_NEW_TRIPS, NEW_BIKES_PERCENTAGE, CALCULATE_STATION_DISTANCES, NPROCESSOR):

    divvy_colmns = ['trip_id', 'start_time', 'end_time', 'bike_id', 'duration',
                    'start_station_id', 'start_station_name', 'end_station_id',
                    'end_station_name', 'user_type', 'member_gender', 'member_birth_day',
                    'age', 'start_day_of_week', 'end_day_of_week', 'distance', 'speed']

    divvy_dtype = {'trip_id': int, 'bike_id': int, 'duration': 'float64',
                   'start_station_id': int, 'start_station_name': str, 'end_station_id': int,
                   'end_station_name': str, 'user_type': str, 'member_gender': str, 'member_birth_day': int, 'age': int,
                   'start_day_of_week': str, 'end_day_of_week': str, 'distance': 'float64', 'speed': 'float64'}

    if IMPUTE_NEW_TRIPS:
        divvy_data = pd.DataFrame(columns=divvy_colmns)
        os.chdir(INPUT + "mic_trips/")
        files = os.listdir()
        files = list(filter(lambda f: f.endswith('.csv'), files))

        if CALCULATE_STATION_DISTANCES:

            # # Distance between stations
            Divvy_Stations.set_index('ID', inplace=True)
            station_dists = pd.DataFrame(index=Divvy_Stations.index, columns=['distances'])
            station_dists = list(station_dists.to_dict().values())[0]
            for idx, item in Divvy_Stations.iterrows():
                aux = {}
                for idx2, item2 in Divvy_Stations.iterrows():
                    aux = {k: aux.get(k, 0) + {idx2: item['geometry'].distance(item2['geometry'])}.get(k, 0) for k in
                           set(aux) | set({idx2: item['geometry'].distance(item2['geometry'])})}
                station_dists[idx] = aux
            with open(INPUT + 'micro_stations/station_dists.pkl', 'wb') as f:
                pickle.dump(station_dists, f)

            divvy_data['distance']=np.nan
            for idx in list(divvy_data.index):
                divvy_data.loc[idx, 'distance'] = station_dists[divvy_data.loc[idx, 'start_station_id']][divvy_data.loc[idx, 'end_station_id']]
        else:
            divvy_data_distances = pd.read_csv(INPUT + 'mic_trips/divvy_data_distances.csv')
            divvy_data_distances.set_index('trip_id', inplace=True)
            divvy_data = divvy_data.merge(divvy_data_distances, left_on=divvy_data.index,
                                          right_on=divvy_data_distances.index, how='left')

        for file in files:
            f = pd.read_csv(file, usecols=divvy_colmns, header=None, skiprows=[0], dtype=divvy_dtype)
            divvy_data = divvy_data.append(f)

        all_samples = tran.impute_bikes_and_trips(NEW_BIKES_PERCENTAGE, divvy_data, NPROCESSOR)
        divvy_data = pd.concat([divvy_data, all_samples], axis=0)

        divvy_data['cyclist'] = np.nan

    else:
        divvy_data = pd.read_csv(INPUT + 'sources/divvy_data_' + str(NEW_BIKES_PERCENTAGE) + '.csv',
                                 usecols=divvy_colmns, dtype=divvy_dtype, parse_dates=['start_time', 'end_time'])

    bikes = divvy_data['bike_id'].unique()
    bikes = pd.DataFrame(bikes, columns=['bike_id'])

    return [divvy_data, bikes]

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

