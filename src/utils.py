import pandas as pd
import pickle
import geopandas as gpd
from graph_tool.all import graph_tool as gt
from graph_tool.all import *
import synthesizer as syn
import os
import networkx as nx

def create_population():

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

    # Save the results
    with open('output/errors.pkl', 'wb') as f:
        pickle.dump(errors, f)
    with open('output/population.pkl', 'wb') as f:
        pickle.dump(population, f)
    with open('output/wps.pkl', 'wb') as f:
        pickle.dump(wps, f)

def create_network():

    # Read synthesized population
    with open('output/population.pkl','rb') as f:
        people = pd.concat(pickle.load(f))

    # Create and save the networks
    g = syn.create_networks(people,k=31,p=.3)
    nx.write_gml(g,'output/contact_network.gml')

    # Create networks by contact types
    for etype in ['hhold','work','school']:
        sg = nx.Graph([(u,v) for u,v,d in g.edges(data=True) if d['etype']==etype])
        nx.write_gml(sg, f'output/{etype}_contact_network.gml')

    # Create a network for work-school contacts
    work_school = nx.Graph([(u,v) for u,v,d in g.edges(data=True) if d['etype'] in ['work','school']])
    nx.write_gml(work_school,'work_school_contact_network.gml')

def read_census():
    print('Census tracts....')
    dp = gpd.read_file('gdrive/MyDrive/ColabNotebooks/Thesis/sources/cook_tracts_prj.shp').set_index('GEOID10')
    return dp

def read_cbd():
    cbd = gpd.read_file(
        'gdrive/MyDrive/ColabNotebooks/Thesis/sources/geo_export_f6a32edf-05fb-4a16-8de3-84a051947cb9.shp')
    cbd.to_crs("EPSG:3857", inplace=True)
    return cbd

def read_workplaces(dp):
    print('Work places...')
    with open('gdrive/MyDrive/ColabNotebooks/Thesis/sources/wps2.pkl', 'rb') as f:
        wps = pd.concat(pickle.load(f))
    wps = gpd.GeoDataFrame(wps)
    wps.rename({0: 'geometry'}, inplace=True, axis=1)
    wps.set_geometry("geometry", inplace=True)
    wps.set_crs(epsg=3857, inplace=True)
    wps2 = gpd.sjoin(wps, dp.loc[:, ['GEOID10', 'geometry']], how="left", op='intersects')
    wps2.drop('index_right', inplace=True, axis=1)
    wps2.rename({'GEOID10': 'GEOID10_wps', 'geometry': 'geometry_wps'}, inplace=True, axis=1)
    return wps

def read_roads():
    print('Roads...')
    road = gpd.read_file('gdrive/MyDrive/ColabNotebooks/Thesis/sources/roads_prj.shp')
    return road

def read_schools(dp):
    print('school....')
    schools = gpd.read_file('gdrive/MyDrive/ColabNotebooks/Thesis/sources/education.shp')
    schools.set_crs(epsg=3857, inplace=True)
    schools2 = gpd.sjoin(schools, dp.loc[:, ['GEOID10', 'geometry']], how="left", op='within')
    schools2.rename({'GEOID10': 'GEOID10_sch', 'geometry': 'geometry_sch'}, inplace=True, axis=1)

    return schools
def read_mic_stations(dp, cbd):
    print('bike stations....')
    Divvy_Stations = gpd.read_file('gdrive/MyDrive/ColabNotebooks/Thesis/sources/DivvyStations.shp')
    Divvy_Stations_Thiessen = gpd.read_file(
        'gdrive/MyDrive/ColabNotebooks/Thesis/sources/station_thiessen_network_final.shp')
    print('Micromobility...')
    Divvy_Stations.to_crs("EPSG:3857", inplace=True)
    Divvy_Stations_Thiessen.set_crs(epsg=3857, inplace=True)
    Divvy_Stations = gpd.sjoin(Divvy_Stations, dp.loc[:, ['GEOID10', 'geometry']], how="left", op='within')
    Divvy_Stations.drop(['index_right'], inplace=True, axis=1)
    Divvy_Stations['dist_to_cbd'] = 0
    for idx, row in Divvy_Stations.iterrows():
        Divvy_Stations.loc[idx, ['dist_to_cbd']] = cbd.distance(row.geometry)[0]

    return [Divvy_Stations, Divvy_Stations_Thiessen]

def read_syn_pop():
    print("Synthetic population...")
    with open('gdrive/MyDrive/ColabNotebooks/Thesis/sources/population.pkl', 'rb') as f:
        people = pd.concat(pickle.load(f))

    people.reset_index(inplace=True)
    mask = people['sex'] == 'm'
    people.loc[mask, ['sex']] = 1
    mask = people['sex'] == 'f'
    people.loc[mask, ['sex']] = 0

    people_joined = pd.read_csv('/content/gdrive/MyDrive/ColabNotebooks/Thesis/sources/people_joined.txt')
    people = people.join(people_joined, lsuffix='_caller', rsuffix='_other')
    people = people.drop(['index_other', 'TARGET_FID', 'Join_Count', 'OBJECTID'], axis=1)
    people = people.rename({'index_caller': 'index', 'GEOID10': 'GEOID10_hhold'}, axis=1)

    return people

def read_network(people, dp, wps2, schools2, Divvy_Stations_Thiessen):
    print('Contact Network...')
    G = gt.load_graph("gdrive/MyDrive/ColabNotebooks/Thesis/sources/N2023_31.gml")
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

def read_mic_trips():
    NEW_BIKES_PERCENTAGE = 0
    divvy_colmns = ['trip_id', 'start_time', 'end_time', 'bike_id', 'duration',
                    'start_station_id', 'start_station_name', 'end_station_id',
                    'end_station_name', 'user_type', 'member_gender', 'member_birth_day',
                    'age', 'start_day_of_week', 'end_day_of_week', 'distance', 'speed']

    divvy_dtype = {'trip_id': int, 'bike_id': int, 'duration': 'float64',
                   'start_station_id': int, 'start_station_name': str, 'end_station_id': int,
                   'end_station_name': str, 'user_type': str, 'member_gender': str, 'member_birth_day': int, 'age': int,
                   'start_day_of_week': str, 'end_day_of_week': str, 'distance': 'float64', 'speed': 'float64'}

    divvy_data = pd.read_csv('/content/gdrive/MyDrive/ColabNotebooks/Thesis/sources/divvy_data_' + str(NEW_BIKES_PERCENTAGE) + '.csv',
                             usecols=divvy_colmns, dtype=divvy_dtype, parse_dates=['start_time', 'end_time'])

    return divvy_data

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

