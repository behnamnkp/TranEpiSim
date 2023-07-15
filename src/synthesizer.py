# General:
import sys
import numpy as np
import pandas as pd
import timeit
from itertools import chain
from sklearn.preprocessing import normalize

# Spatial:
import geopandas as gpd
from shapely.geometry import MultiLineString, LineString, Point, Polygon, GeometryCollection

# Visualization:
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)


# Graph analyses:
from graph_tool.all import *
import networkx as nx

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

def number_of_wp(dp, od, cbp):
    """
    calculate number of workplaces for each tract
    wp_tract = wp_cty * (tract_employed / county_employed)
    """
    # get the number of employees per tract
    dwp = od[['work','S000']].groupby('work').sum()
    dwp = pd.merge(dp.portion.to_frame(),dwp,left_index=True,right_index=True,how='left').fillna(0)
#     dwp = dwp.portion*dwp.S000/10
    wp_class = ["n1_4","n5_9","n10_19","n20_49","n50_99","n100_249","n250_499","n500_999","n1000","n1000_1","n1000_2","n1000_3","n1000_4"]
    dwp['county'] = dwp.index.str[:5]
    a = dwp.groupby('county').sum()
    a = a.join(cbp[wp_class].sum(axis=1).to_frame('wpcty'))
    # note: as Dr. Crooks suggested agents not living in our region included
    dwp = (dwp.portion * dwp.S000 / dwp.county.map(a.S000)) * dwp.county.map(a.wpcty)
    return dwp.apply(np.ceil).astype(int)

def wp_proba(x):
    """
    probability of an employee working in that workplace is lognormal:
    http://www.haas.berkeley.edu/faculty/pdf/wallace_dynamics.pdf
    """
    if x == 0: return np.zeros(0)
    b = np.random.lognormal(mean=2,size=x).reshape(-1, 1)
    return np.sort(normalize(b,norm='l1',axis=0).ravel())

# Main Methods
def clean_schools(school,daycare):
    school = school[school.START_GRAD != 'N'] #Unknown
    """
    Codes for Grade Level
    PK = PreKindergarten
    KG = Kindergarten
    TK = Transitional Kindergarten
    T1 = Transitional First Grade
    01-12 = Grade 1-12
    """
    grade2age = {'PK':3,'KG':5,'UG':5, 'TK':5,'T1':6}
    grade2age.update({'{:02d}'.format(i+1):i+6 for i in range(12)})

    print(school)

    school = school.assign(start_age = school.START_GRAD.map(grade2age))
    school = school.assign(end_age = school.END_GRADE.map(grade2age) +1)
    school.loc[school.END_GRADE=='PK','end_age'] = 6 #not 4
    mask = school.ENROLLMENT<=0 #non-positive enrollments?
    school.loc[mask,'ENROLLMENT'] = school[~mask].ENROLLMENT.median()
    school = school[['start_age','end_age','ENROLLMENT','geometry']]

    daycare = daycare.assign(start_age = 0, end_age = 5)
    daycare = daycare.rename(columns={'POPULATION':'ENROLLMENT'})
    daycare = daycare[['start_age','end_age','ENROLLMENT','geometry']]

    school = school.append(daycare, ignore_index=True)
    school['current'] = 0
    return school

# Create Individuals
def create_individuals(tract):
    """Generate a population of ages and sexes as a DataFrame

    Given the number of individuals for 18 age groups of each sex,
    return a two column DataFrame of age ([0,89]) and sex ('m','f')
    """
    portion = tract.geometry.area / tract.Shape_Area # what portion of the tract is included
    age_sex_groups = (tract[25:62].drop('DP0010039') * portion).astype(int)
    dfs=[]
    for code,count in enumerate(age_sex_groups):
        base_age = (code % 18)*5
        gender = 'm' if code < 18 else 'f'
        ages = []
        for offset in range(4): ages.extend([offset+base_age]*(count//5))
        ages.extend([base_age+4]*(count-len(ages)))
        dfs.append(pd.DataFrame({'code':code, 'age':ages,'sex':[gender]*count}))
    df = pd.concat(dfs).sample(frac=1,random_state=123).reset_index(drop=True)
    df.index = tract.name + 'i' + df.index.to_series().astype(str)
    df['friends'] = [set()] * len(df)
    return df

# Create Households
def create_households(tract, people):
    hh_cnt = get_hh_cnts(tract)
    hholds = pd.DataFrame()
    hholds['htype'] = np.repeat(hh_cnt.index, hh_cnt)
    hholds = hholds[hholds.htype != 6].sort_values('htype', ascending=False).append(hholds[hholds.htype == 6])
    populate_households(tract, people, hholds)

def get_hh_cnts(tract):
    """
    Eleven household types:
    0         h&w (no<18)
    1      h&w&ch (ch<18)
    2        male (no<18)
    3        male (ch<18)
    4      female (no<18)
    5      female (ch<18)
    6     nonfamily group
    7       lone male <65
    8       lone male >65
    9      lone female<65
    10     lone female>65
    """

    portion = tract.geometry.area / tract.Shape_Area  # what portion of the tract is included
    #householdConstraints = (tract[150:166] * portion).astype(int)  # HOUSEHOLDS BY TYPE
    householdConstraints = (tract[153:169] * portion).astype(int)  # HOUSEHOLDS BY TYPE
    hh_cnt = pd.Series(np.zeros(11), dtype=int)  # 11 household types (group quarters are not household)

    # husband/wife families
    hh_cnt[0] = householdConstraints[4] - householdConstraints[5];  # husband-wife family
    hh_cnt[1] = householdConstraints[5];  # husband-wife family, OWN CHILDREN < 18
    # male householders
    hh_cnt[2] = householdConstraints[6] - householdConstraints[7];  # single male householder
    hh_cnt[3] = householdConstraints[7];  # single male householder, OWN CHILDREN < 18
    # female householders
    hh_cnt[4] = householdConstraints[8] - householdConstraints[9];  # single female householder
    hh_cnt[5] = householdConstraints[9];  # single female householder, OWN CHILDREN < 18
    # nonfamily householders
    hh_cnt[6] = householdConstraints[10] - householdConstraints[11];  # nonfamily group living
    hh_cnt[7] = householdConstraints[12] - householdConstraints[13];  # lone male < 65
    hh_cnt[8] = householdConstraints[13];  # lone male >= 65
    hh_cnt[9] = householdConstraints[14] - householdConstraints[15];  # lone female < 65
    hh_cnt[10] = householdConstraints[15];  # lone female >= 65
    return hh_cnt

def populate_households(tract, people, hholds):
    """
    Eleven household types:
    0         h&w (no<18)
    1      h&w&ch (ch<18)
    2        male (no<18)
    3        male (ch<18)
    4      female (no<18)
    5      female (ch<18)
    6     nonfamily group
    7       lone male <65
    8       lone male >65
    9      lone female<65
    10     lone female>65
    """
    mask = pd.Series(True, index=people.index)  # [True]*len(people)
    hholds['members'] = hholds.htype.apply(gen_households, args=(people, mask,))
    """The seven types of group quarters are categorized as institutional group quarters
    (correctional facilities for adults, juvenile facilities, nursing facilities/skilled-nursing facilities,
    and other institutional facilities) or noninstitutional group quarters (college/university student housing,
    military quarters, and other noninstitutional facilities)."""
    portion = tract.geometry.area / tract.Shape_Area  # what portion of the tract is included
    group_population = int(tract.DP0120014 * portion)  # people living in group quarters (not in households)
    # gq_indices = people[(people.age>=65) | (people.age<18)].index[:group_population]
    gq_indices = people[mask].index[:group_population]
    # for i in gq_indices: mask[i] = False
    mask.loc[gq_indices] = False

    # now distribute the remaining household guys as relatives...
    relatives = people[mask].index
    it = iter(relatives)  # sample by replacement
    relative_hhs = hholds[hholds.htype < 7].sample(n=len(relatives), replace=True)
    relative_hhs.members.apply(lambda x: x.append(next(it)))  # appends on mutable lists
    # for i in relatives: mask[i] = False
    mask.loc[relatives] = False
    # print('is anyone left homeless:',any(mask))
    # add those living in group quarters as all living in a house of 12th type
    if group_population > 0:
        hholds.loc[len(hholds)] = {'htype': 11, 'members': gq_indices}

    hholds = hholds.set_index(tract.name + 'h' + pd.Series(np.arange(len(hholds)).astype(str)))

    def hh_2_people(hh, people):
        for m in hh.members:
            people.loc[m, 'hhold'] = hh.name
            people.loc[m, 'htype'] = hh.htype

    hholds.apply(hh_2_people, args=(people,), axis=1)
    people['htype'] = people.htype.astype(int)

def gen_households(hh_type, people, mask):
    """Helper
    """
    members = []
    head_ranges = [range(4, 18), range(4, 14), range(4, 18), range(4, 14), range(22, 36), range(22, 30),
                   chain(range(1, 18), range(19, 36)), range(4, 13), range(13, 18), range(21, 31), range(31, 36)]

    # head_age =  [range(15,99), range(20,70), range(15,99), range(20,70), range(15,99),
    # range(20,70), range(15,99), range(15,65), range(65,99), range(15,65), range(65,99)]
    # head_sex= [('m'),('m'),('m'),('m'),('f'),('f'),('m','f'),('m'),('m'),('f'),('f')]

    if hh_type != 6:
        # add the householder
        pot = people[mask].code
        iindex = pot[pot.isin(head_ranges[hh_type])].index[0]
        h1 = people.loc[iindex]  # age & sex of h1
        mask[iindex] = False
        members.append(iindex)

        # if living alone then return the household
        if hh_type > 6:
            return members

        # if husband and wife, add the wife
        if hh_type in (0, 1):

            pot = people[mask].code
            # behnam changed this from 17-19 to 16-20. That is why nobody existed to become the wife
            iindex = pot[pot.isin(range(h1.code + 16, h1.code + 20))].index[0]
            h2 = people.loc[iindex]  # -4 < husband.age - wife.age < 9
            mask[iindex] = False
            members.append(iindex)

        # if may have children 18+ then add them
        #         if (hh_type <= 5) and (h1.age > 36) and (np.random.random() < 0.3):
        #         #to have a child older than 18, h1 should be at least 37
        #             pot = people[mask]
        #             iindex = pot[pot.age.isin(range(18,h1.age-15))].index[0]
        #             ch18 = people.ix[iindex] #child should be at least 19 years younger than h1
        #             mask[iindex] = False
        #             members.append(iindex)

        """A child includes a son or daughter by birth (biological child), a stepchild,
        or an adopted child of the householder, regardless of the child’s age or marital status.
        The category excludes sons-in-law, daughters- in-law, and foster children."""
        # household types with at least one child (18-)
        if hh_type in (1, 3, 5):

            # https://www.census.gov/hhes/families/files/graphics/FM-3.pdf
            if hh_type == 1:
                num_of_child = max(1, abs(int(np.random.normal(2))))  # gaussian touch
            elif hh_type == 3:
                num_of_child = max(1, abs(int(np.random.normal(1.6))))  # gaussian touch
            elif hh_type == 5:
                num_of_child = max(1, abs(int(np.random.normal(1.8))))  # gaussian touch

            pot = people[mask]
            if hh_type == 1 :
                iindices = pot[(pot.age < 18) & (45 > h2.age - pot.age)].index[:num_of_child]
            else:  # father (mother) and child age difference not to exceed 50 (40)
                age_diff = 45 if hh_type == 5 else 55
                iindices = pot[(pot.age < 18) & (age_diff > h1.age - pot.age)].index[:num_of_child]
            for i in iindices:
                child = people.loc[i]
                mask[i] = False
                members.append(i)

    else:
        # add the householder
        pot = people[mask].code

        try:
            iindex = pot[pot.isin(head_ranges[hh_type])].index[0]
            h1 = people.loc[iindex]  # age & sex of h1
            mask[iindex] = False
            members.append(iindex)

            # if nonfamily group (type 6) then either friends or unmarried couples
            num_of_friends = max(1, abs(int(np.random.normal(1.3))))  # gaussian touch
            iindices = pot[pot.isin(range(h1.code - 2, h1.code + 3))].index[:num_of_friends]
            for i in iindices:
                friend = people.loc[i]
                mask[i] = False
                members.append(i)
        except:
            pass

    return members

# Assign Workplaces
def assign_workplaces(tract, people, od):
    """
    if the destination tract of a worker is not in our DP dataset
    then we assign his wp to 'DTIDw', otherwise 'DTIDw#'

    the actual size distribution of establishments is lognormal
    https://www.princeton.edu/~erossi/fsdae.pdf
    """
    # get true od numbers considering the proportions
    portion = tract.geometry.area / tract.Shape_Area
    td = od[od['home'] == tract.name].set_index('work').S000
    td = (td * portion).apply(np.ceil).astype(int)  # from this tract to others
    # 58.5%: US population (16+) - employment rate
    # https://data.bls.gov/timeseries/LNS12300000
    while td.sum() > people[people.age >= 18].shape[0]:
        drop_value = np.random.choice(td.index, 1, replace=False)
        td = td.drop(drop_value, axis=0)
    employed = people[people.age >= 18].sample(td.sum()).index  # get the employed
    dtract = pd.Series(np.repeat(td.index.values, td.values))  # get the destination tract
    #if 'wp' in people.columns: people.drop('wp',axis=1,inplace=True)
    people.loc[employed, 'wp'] = dtract.apply(lambda x: x + 'w' + str(np.random.choice(dp.loc[x, 'WP_CNT'], p=dp.loc[x, 'WP_PROBA'])) if x in dp.index else x + 'w').values

# Space Creation
#shapely geometries are not hashable, here is my hash function to check the duplicates
def hash_geom(x):
    if x.geom_type == 'MultiLineString':
        return tuple((round(lat,6),round(lon,6)) for s in x for lat,lon in s.coords[:])
    else:
        return tuple((round(lat,6),round(lon,6)) for lat,lon in x.coords[:])

# create spaces
#HD=0.0005, WD=0.0002, avg_wp = 10
def create_spaces(tract, hcnt, od, road, HD=50, WD=20, avg_wp = 10):
    portion = tract.geometry.area / tract.Shape_Area # what portion of the tract is included
    # create houses
    # DP0180001: Total housing units, DP0180002: Occupied housing units
    hcnt = int(tract.DP0180002 * portion) #number of households DP0130001 == DP0180002
    if tract.DP0120014 > 0:
        hcnt += 1 #people living in group quarters (not in households)
    mask = road.intersects(tract.geometry)
    hmask = mask & road.MTFCC.str.contains('S1400|S1740')
    hrd = road[hmask].intersection(tract.geometry)
    hrd = hrd[hrd.geom_type.isin(['LinearRing', 'LineString', 'MultiLineString'])]
    hrd = hrd[~hrd.apply(hash_geom).duplicated()]
    houses = hrd.apply(lambda x: pd.Series([x.interpolate(seg) for seg in np.arange(HD,x.length,HD)]))
    houses = houses.unstack().dropna().reset_index(drop=True) #flatten
    houses = houses.sample(n=hcnt,replace=True).reset_index(drop=True)
    houses.index = tract.name + 'h' + houses.index.to_series().astype(str)
    #create workplaces
    jcnt = int(portion * od[od.work==tract.name].S000.sum() / avg_wp)
    wmask = mask & road.MTFCC.str.contains('S1400|S1200')
    wrd = road[wmask].intersection(tract.geometry)
    wrd = wrd[wrd.geom_type.isin(['LinearRing', 'LineString', 'MultiLineString'])]
    wrd = wrd[~wrd.apply(hash_geom).duplicated()]
    #workplaces on S1400|S1200
    swps = wrd.apply(lambda x: pd.Series([x.interpolate(seg) for seg in np.arange(x.length,WD)]))
    #workplaces on the joints of S1400|S1740
    rwps = hrd.apply(lambda x: Point(x.coords[0]) if type(x) != MultiLineString else Point(x[0].coords[0]))
    wps = rwps.append(swps.unstack().dropna().reset_index(drop=True))
    wps = wps.sample(n=tract.WP_CNT,replace=True).reset_index(drop=True)
    wps.index = tract.name + 'w' + wps.index.to_series().astype(str)
    return gpd.GeoSeries(houses), gpd.GeoSeries(wps)

# # Populate Households
# def populate_households(tract, people, hholds):
#     """
#     Eleven household types:
#     0         h&w (no<18)
#     1      h&w&ch (ch<18)
#     2        male (no<18)
#     3        male (ch<18)
#     4      female (no<18)
#     5      female (ch<18)
#     6     nonfamily group
#     7       lone male <65
#     8       lone male >65
#     9      lone female<65
#     10     lone female>65
#     """
#
#     def gen_households(hh_type, mask):
#         """Helper
#         """
#         members = []
#         head_ranges = [range(4, 18), range(4, 14), range(4, 18), range(4, 14), range(22, 36), range(22, 30),
#                        chain(range(1, 18), range(19, 36)), range(4, 13), range(13, 18), range(21, 31), range(31, 36)]
#
#         # head_age =  [range(15,99), range(20,70), range(15,99), range(20,70), range(15,99),
#         # range(20,70), range(15,99), range(15,65), range(65,99), range(15,65), range(65,99)]
#         # head_sex= [('m'),('m'),('m'),('m'),('f'),('f'),('m','f'),('m'),('m'),('f'),('f')]
#
#         # add the householder
#         pot = people[mask].code
#         iindex = pot[pot.isin(head_ranges[hh_type])].index[0]
#         h1 = people.loc[iindex]  # age & sex of h1
#         mask[iindex] = False
#         members.append(iindex)
#
#         # if living alone then return the household
#         if hh_type > 6:
#             return members
#
#         # if husband and wife, add the wife
#         if hh_type in (0, 1):
#             pot = people[mask].code
#             iindex = pot[pot.isin(range(h1.code + 17, h1.code + 19))].index[0]
#             h2 = people.loc[iindex]  # -4 < husband.age - wife.age < 9
#             mask[iindex] = False
#             members.append(iindex)
#
#         # if may have children 18+ then add them
#         #         if (hh_type <= 5) and (h1.age > 36) and (np.random.random() < 0.3):
#         #         #to have a child older than 18, h1 should be at least 37
#         #             pot = people[mask]
#         #             iindex = pot[pot.age.isin(range(18,h1.age-15))].index[0]
#         #             ch18 = people.ix[iindex] #child should be at least 19 years younger than h1
#         #             mask[iindex] = False
#         #             members.append(iindex)
#
#         """A child includes a son or daughter by birth (biological child), a stepchild,
#         or an adopted child of the householder, regardless of the child’s age or marital status.
#         The category excludes sons-in-law, daughters- in-law, and foster children."""
#         # household types with at least one child (18-)
#         if hh_type in (1, 3, 5):
#             # https://www.census.gov/hhes/families/files/graphics/FM-3.pdf
#             if hh_type == 1:
#                 num_of_child = max(1, abs(int(np.random.normal(2))))  # gaussian touch
#             elif hh_type == 3:
#                 num_of_child = max(1, abs(int(np.random.normal(1.6))))  # gaussian touch
#             elif hh_type == 5:
#                 num_of_child = max(1, abs(int(np.random.normal(1.8))))  # gaussian touch
#
#             pot = people[mask]
#             if hh_type == 1:
#                 iindices = pot[(pot.age < 18) & (45 > h2.age - pot.age)].index[:num_of_child]
#             else:  # father (mother) and child age difference not to exceed 50 (40)
#                 age_diff = 45 if hh_type == 5 else 55
#                 iindices = pot[(pot.age < 18) & (age_diff > h1.age - pot.age)].index[:num_of_child]
#             for i in iindices:
#                 child = people.loc[i]
#                 mask[i] = False
#                 members.append(i)
#
#         # if nonfamily group then either friends or unmarried couples
#         if hh_type == 6:
#             pot = people[mask].code
#             num_of_friends = max(1, abs(int(np.random.normal(1.3))))  # gaussian touch
#             iindices = pot[pot.isin(range(h1.code - 2, h1.code + 3))].index[:num_of_friends]
#             for i in iindices:
#                 friend = people.loc[i]
#                 mask[i] = False
#                 members.append(i)
#
#         return members
#
#     mask = pd.Series(True, index=people.index)  # [True]*len(people)
#     hholds['members'] = hholds.htype.apply(gen_households, args=(mask,))
#     """The seven types of group quarters are categorized as institutional group quarters
#     (correctional facilities for adults, juvenile facilities, nursing facilities/skilled-nursing facilities,
#     and other institutional facilities) or noninstitutional group quarters (college/university student housing,
#     military quarters, and other noninstitutional facilities)."""
#     portion = tract.geometry.area / tract.Shape_Area  # what portion of the tract is included
#     group_population = int(tract.DP0120014 * portion)  # people living in group quarters (not in households)
#     # gq_indices = people[(people.age>=65) | (people.age<18)].index[:group_population]
#     gq_indices = people[mask].index[:group_population]
#     # for i in gq_indices: mask[i] = False
#     mask.loc[gq_indices] = False
#
#     # now distribute the remaining household guys as relatives...
#     relatives = people[mask].index
#     it = iter(relatives)  # sample by replacement
#     relative_hhs = hholds[hholds.htype < 7].sample(n=len(relatives), replace=True)
#     relative_hhs.members.apply(lambda x: x.append(next(it)))  # appends on mutable lists
#     # for i in relatives: mask[i] = False
#     mask.loc[relatives] = False
#     # print('is anyone left homeless:',any(mask))
#     # add those living in group quarters as all living in a house of 12th type
#     if group_population > 0:
#         hholds.loc[len(hholds)] = {'htype': 11, 'members': gq_indices}
#
#     hholds.set_index(tract.name + 'h' + hholds.index.values)
#
#     def hh_2_people(hh, people):
#         for m in hh.members:
#             people.loc[m, 'hhold'] = hh.name
#             people.loc[m, 'htype'] = hh.htype
#
#     hholds.apply(hh_2_people, args=(people,), axis=1)

# Get Errors
def get_errors(tract,people):
    """Percentage errors
    """
    err = {}
    portion = tract.geometry.area / tract.Shape_Area # what portion of the tract is included
    senior_actual = int(tract.DP0150001 * portion) # Households with individuals 65 years and over
    minor_actual = int(tract.DP0140001 * portion) # Households with individuals under 18 years
#     err['tract'] = tract.name
    err['population'] = tract.DP0010001
    err['in_gq'] = tract.DP0120014
    avg_synthetic_family = people[people.htype<6].groupby('hhold').size().mean()
    err['avg_family'] = 100*(avg_synthetic_family - tract.DP0170001) / tract.DP0170001
    err['avg_hh'] = 100*(people[people.htype!=11].groupby('hhold').size().mean() - tract.DP0160001) / tract.DP0160001
    err['senior_hh'] = 100*(people[people.age>=65].hhold.nunique() - senior_actual) / senior_actual
    err['minor_hh'] = 100*(people[people.age<18].hhold.nunique() - minor_actual) / minor_actual
    return pd.Series(err,name=tract.name)


def assign_hholds_to_houses(hholds, houses, people):
    hholds['house'] = houses.sample(frac=1, random_state=123).index

    def hh_2_people(hh, people):
        for m in hh.members:
            people.loc[m, 'house'] = hh.house
            people.loc[m, 'htype'] = hh.htype

    people['house'] = 'homeless'
    people['htype'] = 'homeless'
    people['work'] = 'unemployed'
    hholds.apply(hh_2_people, args=(people,), axis=1)
    people['geometry'] = people.house.map(houses)
    return gpd.GeoDataFrame(people)

#buffer=0.3
# Assign Students to Schools
def assign_students_to_schools(tract, people, school, buffer=10000):
    """
    Get the schools within 30km that accepts students at this age.
    loop over schools from nearest to farest:
      if school has any space then enroll
    if no school available then
      enroll to the school with the lowest enrollment rate
    update the enrollment of the school
    PERCENTAGE ERRORS
    """

    def assign_school(x, pot, school):
        sch = pot.distance(x.geometry).sort_values()
        for s in sch.index:  # from nearest to farest
            if school.loc[s, 'current'] < school.loc[s, 'ENROLLMENT']:
                school.loc[s, 'current'] += 1
                return s

        # if no space left at any school within the buffer
        least_crowded = (pot.current / pot.ENROLLMENT).idxmin()
        school.loc[least_crowded, 'current'] += 1
        return least_crowded

    kids = people.age < 18
    buff = tract.geometry.buffer(buffer)

    sch_pot = school[school.intersects(buff)]  # filter potential schools and daycares
    people.loc[kids, 'wp'] = people[kids].apply(assign_school, args=(sch_pot, school), axis=1)

# Synthesize
def synthesize(tract, od, road, school, errors, population, wps):
    start_time = timeit.default_timer()
    print(tract.name,'started...',end=' ')
    people = create_individuals(tract)
    # I added this if to eliminate tracts with zero population
    if people.empty != True:
        create_households(tract,people)
        assign_workplaces(tract,people,od)
        #create spaces
        houses, wp = create_spaces(tract, people.hhold.nunique(), od, road)
        #assign households to houses
        people['geometry'] = people.hhold.map(houses)
        assign_students_to_schools(tract,people,school)
#       people['friends'] = create_networks(people)
        err = get_errors(tract,people)
        wps.append(wp)
        population.append(people)
        errors.append(err)
#       giant = max(nx.connected_component_subgraphs(g), key=len)
#       nms.append(net_metrics(giant))
        print(tract.name,'now ended ({:.1f} secs)'.format(timeit.default_timer() - start_time))
    else:
        print(tract.name,'has zero population...')

# Social Networks
def create_networks(people, k, p=.3):
    g = nx.Graph()
    g.add_nodes_from(people.index)

    grouped = people.groupby('hhold')
    # hhold = 1
    grouped.apply(lambda x: create_edges(x, g, etype=1, k=k, p=p))

    grouped = people[people.age >= 18].groupby('wp')
    # work = 2
    grouped.apply(lambda x: create_edges(x, g, etype=2, k=k, p=p))

    grouped = people[people.age < 18].groupby('wp')
    # school = 3
    grouped.apply(lambda x: create_edges(x, g, etype=3, k=k, p=p))

    return g  # max(nx.connected_component_subgraphs(g), key=len)

def create_edges(x, g, etype, k=4, p=.3):
    """Creates the edges within group `g` and adds them to `edges`

    if the group size is <=5, then a complete network is generated. Otherwise,
    a Newman–Watts–Strogatz small-world graph is generated with k=4 and p=0.3

    http://www.sciencedirect.com/science/article/pii/S0375960199007574
    """

    # High school contacts> 5 min
    if ((etype==1) or (etype==3)) and (len(x) < 31):
        sw = nx.complete_graph(len(x))

    elif ((etype==1) or (etype==3)) and (len(x) >= 31):
        sw = nx.newman_watts_strogatz_graph(len(x), 31, p)

    # Transmission of Influenza A in a Student Office Basedon Realistic Person-to-Person Contact and SurfaceTouch Behaviour
    elif (etype==2) and (len(x) < 9):
        sw = nx.complete_graph(len(x))

    elif (etype==2) and (len(x) >= 9):
        sw = nx.newman_watts_strogatz_graph(len(x), 9, p)

    sw = nx.relabel_nodes(sw, dict(zip(sw.nodes(), x.index.values)))

    if etype:
        g.add_edges_from(sw.edges(), etype=etype)
    else:
        g.add_edges_from(sw.edges())

# def create_edges(x, g, etype, k=4, p=.3):
#     """Creates the edges within group `g` and adds them to `edges`

#     if the group size is <=5, then a complete network is generated. Otherwise,
#     a Newman–Watts–Strogatz small-world graph is generated with k=4 and p=0.3

#     http://www.sciencedirect.com/science/article/pii/S0375960199007574
#     """
#     if len(x) <= 5:
#         sw = nx.complete_graph(len(x))
#     else:
#         sw = nx.newman_watts_strogatz_graph(len(x), k, p)
#     sw = nx.relabel_nodes(sw, dict(zip(sw.nodes(), x.index.values)))
#     if etype:
#         g.add_edges_from(sw.edges(), etype=etype)
#     else:
#         g.add_edges_from(sw.edges())

# def create_edges(x, g, etype, k=4, p=.3):
#     """Creates the edges within group `g` and adds them to `edges`

#     if the group size is <=5, then a complete network is generated. Otherwise,
#     a Newman–Watts–Strogatz small-world graph is generated with k=4 and p=0.3

#     http://www.sciencedirect.com/science/article/pii/S0375960199007574
#     """

#     if (etype==1) & len(x)<=5:
#         sw = nx.complete_graph(len(x))
#     elif (etype==1) & len(x)>5:
#         print(np.floor((k/100)*len(x)))
#         sw = nx.newman_watts_strogatz_graph(len(x), np.floor((k/100)*len(x)), p)
#     elif etype!=1 & len(x)<=5:
#         sw = nx.complete_graph(len(x))
#     elif etype!=1 & len(x)>5:
#         print(np.floor((k/100)*len(x)))
#         sw = nx.newman_watts_strogatz_graph(len(x), np.floor((k/100)*len(x)), p)
#     sw = nx.relabel_nodes(sw, dict(zip(sw.nodes(), x.index.values)))
#     if etype:
#         g.add_edges_from(sw.edges(), etype=etype)
#     else:
#         g.add_edges_from(sw.edges())