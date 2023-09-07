import sim
import utils

def main ():

    # Create synthetic population
    sim.create_population()
    # Create contact network
    sim.create_network()

    print("Reading data...")
    # Census tracts
    dp = utils.read_census()
    # Central business district
    cbd = utils.read_cbd()
    # Workplaces
    wps = utils.read_workplaces(dp)
    # Roads
    road = utils.read_roads()

    # Schools
    schools = utils.read_schools(dp)
    # Micromobility stations
    Divvy_Stations, Divvy_Stations_Thiessen = utils.read_mic_stations(dp, cbd)

    # Synthetic population
    people = utils.read_syn_pop()

    # Contact network
    G = utils.read_network(people, dp, wps, schools, Divvy_Stations_Thiessen)

    # Micromobility trips
    divvy_data = utils.read_mic_trips()

if __name__ == '__main__':
    main()