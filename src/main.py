import sim
import utils
import json

def main ():

    # Load the JSON configuration file
    with open('config.json') as config_file:
        config = json.load(config_file)

    # Create synthetic population
    utils.create_population(config)
    # Create contact network
    utils.create_network(config)
    # Example simulation
    sim.simulate(config)

if __name__ == '__main__':
    main()