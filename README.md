# TranEpiSim: Urban Transportation Disease Spread Simulator

Transportation systems can contribute to the spread of diseases by introducing pathogens to new geographic areas, by accelerating their diffusion, or by becoming a disease vector. In many respects, this mirrors the transmission of vector-borne diseases by living vectors such as mosquitoes. The latter role manifests itself within moving vehicles or in facilities where individuals share space with others, thus leading to new contacts that may facilitate disease transmission. TranEpiSim is an agent-based simulator built on top of a synthetic human contact network for Cook County, IL. The current version focuses on a single mode of transportation, micromobility systems, and introduces an agent-based approach that explicitly models a viral disease transmission through the use of micromobility vehicles in an urban area.  

![Transportation and epidemics](plot/transportation_and_epidemics.png)

## Table of Contents

- [Technologies Used](#technologies-used)
- [Features](#features)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Examples](#examples)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

## Technologies Used

- Python 3.10
- graph-tool & networkx
- geopandas & Shapely
- High Performance Computing (HPC)

## Features

- Transmission of viral diseases through fomites on miromobility vehicles and through close proximity in a large scale synthetic human contact network.
- Using actual individual micromobility trips
- Featured with a novel disease transmission model called SIR-SC.
- Built on top of graph-tool suited for fast, large-scale network analysis.

![AMB model](plot/platform.png)

## Getting Started

To get started with TranEpiSim, follow these steps:

1. Clone the repository: `git clone https://github.com/behnamnkp/Behavioral-Interventions.git`
2. Organize directories as represented below
3. Make sure you use the right versions of Python
4. Install the required dependencies in `requirements.txt`
5. Configure the flags and paths in the `.config.json` file. 
6. I have developed multiple notebooks that provide detailed, step-by-step instructions to make it easier for users to utilize the repository.
   1. ![Create synthetic human contact network](src/synthetic_human_contact_network.ipynb)
   2. Micromobility and disease spread in population

## Usage
Once you organize files and directories, you will be able to run the code to create the synthetic population and human contact network for Cook County, read micromobility trips, and calculate the dynamics of disease by setting up different scenarios.

## Examples
Here is an example of how you can create the synthetic population 

## Folder structure
```
📁 TranEpiSim
├── 📁 docs
├── 📁 data
│   ├── 📁 cbp
│   ├── 📁 dp
│   ├── 📁 education
│   ├── 📁 od
│   ├── 📁 road
├── 📁 src
│   ├── 🐍 main.py
│   ├── 🐍 synthesizer.py
│   ├── 📝 synthetic_human_contact_network.ipynb
├── 📁 output
├── 📁 plot             
└── 📄 config.json
└── 📄 README.md
└── 📄 Requirements.txt
└── ...
```

## Documentation

For detailed documentation on the API endpoints and available functions, please refer to the [API Documentation](docs/api-docs.md) file.

## Contributing

To contribute, please follow these steps:

1. Fork the repository
2. Create a new branch: `git checkout -b feature/new-feature`
3. Make your changes and commit them: `git commit -am 'Add new feature'`
4. Push the changes to your forked repository: `git push origin feature/new-feature`
5. Submit a pull request

Please adhere to the [Code of Conduct](CODE_OF_CONDUCT.md) and follow the coding conventions specified in the project.

## License

This project is not licensed yet.

## Acknowledgments

- 

## Contact

For any questions, suggestions, or feedback, please contact us:

- Email: bzn5190@psu.edu
- Twitter: @behnam_nkp