# Behavioral Interventions of Respiratory Diseases

A pipeline to study behavioral interventions of respiratory diseases such as COVID-19 and Seasonal Influenza. We use 
individual responses to a large scale longitudinal survey and cellphone visits to places of interests to measure and 
study human behavior variation, and it's impact on respiratory diseases. This project is part of a project at 
[Bharti lab](https://www.humidlab.com/) (PI: [Dr. Nita Bharti](https://www.huck.psu.edu/people/nita-bharti)).

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

- Python 3.9
- R version 4.2.3 (2023-03-15 ucrt)

## Features

- Creates time series of human behavior in response to behavioral interventions during COVID-19
- Behaviors include, visits to places of interest, facemasking, maintaining six feet distance, gatherings of different sizes, personal hygiene, going to bar and restaurants, working and attending classes remotely, etc.
- Behaviors are collected separately for two cohorts (student and non-students) at Centre County, PA.
- Pipeline reads, preprocesses, and mines the raw data sets and applies statistical time series analysis and modeling.

## Getting Started

To get started with the Behavioral-Interventions repository, follow these steps:

1. Clone the repository: `git clone https://github.com/behnamnkp/Behavioral-Interventions.git`
2. Organize directories as below
3. Make sure you use the right versions of Python and R
4. Install the required dependencies in `requirements.txt`
5. Configure the flags and paths in the `.config.json` file. Here are some of the flags you need to make sure set to 1 when running the code for the first time. You can set them to 0 to save run time after the first run.
   - flags = {"READ_SAFEGRAPH_PATTERNS": 0, "PROCESS_SAFEGRAPH_PATTERNS": 1, "READ_SOCIAL_DISTANCING": 0, "RESAMPLE_LAYERING": 0, "FILL_SG_MISSING": 0, "START_WEEK": "2015-07-03", "END_WEEK": "2022-12-25"}
## Usage
Once you organize files and directories, you will be able to run the code. Code creates visualizations of uptake, persistence, and waning of behavioral interventions ove time, and conducts time series regression analysis.

## Examples

Here is an example of how you can set up the flags and paths, input, and process data:

## Folder structure
```
📁 BI
├── 📁 documents
├── 📁 temp
├── 📁 codes
│   ├── 📁 old
│   ├── 🐍 main.py
│   ├── 📊 main.R
│   ├── 🐍 bi.py
│   ├── 🐍 read_data.py
│   ├── 🐍 plot.py
│   ├── 🐍 resample_data.py
│   ├── 📊 functions.R
│   ├── 📊 epidemic_weeks.R
├── 📁 src
│   ├── 📁 d4a
│   ├── 📁 census
│   ├── 📁 safegraph
│   ├── 📁 psu_calendar
│   ├── 📁 environment
│   ├── 📁 vaccination
│   ├── 📁 disease_incidence
│   ├── 📁 traffic_cameras
│   ├── 📁 safegraph_home_panel_summaries
│   ├── 📁 safegraph_social_distancing
├── 📁 output             
│   ├── 📁 exploratory_analysis   
│   ├── 📁 timeseries_regression      
│   └── 📁 reports  
└── 📄 config.json
└── 📄 README.md
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

- This project was funded by [Data4Action project](https://covid19.ssri.psu.edu/data4action#:~:text=The%20Data%204%20Action%20Project,Clinical%20and%20Translational%20Science%20Institute.) and National Science Foundation (NSF)
- COVID-19 and Seasonal Influenza cases and vaccination were provided by PSU Health Services and Pennsylvania Department of Health.

## Contact

For any questions, suggestions, or feedback, please contact us:

- Email: bzn5190@psu.edu
- Twitter: @behnam_nkp