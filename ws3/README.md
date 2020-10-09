## Emergent Alliance - Economic Simulation Engine

This repository contains the code used in WS3. Note that most of datasets have to be downloaded, as specified in the corresponding notebook.

### Economic documentation
These set of notebooks provide some useful code to work with Input-Output tables. All the code for our blog [Input-Output Economics and the Impact of Covid-19](https://emergentalliance.org/?p=1689) is contained in these notebooks as well. 

### Simulation engine app
We have created a web browser app provides a visual interface to our Simulation Engine. To launch the app, you need the Python package `streamlit`. You can learn more about it in its [docs](https://docs.streamlit.io/en/stable/). Once you have installed the package, run in your terminal the following command: `streamlit run SimEngine.py`. The app should automatically launch in your default web browser. 

As of the last update of the deployed version of the app (17/09/2020), the part about countermeasuring the shock is purely illustrational/experimental - this section is work in progress and numbers do not have any meaning. You can use this version of the app in this [link](http://shock-dashboard.emergent.ml/). These issues have been fixed from version 7, which you can clone and run locally.

The current version of the simulation engine app uses the latest version of UK's Input-Output tables, corresponding to the year 2016. For more information on the data, please see:
- Office for National Statistics (2020), *UK input-output analytical tables - industry by industry*, URL: https://www.ons.gov.uk/economy/nationalaccounts/supplyandusetables/datasets/ukinputoutputanalyticaltablesindustrybyindustry, last accessed: 27 August 2020. Contains public sector information licensed under the Open Government License v3.0. http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/
