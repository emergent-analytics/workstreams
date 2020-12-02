# Emergent Analytics CookieCutter

## Problem we are trying to solve

The cookiecutter is a labelling and data selection tool to provide data driven knowledge management of COVID-19 infection data. When performing analytics on these data it is both key to deeply understand the data, where cookiecutter can be used as a browser, but also be able to create labelled datasets that can be used for further analyses.

Cookiecutter and its associated jupyter notebook infrastructure can be configured to tap into a variety of COVID-19 case data sources, and visualise it together with the exciting [Oxford Covid-19 Government Response Tracker (OxCGRT)](https://github.com/OxCGRT/covid-policy-tracker) dataset, so it becomes easier to understand and correlate the health situation and government responses aiming at managing the health situation. It also features clustering of government measures and a novel way of fitting [Gumbel dtstributions to the health datasets, provided by Damiaan Zwietering](https://pydata.org/eindhoven2020/schedule/presentation/15/building-a-coronaradar/).

In addition, as one of the goals of the Regional Risk Pulse Index was to study the impact of COVID-19 onto the economy, a tab provides the ability to model shocks to the economy, which can be used by the [Emergent Economic Engine E³](https://github.com/emergent-analytics/workstreams/tree/master/ws3) found in this repository.

This tool has been written very fast without too much consideration on code quality or maintainability, as a Minimum Viable Product to understand the features actually required to make such a tool useful.

## Installation

### Standalone

Check out the repository, create a virtual environment using tools such as [conda](https://docs.conda.io/en/latest/), or [pip, pipenv](https://packaging.python.org/guides/tool-recommendations/) or any similar tool. I found Python 3.6 to be a good base version for running cookiecutter. As we also need the [sklearn-contrib-py-earth](https://pypi.org/project/sklearn-contrib-py-earth/), which seems to be non trivial to get installed on Windows and some versions of python, my recommendation is to actually use the docker image.

You also need to have access to a SQL datastore such as postgres, or db2. sqlite3 may work, but no effort has been made to test it, not with any other SQL databases such as MySQL, SQL Server, or Oracle.

After installing the packages found in `requirements.txt`, you can launch the bokeh server application

```
SQL_CONNECT=postgresql://user:password@datbasehost:5432/database bokeh serve cookiecutter/  --allow-websocket-origin"*" -address 0.0.0.0
```

by adding the user name, password, database server name (such as localhost), and database name to connect to, when using postgres. I find [DigitalOcean's user guides](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-postgresql-on-ubuntu-18-04) as being of generally impressive and high quality. Also, if you never set up a posgres database before, getting [pg_hba.conf](https://www.linuxtopia.org/online_books/database_guides/Practical_PostgreSQL_database/c15679_002.htm) to accept connections can be frustrating.

When using db2, the connect string will be like

```
SQL_CONNECT=db2+ibm_db://user:password@datbasehost:50000/database bokeh serve cookiecutter/  --allow-websocket-origin"*" -address 0.0.0.0
```

I was using a [db2 docker image](https://hub.docker.com/r/ibmcom/db2) for set-up and testing and kept using the default db2inst1 user name, the set up a database under this user.

You can then connect to the bokeh server web frontend by entering

```
http://yourmachine:5006/cookiecutter
```

Before doing that, though, download data first.

### Docker

This is the recommended installation method as it contains all relevant packages in a working configuration. You will need [docker-compose](https://docs.docker.com/compose/install/) and a working docker installation. As a first step, you need a definition of environment variable for your build. In the folder containing this README.md and the docker-compose.yml files,

```
cp .env.example .env
```

Then, open up the `.env` file to change/edit its settings (altough the defaults should result in a working configuration)

```
PUBLIC_WEBSERVER_PORT=8080
GUMBEL_MAX_WAVES=25
POSTGRES_PORT=15432
```

The parameter settings are as follows

| Parameter               | Meaning                                                      | Default |
| ----------------------- | ------------------------------------------------------------ | ------- |
| `PUBLIC_WEBSERVER_PORT` | The IP port the webserver will make the cookiecutter app available. Deliberately set to 8080 to avoid clashes with a possibly existing webserver on port 80 | 8080    |
| `GUMBEL_MAX_WAVES`      | An internal setting for cookiecutter logic defining how many Gumbel distributions should be fitted to a time series. At the time of this writing, a value of about 25 waves seemed sufficient for the 2020 March-end of November time period, as time progresses, a higher number may become necessary. Setting this value to a higher number will cause the application to provision a lot more memory space for plotting the data, which could also cause memory issues on user browsers (this could not be tested).<br />Recommend leave at default unless, by 2021, you see deficiencies. | 25      |
| `POSTGRES_PORT`         | The externally visible port number to the postgres db inside the docker container suite. Postgres normally uses port 5432, we set it to 15432 to avoid clashes with possibly existing installations. This value needs to be used for the helper jupyter notebook database connection strings. | 15432   |

Then, download and build the images

```
docker-compose build
```

Check for any errors. 

The image comes with a dedicated nginx web server and its own copy of postgres. You can specify another SQL datastore (connection to postgres and db2 is available, else, you may need to install [other python packages](https://docs.sqlalchemy.org/en/13/core/engines.html)) by specifying its [connection string](https://docs.sqlalchemy.org/en/13/core/engines.html) in a file named `.cookiecutter.env`, located in the same folder as `docker-compose.yml`. Its content would read like, for postgres

```
SQL_CONNECT=postgresql://user:password@datbasehost:5432/database
```

or, for db2

```
SQL_CONNECT=db2+ibm_db://user:password@datbasehost:50000/database
```

or, commonly used for db2 installations on the cloud with enabled transport encryption

```
SQL_CONNECT=db2+ibm_db://user:password@datbasehost:50001/database;Security=ssl;
```

## Data

Before running the image or bokeh app, you need to download a number of datafiles. Please check the usage conditions and licensing of the source pages to ensure you are using and consuming these data in accordance to the data owners, considering your intended use.

You will need to change the connect strings according to your database configuration. When using the docker image, make sure it runs so the notebooks have access to the postgres database. Note that I configured the postgres sub-container to accept connections on port 15432, thus allowing cookiecutter to not interfere with a possibly existing postgres installation. The connect string will look like

```
postgres://cookiecutter:cookiecutter@localhost:15432/cookiec
```

The configuration above is done partially in `docker-compose.yml`, and `Dockerfile-postgresql`.

The folder `helper_notebooks` contains three of them, namely

* `Download Reference Data.ipynb`, which downloads country names, population data etc. This is a one-off import.

* `Download Case Data.ipynb` connects to data from Johns Hopkins, ECDC, and the German RKI, and the OxCGRT data, this is recommeneded to be run, say, daily
* `Download Economic Indicators.ipynb` connects and downloads data from a variety of sources found on the web, mainly as a proof of concept of various web crawling techniques. Please ensure you use the data in accordance to the allowed use, we assume this is personal use for academic and educational purposes only.

## Running

### Standalone

When deploying in standalone mode, change into this directory, then, in a console

```
SQL_CONNECT=postgresql://user:password@datbasehost:5432/database bokeh serve cookiecutter/  --allow-websocket-origin"*" -address 0.0.0.0
```
or

```
SQL_CONNECT=db2+ibm_db://user:password@datbasehost:50000/database bokeh serve cookiecutter/  --allow-websocket-origin"*" -address 0.0.0.0
```

Change username and password, machine name and database name to your configuration

then, from a web browser, enter

```
http://<machine_name>:5006/cookiecutter
```

## Docker

Change into this directory, then (after editing .env as required although it contains a good default, see below)
```
mv .env.docker-compose .env
docker-compose build
docker-compose up
```
If you want to run this in the background, use `docker-compose up -d`.

The app is now available on
```
http://<machine_name>:8080
```

To change the publically available port number from 8080, and to allow for persistence of the storage, there exists a file named .env which defines this entry.

```
PUBLIC_WEBSERVER_PORT=8080
```

The container image also ships with an installation of postgres to persist data. We map the postgres port to 15432 by default. You can specify another database backend via a config variable using SQLalchemy style connect strings, note the image only has drivers for postgres and db2 at the moment.

## Introduction videos

### Health and Stringency Tab

![Health and Scringency Tab](cookiecutter.small.gif)

### Economic and Economic Proxies Tab

![Economic and Economic Proxies Tab](cookiecutter.eco.small.gif)

### Fly Forward Modeller Tab

![Fly Forward Modeller Tab](cookiecutter.whatif.small.gif)

## UI

(need to provide description what it does)

![Screenshot](Screenshot1.JPG)

Basic screenshot.

UI Elements are
* a top status bar which initially displays how old the underlying data are, or if there was an issue locating it. When running the
  app for the first time, press "Load Data". When running locally you will see a list of countries as their associated data are preprocessed.
  TODO: Provide a feedback/reload the page as the docker variant does not provide user feedback except the first dataset
  being displayed (see bottom screenshot for initial screen)
* The dropdown boxes allow the selection of data sets and countries
* A radio button and Occurence number spinner to label the selections as "Wave" or a period of "Calm"ness, and number them. The author
  uses Wave1-Calm1-Wave2 etc, but this is merely a way to structure one's data
* A save button which becomes active when the user made a selection
* The top time series plot shows bars representing new cases reported, a smoothed trend line to counteract weekend phenomena, and a total
 active cases line, for the selected country
* the center time series shows the ["Stringency Index for Display" value](https://github.com/OxCGRT/covid-policy-tracker/blob/master/documentation/codebook.md)
* the grey heatmap details [which measures were taken](https://github.com/OxCGRT/covid-policy-tracker/blob/master/documentation/codebook.md)
* the wave data use the algorithms developed by D. Zwietering [here](https://pydata.org/eindhoven2020/schedule/presentation/15/building-a-coronaradar/)
* the band below the grey heatmap shows previous labelling data (aka Votes). Deletion of previous votes needs to be done via the file system.
  The height of the bands is randomly chosen and has no further meaning.
* a text entry field that allows for capturing a user identity, this is useful if a team of people are asked to label the data
* The table to the right gives a synposis of which countries have been voted on and which may need a vote. While this is being re-engineered,
 it will only be updated after pressing F5 or page refresh
* At the bottom, there are two histograms and one heatmap which illustrate the labelling/voting results for how long Waves or periods of Calm
  take, and if and how wave peaks (normalized by 100000 population of a country) scale with duration (hint: they are not)

![Screenshot select episode](Screenshot2.JPG)

Selection of an episode (Israel WAVE 2) and naming/classifying it (see the selections of the radio buttons).


![Screenshot episode saved](Screenshot3.JPG)

Saving a selected episode, in this case, Australia WAVE 2. After pressing "Save" the status bar will display the table the vote is saved to, together with a wave_id.

| Column Name | Description |
|---------------------------|---------------------------|
| c1_school closing | OxCGRT flag [Codebook](https://github.com/OxCGRT/covid-policy-tracker/blob/master/documentation/codebook.md) for selected time range |
| c2_workplace closing | OxCGRT flag [Codebook](https://github.com/OxCGRT/covid-policy-tracker/blob/master/documentation/codebook.md) for selected time range |
| c3_cancel public events | OxCGRT flag [Codebook](https://github.com/OxCGRT/covid-policy-tracker/blob/master/documentation/codebook.md) for selected time range |
| c4_restrictions on gatherings | OxCGRT flag [Codebook](https://github.com/OxCGRT/covid-policy-tracker/blob/master/documentation/codebook.md) for selected time range |
| c6_stay at home requirements | OxCGRT flag [Codebook](https://github.com/OxCGRT/covid-policy-tracker/blob/master/documentation/codebook.md) for selected time range |
| c7_restrictions on internal movement | OxCGRT flag [Codebook](https://github.com/OxCGRT/covid-policy-tracker/blob/master/documentation/codebook.md) for selected time range |
| c8_international travel controls | OxCGRT flag [Codebook](https://github.com/OxCGRT/covid-policy-tracker/blob/master/documentation/codebook.md) for selected time range |
| e1_income support | OxCGRT flag [Codebook](https://github.com/OxCGRT/covid-policy-tracker/blob/master/documentation/codebook.md) for selected time range |
| e2_debt/contract relief | OxCGRT flag [Codebook](https://github.com/OxCGRT/covid-policy-tracker/blob/master/documentation/codebook.md) for selected time range |
| e3_fiscal measures | OxCGRT flag [Codebook](https://github.com/OxCGRT/covid-policy-tracker/blob/master/documentation/codebook.md) for selected time range |
| e4_international support | OxCGRT flag [Codebook](https://github.com/OxCGRT/covid-policy-tracker/blob/master/documentation/codebook.md) for selected time range |
| h1_public information campaigns | OxCGRT flag [Codebook](https://github.com/OxCGRT/covid-policy-tracker/blob/master/documentation/codebook.md) for selected time range |
| h2_testing policy | OxCGRT flag [Codebook](https://github.com/OxCGRT/covid-policy-tracker/blob/master/documentation/codebook.md) for selected time range |
| h3_contact tracing | OxCGRT flag [Codebook](https://github.com/OxCGRT/covid-policy-tracker/blob/master/documentation/codebook.md) for selected time range |
| h4_emergency investment in healthcare | OxCGRT flag [Codebook](https://github.com/OxCGRT/covid-policy-tracker/blob/master/documentation/codebook.md) for selected time range |
| h5_investment in vaccines | OxCGRT flag [Codebook](https://github.com/OxCGRT/covid-policy-tracker/blob/master/documentation/codebook.md) for selected time range |
| datetime_date | datetime of selected data point(s) |
| new_cases | number of new cases reported at that datetime for the selected country from the selected dataset |
| new_cases_rel | number of new cases reported divided by population (in 100000) at that datetime for the selected country from the selected dataset |
| trend | trend for new cases reported, 7 day seasonality removed to cater for weekend lack of data |
| stringencyindex | OxCGRT StringencyIndexForDisplay [Codebook](https://github.com/OxCGRT/covid-policy-tracker/blob/master/documentation/codebook.md) for selected time range |
| rel_peak_new_cases | max of new_cases_rel |
| kind | one of "wave" or "calm", extension to "onset" under consideration to mark up initial waves |
| kind_counter | number as entered on the UI |
| from_dt | selection range from datetime |
| to_dt | selection range to datetime |
| duration | in days |
| user | user name as entered in the UI |
| identifier | ISO 3166-1 short three character country code, or a synthesis of two character country and two character postal/ISO code for state (e.g. US-NC, DE-BY) |
| vote_datetime | datetime the vote was made |
| vote_id | running number of vote |



## Disclaimer

Disclaimer: This information can be used for educational and research use. The author is not a health care professional and it is not recommended to use the views in this document for any healthcare related decision making.
