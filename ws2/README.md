# Results from Workstream 2

## News analysis

## Meltwater python client

The meltwater folder is a Python package that provides a client to access the Export APIs from Meltwater: https://developer.meltwater.com. Note that we do not provide an interface to all the endpoints exposed by API. We only provide an interface to two groups of endpoints: the "searches" endpoints and the "One-time export" endpoints. However, the client can easily be extended to include the other groups of endpoints. 

## NOTAM analysis

The folder - airport_restrictions contain the analysis done on NOTAM data to extract quarantine and country restrictions. Please note that this folder contains only code and the NOTAM data will have to be downloaded manually. The following provides a brief overview of the different notebooks available in the folder:

* ws2_snr_NOTAMs_1_data_preparation.ipynb - Basic preprocessing of NOTAM - removing special characters, expanding abbreviations, removing stop words
* ws2_snr_NOTAMs_2_topic_modeling.ipynb -  Identification of differnt topics present in the NOTAM
* WS2_snr_notams_3_quarantine_text.ipynb -  Extraction of quarantine duration from NOTAM using Named Entity Recognition (NER) and regex
* ws2_snr_NOTAMs_1_data_preparation_mulitple files.ipynb - Similar to the first notebook on data preparation. Iteration of data preprocessing to multiple files
* ws2_snr_NOTAMs_country_level_restrictions_timeline.ipynb - Information extraction of restriction on foreigners using NER, Part of speech tagging and dependency parser.

## Airbnb analysis

The Airbnb folder contains the analysis done on the InsideAirbnb data: http://insideairbnb.com/get-the-data.html. A predictive model using FBProphet characterizes the expected Airbnb demand if Covid-19 pandemic did not happen. The user can see the effect created by the pandemic. Additionally, a geo distribution of the Airbnb demand in cities around the world provides insights into the new trends in hospitality. The following blog post can provide more information. 
