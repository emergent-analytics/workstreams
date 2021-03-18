# Standard library imports
import os
import pandas as pd
import numpy as np
from datetime import timedelta, datetime

import re

# Related third party imports
from sqlalchemy import create_engine

# Local application specific imports
from util import mapping


# DB2 Warehouse credentials
SQL_URL = os.environ["SQL_URL"]
SQL_USERNAME = os.environ["SQL_USERNAME"]
SQL_PASSWORD = os.environ["SQL_PASSWORD"]
SQL_DATABASE = os.environ["SQL_DATABASE"]
SQL_PORT = os.environ["SQL_PORT"]


# Initiate communication to db2. 
#sql_url = "db2+ibm_db://{username}:{password}@{host}:{port}/{database};Security=ssl;".format(username=SQL_USERNAME, password=SQL_PASSWORD, host=SQL_URL, port=SQL_PORT, database=SQL_DATABASE)
sql_url = "db2+ibm_db://{}:{}@{}:{}/{};Security=ssl;".format(SQL_USERNAME, 
                                                             SQL_PASSWORD, 
                                                             SQL_URL, 
                                                             SQL_PORT, 
                                                             SQL_DATABASE
                                                             )
conn = create_engine(sql_url)

secondary_country_list = ["England","Scotland","Wales","Northern Ireland"]

"""
Args:
    file_loc (str): The file location of the spreadsheet
    print_cols (bool): A flag used to print the columns to the console
        (default is False)

Returns:
    list: a list of strings representing the header columns
"""


def push_db2(feedback=None):   
    # 'feedback' will come from webhoo
    now = datetime.now()
    date = now.strftime("%Y-%m-%d %H:%M:%S")
    conn.execute("INSERT INTO \"CHATBOT_FEEDBACK\" (FEEDBACK, DATE) VALUES ('{}', '{}')".format(feedback[:1020], date))

def push_logs_db2(node_name=None, user_input=None, output=None, entities=None, intents=None, helpful=None):   
    
    def ListToStr(list_):
        str_val = ""
        for ind_, item in enumerate(list_):
            if ind_ == len(list_)-1:
                str_val += item
            else:
                str_val += item + ", "
        return str_val

    entt_ = []
    vals_ = []

    for i in entities:
        entt_.append(i["entity"])
        vals_.append(i["value"])
    
    now = datetime.now()
    date = now.strftime("%Y-%m-%d %H:%M:%S")
    conn.execute("INSERT INTO \"CHATBOT_LOGS_POC2\" (DATE, NODE_NAME, USER_INPUT, OUTPUT, INTENTS, ENTITIES, ENTITIES_VAL, HELPFULNESS) VALUES ('{}', '{}', '{}', '{}', '{}', '{}', '{}', '{}')".format(date, node_name, user_input, output[0], intents[0]["intent"], ListToStr(entt_), ListToStr(vals_), helpful))


def overall_confirmed_cases(country):
    if country in secondary_country_list:
        lastentry_date = pd.read_sql("SELECT max(datetime_date) FROM oxford_stringency_index WHERE regionname='{}' AND ConfirmedCases IS NOT NULL AND ConfirmedDeaths IS NOT NULL".format(country),conn)
        overall_cases = pd.read_sql("SELECT ConfirmedCases FROM oxford_stringency_index WHERE regionname='{}' AND datetime_date='{}'".format(country,lastentry_date.iloc[0].values[0]),conn)
        return int(overall_cases.iloc[0].values[0]), lastentry_date.iloc[0].values[0]
    else:
        lastentry_date = pd.read_sql("SELECT max(datetime_date) FROM oxford_stringency_index WHERE countryname='{}' AND ConfirmedCases IS NOT NULL AND ConfirmedDeaths IS NOT NULL AND jurisdiction = '{}'".format(country, "NAT_TOTAL"),conn)
        overall_cases = pd.read_sql("SELECT ConfirmedCases FROM oxford_stringency_index WHERE countryname='{}' AND datetime_date='{}' AND jurisdiction = '{}'".format(country,lastentry_date.iloc[0].values[0],"NAT_TOTAL"),conn)
        return int(overall_cases.iloc[0].values[0]), lastentry_date.iloc[0].values[0]

def overall_confirmed_deaths(country):
    if country in secondary_country_list:
        lastentry_date = pd.read_sql("SELECT max(datetime_date) FROM oxford_stringency_index WHERE regionname='{}' AND ConfirmedDeaths IS NOT NULL AND ConfirmedCases IS NOT NULL".format(country),conn)
        overall_deaths = pd.read_sql("SELECT ConfirmedDeaths FROM oxford_stringency_index WHERE regionname='{}' AND datetime_date='{}'".format(country,lastentry_date.iloc[0].values[0]),conn)
        return int(overall_deaths.iloc[0].values[0]), lastentry_date.iloc[0].values[0]
    else:
        lastentry_date = pd.read_sql("SELECT max(datetime_date) FROM oxford_stringency_index WHERE countryname='{}' AND ConfirmedDeaths IS NOT NULL AND ConfirmedCases IS NOT NULL AND jurisdiction = '{}'".format(country, "NAT_TOTAL"),conn)
        overall_deaths = pd.read_sql("SELECT ConfirmedDeaths FROM oxford_stringency_index WHERE countryname='{}' AND datetime_date='{}' AND jurisdiction = '{}'".format(country,lastentry_date.iloc[0].values[0],"NAT_TOTAL"),conn)
        return int(overall_deaths.iloc[0].values[0]), lastentry_date.iloc[0].values[0]


def deathrate(country):
    """Returns the covid-19 fatality rate for specific country."""
    cases, lastentry_date = overall_confirmed_cases(country)
    deaths, _ = overall_confirmed_deaths(country)

    return (deaths/cases)*100, lastentry_date



def hotspots(sample_size):
    """Return the number of hotspots in UK and a list with random choosen hotspots areas.
    
    Args:
        sample_size (int): The size of the list to return
    """
    
    df = pd.read_sql("SELECT \"date\", \"Area_Name\" FROM EALUSER.COV19_HOTSPOTS_SURVEILLANCE WHERE QUADRANT_INFECTIONS = '{}' OR QUADRANT_INFECTIONS = '{}'".format(1,4),conn)
    df.date = pd.to_datetime(df.date)
    latest = df.date.max() - timedelta(1) # Get the yesterday's data (more robust)
    df = df[df.date == latest]
    
    size = df.shape[0]
    
    df_new = df.sample(sample_size)
    hotspot_list = df_new.Area_Name.values
    x = ", ".join([i for i in list(hotspot_list)]) #convert list to string for the chatbot
    
    return x, size



def counter_hotspots(usr_region):
    """Check if an area in UK is a hotspot.
    
    Args:
        usr_region (str): The name of the area
    """
    
    usr_region = usr_region.title() # fix the name format to match our database entries
    
    df = pd.read_sql("SELECT \"date\", \"Area_Name\" FROM EALUSER.COV19_HOTSPOTS_SURVEILLANCE WHERE (QUADRANT_INFECTIONS = '{}' OR QUADRANT_INFECTIONS = '{}') AND \"Area_Name\"='{}'".format(1,4, usr_region),conn)
    df.date = pd.to_datetime(df.date)
    latest = df.date.max() - timedelta(1) # Get the yesterday's data (more robust)
    
    df = df[df.date == latest]
    if len(df) > 0:
        return "Yes, {} is a hotspot area related to covid so you need to be careful if you are planning to travel there.".format(usr_region)
    else:
        return "No, {} is not a hotspot right now.".format(usr_region)
    


def infection_risk(ltla_name):
    """Check the infection risk of a lower-tier-local-authority area in UK.
    
    Args:
        ltla_name (str): The name of the area
    """
    
    utla_name = mapping(ltla_name.title()) #function to map the local-tier to upper tier local authority area

    df = pd.read_sql("SELECT RISK_INDEX,DATE FROM EALUSER.EMERGENT_RISK_INDEX WHERE COUNTRY = '{}' AND AREA_NAME = '{}' AND AREA_TYPE = '{}'".format("England", utla_name, "UTLA"),conn)
    df.DATE = pd.to_datetime(df.DATE)
    latest = df.DATE.max() - timedelta(6) #Need to add comm to that!!!!!!!!!!!!!!!!!!!!!!!
    df = df[df.DATE == latest]
    
    risk_index = df.risk_index.values[0]

    if risk_index < 20:
        return "very low health risk index (risk = {} - on a scale from 0 to 100)".format(risk_index)
    elif risk_index >= 20 and risk_index < 50:
        return "low health risk index (risk = {} - on a scale from 0 to 100)".format(risk_index)
    elif risk_index >= 50 and risk_index < 70:
        return "medium health risk index (risk = {} - on a scale from 0 to 100)".format(risk_index)
    elif risk_index >=70 and risk_index < 90:
        return "high health risk index (risk = {} - on a scale from 0 to 100)".format(risk_index)
    else:
        return "very high health risk index (risk = {} - on a scale from 0 to 100)".format(risk_index)

    
    
def travel_risk(ltla_destination, ltla_origin):
    """Return a traveling advice for journey within UK.
    
    Args:
        ltla_origin (str): The area name you start your journey
        ltla_destination (str): The area name you end your journey
    """
        
    #functions to map the local-tier to upper tier local authority area
    utla_destination = mapping(ltla_destination.title()) 
    utla_origin = mapping(ltla_origin.title())

    df_dest = pd.read_sql("SELECT RISK_INDEX,DATE FROM EALUSER.EMERGENT_RISK_INDEX WHERE COUNTRY = '{}' AND AREA_NAME = '{}' AND AREA_TYPE = '{}'".format("England", utla_destination, "UTLA"),conn)
    df_origin = pd.read_sql("SELECT RISK_INDEX,DATE FROM EALUSER.EMERGENT_RISK_INDEX WHERE COUNTRY = '{}' AND AREA_NAME = '{}' AND AREA_TYPE = '{}'".format("England", utla_origin, "UTLA"),conn)

    # Calculate the Risk Index of the destination area
    df_dest.DATE = pd.to_datetime(df_dest.DATE)
    latest = df_dest.DATE.max() - timedelta(6) # Need to comment this one.
    df_dest = df_dest[df_dest.DATE == latest]
    dest_risk = df_dest.risk_index.values[0]

    # Calculate the Risk Index of the origin area
    df_origin.DATE = pd.to_datetime(df_origin.DATE)
    latest = df_origin.DATE.max() - timedelta(6)
    df_origin = df_origin[df_origin.DATE == latest]
    origin_risk = df_origin.risk_index.values[0]
    
    def safe_or_not_v2(ltla_destination, dest_risk, ltla_origin, origin_risk):
        if dest_risk >= 50:
            return "It is not recommneded to travel to {} tomorrow as we predicting a high risk index in the area (Risk index = {})".format(ltla_destination, dest_risk)
        elif dest_risk < 50:
            return "It is okay to travel to {} tomorrow as we predicting a low risk index in the area (Risk index = {})".format(ltla_destination, dest_risk)

    return safe_or_not_v2(ltla_destination,dest_risk, ltla_origin, origin_risk)



def international_travel_risk(country_origin, country_dest): # ANANDA TO FILL THAT
    """Return a traveling advice based on international 
    travel restrictions around the globe.
    
    Args:
        country_origin (str): The country name you start your journey
        country_dest (str): The country name you end your journey
    """
    
    username = "EALUSER"
    text_user_origin = ""
    text_user_dest = ""
    
    def get_links(df, country):
        if len(df) == 1:
            data_link = df['sources'].values[0]
            url_find = re.findall(r'<a href[A-Za-z0-9 \s\S]*</a>/', data_link)
            return url_find
        else:
            df = df[df["adm0_name"] == country]
            data_link = df['sources'].values[0]
            url_find = re.findall(r'<a href[A-Za-z0-9 \s\S]*</a>/', data_link)
            return url_find
    
    if country_origin in secondary_country_list:
        user_origin = country_origin
        text_user_origin = "{} is not in our database so we will provide information for United Kingdom: <br><br>".format(user_origin)
        #text_user_origin = "We don't have the travel restrictions for {} but I can give you information regarding United Kingdom: <br>".format(user_origin)
        country_origin = "United Kingdom"
    
    if country_dest in secondary_country_list:
        user_dest = country_dest
        text_user_dest = "{} is not in our database so we will provide information for United Kingdom: <br><br>".format(user_dest)
        #text_user_dest = "We don't have the travel restrictions for {} but I can give you information regarding United Kingdom: <br>".format(user_dest)
        country_dest = "United Kingdom"
    
    if (country_origin == country_dest):
        if (country_origin=="United Kingdom"):
            return "Unfortunately we don't support domestic traveling within United Kingdom",""
        else:
            return "Unfortunately we don't support domestic traveling",""
    
    in_db = True
    
    df_data_src = pd.read_sql("SELECT \"ADM0_A3\" FROM {}.JOHNS_HOPKINS_COUNTRY_MAPPING WHERE \"NAME\" = '{}'".format(username.upper(), country_origin.title()),conn)
    try:
        data_src = df_data_src.iloc[0].values[0]
    except IndexError:
        in_db = False
        country_not_db = str(country_origin)
    
    df_data_dst = pd.read_sql("SELECT \"ADM0_A3\" FROM {}.JOHNS_HOPKINS_COUNTRY_MAPPING WHERE \"NAME\" = '{}'".format(username.upper(), country_dest.title()),conn)
    try:
        data_dst = df_data_dst.iloc[0].values[0]
    except IndexError:
        in_db = False
        country_not_db = str(country_dest)


    if in_db:
        df_latest_date = pd.read_sql("SELECT MAX(DOWNLOAD_DATE) FROM TRAVEL_RESTRICTIONS_RESULTS WHERE \"HOME\" = '{}' AND \"OTHER\" = '{}'".format(data_dst, data_src),conn)
        latest_date = df_latest_date.iloc[0].values[0]

        df_data_riskidx = pd.read_sql("SELECT \"RESTRICTION\" FROM TRAVEL_RESTRICTIONS_RESULTS WHERE \"DOWNLOAD_DATE\" = '{}' AND \"HOME\" = '{}' AND \"OTHER\" = '{}'".format(latest_date,data_dst, data_src),conn)
        data_riskidx = df_data_riskidx.iloc[0].values[0]
        #print(df_data_riskidx)

        df_latest_date_links = pd.read_sql("SELECT MAX(DOWNLOAD_DATE) FROM EALUSER.TRAVEL_RESTRICTIONS_COUNTRY WHERE ADM0_NAME LIKE '%{}%'".format(country_dest.title()),conn)
        latest_date = df_latest_date_links.iloc[0].values[0] #latest date for links is different that the restrictions value (from 3 lines before)

        if(data_riskidx == None):

            df_data_link = pd.read_sql("SELECT \"ADM0_NAME\",\"SOURCES\" FROM EALUSER.TRAVEL_RESTRICTIONS_COUNTRY WHERE DOWNLOAD_DATE = '{}' AND ADM0_NAME LIKE '%{}%'".format(latest_date, country_dest.title()),conn)

            url_find = get_links(df_data_link,country_dest)
            response = text_user_origin + text_user_dest + "The data provided by {} does not provide any information regarding travelling from {}. Please check the following links for more information".format(country_dest.title(), country_origin.title())
            links = "For more Information: <br> {} <br>".format(str(np.squeeze(url_find)).replace("\n", "<br>").replace("</a>/<br>", "</a> <br>").replace("\"\"", "\"").replace("</a>/ <br>", "</a> <br>").replace("<br>\"<a", "<br> <a").replace("</a>/\"<br>", "</a> <br>").replace("</a>/\"", "</a>").replace("</a>/", "</a>"))
            return response, links

        elif(int(float(data_riskidx)) == 0):
            return text_user_origin + text_user_dest + "You are not allowed to enter {} when you travel from {} because of travel restrictions".format(country_dest.title(),country_origin.title()),""
        elif(int(float(data_riskidx)) == 3):
            return text_user_origin + text_user_dest + "You are allowed to enter {} when you travel from {}".format(country_dest.title(),country_origin.title()),""
        elif(int(float(data_riskidx)) == 2 or int(float(data_riskidx)) == 1):

            df_data_link = pd.read_sql("SELECT \"ADM0_NAME\",\"SOURCES\" FROM EALUSER.TRAVEL_RESTRICTIONS_COUNTRY WHERE DOWNLOAD_DATE = '{}' AND ADM0_NAME LIKE '%{}%'".format(latest_date, country_dest.title()),conn)
            url_find = get_links(df_data_link,country_dest)            
            response = text_user_origin + text_user_dest + "There are some restrictions applied regarding traveling to {} from {}. Please check the following links for more information".format(country_dest.title(),country_origin.title())
            links = "For more Information: <br> {} <br>".format(str(np.squeeze(url_find)).replace("\n", "<br>").replace("</a>/<br>", "</a> <br>").replace("\"\"", "\"").replace("</a>/ <br>", "</a> <br>").replace("<br>\"<a", "<br> <a").replace("</a>/\"<br>", "</a> <br>").replace("</a>/\"", "</a>").replace("</a>/", "</a>"))

            return response, links
    else:
        response = "Unfortunately {} is not in our database.".format(country_not_db)
        return response, ""


    
def lockdown_measures(country):
    """Return information about the lockdown measures of a specific country 
    
    Args:
        country (str): The country you are interested in
        
    Returns:
        text (str): Sentence containing the lockdown measures
        lastentry_date (datetime): Date of the lockdown measures taking place
        num_of_measures (int): number of lockdown measures
        response (list): list of the lockdown measures
        options_result (dict): dictionary of the lockdown measures
    """

    df_data_1 = pd.read_csv('./chatbot_lockdown_measures.csv') #Csv file containing responses

    # Pull the data from database
    if country in secondary_country_list:
        lastentry_date = pd.read_sql("SELECT max(datetime_date) FROM oxford_stringency_index WHERE regionname='{}' AND \"c1_school closing\" IS NOT NULL".format(country),conn)
        df = pd.read_sql("SELECT * FROM oxford_stringency_index WHERE regionname='{}' AND datetime_date='{}'".format(country,lastentry_date.iloc[0].values[0]),conn)       
    else:
        lastentry_date = pd.read_sql("SELECT max(datetime_date) FROM oxford_stringency_index WHERE countryname='{}' AND \"c1_school closing\" IS NOT NULL AND jurisdiction = '{}'".format(country, "NAT_TOTAL"),conn)
        df = pd.read_sql("SELECT * FROM oxford_stringency_index WHERE countryname='{}' AND datetime_date='{}' AND jurisdiction = '{}'".format(country,lastentry_date.iloc[0].values[0], "NAT_TOTAL"),conn)
        #print("df ",df)

    if df.empty:
        text = "There are no recorded lockdown measures for {}".format(country)
        return text, lastentry_date, 0, [],{}
    

    df_CONTAINMENT = df[['c1_school closing','c2_workplace closing', 
                     'c3_cancel public events','c4_restrictions on gatherings', 'c5_close public transport',
                     'c6_stay at home requirements', 'c7_restrictions on internal movement', 'c8_international travel controls']]
    df_CONTAINMENT = df_CONTAINMENT.transpose()
    df_important = df_CONTAINMENT[df_CONTAINMENT > 1]
    df_important.dropna(inplace=True)
    #print("df_important ",df_important)
    num_of_measures = df_important.shape[0]
    #print("num_of_measures: ", num_of_measures)
    
    lock_measures_list = df_important.index.values
    response = []

    # For loop to return the most important lockdown measures in a sentence
    for lock_measure in lock_measures_list:
        value = int(df_important.loc[lock_measure].values)
        data = df_data_1['name'].loc[(df_data_1['stringency index'] == lock_measure) & (df_data_1['value'] == value)].iloc[0]
        response.append(data)
        text1 = ", ".join([i for i in response[:-1]])
        text2 = " and "+ response[-1]
        text = text1 + text2
    
    index_list = ["measure"+str(i+1) for i in range(len(lock_measures_list))]
    options_result = dict(zip(index_list, response)) # convert dictionary to list
    print(options_result)
    
    return text, lastentry_date, num_of_measures, response, options_result


def lockdown_measures_extended(country, measure):
    """Return information about a specific lockdown measure of a specific country 
    
    Args:
        country (str): The country you are interested in
        measure (str): The lockdown measure you are interested in
    """

    #Read csv file with responses:
    df_data_1 = pd.read_csv('./chatbot_lockdown_measures.csv')

    if country in secondary_country_list:
        lastentry_date = pd.read_sql("SELECT max(datetime_date) FROM oxford_stringency_index WHERE regionname='{}' AND \"c1_school closing\" IS NOT NULL".format(country),conn)
        df = pd.read_sql("SELECT * FROM oxford_stringency_index WHERE regionname='{}' AND datetime_date='{}'".format(country,lastentry_date.iloc[0].values[0]),conn)       
    else:
        lastentry_date = pd.read_sql("SELECT max(datetime_date) FROM oxford_stringency_index WHERE countryname='{}' AND \"c1_school closing\" IS NOT NULL AND jurisdiction = '{}'".format(country, "NAT_TOTAL"),conn)
        df = pd.read_sql("SELECT * FROM oxford_stringency_index WHERE countryname='{}' AND datetime_date='{}' AND jurisdiction = '{}'".format(country,lastentry_date.iloc[0].values[0], "NAT_TOTAL"),conn)
    
    stringency_index = df_data_1['stringency index'].loc[(df_data_1['name'] == measure)].iloc[0]
    
    df_CONTAINMENT = df[stringency_index]
    value = int(df_CONTAINMENT.values[0])
    
    response = df_data_1['Description'].loc[(df_data_1['stringency index'] == stringency_index) & (df_data_1['value'] == value)].iloc[0]

    return response


 