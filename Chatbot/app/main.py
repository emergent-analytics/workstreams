# Related third party imports
from fastapi import FastAPI, Depends
from fastapi.security.api_key import APIKey
from pydantic import BaseModel
from typing import List, Optional

# Local application specific imports
from security import get_api_key
import functions_v2 as func

app = FastAPI()

class Item(BaseModel):
    """My API request body format. Those 
    paremeters are being send from the chatbot
    """
    
    country: str=None
    casesvsdeaths: str=None
    func_number: int=None
    area_name: str=None
    areaorigin: str=None
    areadestination: str=None
    lockdown_measures: str=None
    country_origin: str=None
    country_dest: str=None
    feedback: str=None
    node_name: str=None
    user_input: str=None
    output: List[str]=None
    entities: List[dict]=None
    intents: List[dict] = None
    helpful: str=None


#---------
# Routes #
#---------

# index route
@app.get("/")
async def read_root():
    """My index roote for testing purposes"""
    return {"result": "Welcome to the covid-19 Chatbot Back-end"}

# authorisation route
@app.get("/secure_endpoint", tags=["test"])
async def get_open_api_endpoint(api_key: APIKey = Depends(get_api_key)):
    """To check if my authorisation process was succesfull"""
    return "Certification is accepted"

# main route
@app.post("/items/")
async def create_item(item: Item, api_key: APIKey = Depends(get_api_key)):
    """
    The main route that the chatbot calls using POST API. Based on
    the function parameter it returns the appropriate response to the chatbot. 
    
    if statements (item.func_number == ?):
        1: Return the overall number of deaths/cases of a country.
        2: Returns the covid-19 fatality rate for specific country.
        4: Checks the infection risk of a lower-tier-local-authority area in UK.
        5: Return the number of hotspots in UK and a list with random choosen hotspots areas.
        6: Check if an area in UK is a hotspot.
        7: Return a traveling advice based on international travel restrictions around the world.
        8: Return information about the all lockdown measures of a specific country.
        9: Return information about a specific lockdown measure of a specific country
        10: Return a traveling advice for journey within UK.
        20: Sends feedback in the DB2 table - CHATBOT_FEEDBACK
        21: Variables being saved to DB2 table - CHATBOT_LOGS_POC2
        31: Print variables and logs for debug
    """
    
    # Return the overall number of deaths/cases of a country
    if item.func_number == 1:
        if item.casesvsdeaths == "Deaths":
            try:
                response, lastentry_date = func.overall_confirmed_deaths(item.country)
                result = {"response": response,
                        "lastentry_date": "(Last update: {})".format(lastentry_date)}
            except IndexError:
                response = "not in our database unfortunately."
                result = {"response": response,
                        "lastentry_date": ""}
        elif item.casesvsdeaths == "Cases":
            try:
                response, lastentry_date = func.overall_confirmed_cases(item.country)
                result = {"response": response,
                        "lastentry_date": "(Last update: {})".format(lastentry_date)}
            except IndexError:
                response = "not in our database unfortunately."
                result = {"response": response,
                        "lastentry_date": ""}
        else:
            result = "Not Found"
    elif item.func_number == 2:
        try:
            response, lastentry_date = func.deathrate(item.country)
            response = round(response,3)
            result = {"response": str(response)+"%",
                    "lastentry_date": "(Last update: {})".format(lastentry_date)}
        except IndexError:
            response = "not in our database unfortunately."
            result = {"response": response,
                        "lastentry_date": ""}
    elif item.func_number == 4:
        result = func.infection_risk(item.area_name)
    elif item.func_number == 5:
        hot_list, size = func.hotspots(5)
        result = {"hot_list": hot_list,
                    "size": size}
    elif item.func_number == 6:
        result = func.counter_hotspots(item.area_name)
    elif item.func_number == 7:
        response, links = func.international_travel_risk(item.country_origin, item.country_dest)  
        result = {"response": response,
                    "links": links}
    elif item.func_number == 8:
        result = func.lockdown_measures(item.country)
    elif item.func_number == 9:
        result = func.lockdown_measures_extended(item.country, item.lockdown_measures)
    elif item.func_number == 10:
        result = func.travel_risk(item.areadestination, item.areaorigin)
    elif item.func_number == 20:
        func.push_db2(item.feedback)
        result = "Feedback pushed on Db2"
    elif item.func_number == 21:
        func.push_logs_db2(item.node_name, item.user_input, item.output, item.entities, item.intents, item.helpful)
        #Add the code for the logs to send to db2.
        result="Push successfully"
    elif item.func_number == 31:
        print("node_name: ",item.node_name)
        print("user_input: ",item.user_input)
        print("output: ",item.output)
        print("entities: ",item.entities)
        print("intents: ",item.intents)
        print("helpful: ",item.helpful)
        result="variables send successful!!"
    else:
        result = "Not Found"

    return {"result": result}