
import requests
import json
from jsonschema import validate
import wget
from os.path import isfile
from os import remove
import pandas as pd


class MeltWaterClient:
    
    class RequestHandler:
        
        def __init__(self, url, user_key, access_token):
            self.__user_key__ = user_key
            self.__access_token__ = access_token
            
        def get(self, url):
            """
            Sends an authenticated HTTP request with the GET method to the specified Meltwater endpoint.
            Handles the error, if any, or returns the requested resource in JSON format.
            """
            response = requests.get(url, 
                                    headers={
                                        "Accept": "application/json",
                                        "user-key": self.__user_key__,
                                        "Authorization": f"Bearer {self.__access_token__}"
                                    })
            if not response.ok:
                print(f"{response.status_code}: {response.reason}")
                print(response.json())
                raise Exception("API request failed")
            else:
                return response.json()
            
        def delete(self, url):
            """
            Sends an authenticated HTTP request with the DELETE method to the specified Meltwater endpoint.
            Handles the error, if any.
            """
            response = requests.delete(url, 
                                       headers={
                                           "Accept": "application/json",
                                           "user-key": self.__user_key__,
                                           "Authorization": f"Bearer {self.__access_token__}"
                                       })
            if not response.ok:
                print(f"{response.status_code}: {response.reason}")
                print(response.json())
                raise Exception("API request failed")
            
        def post(self, url, payload):
            """
            Sends an authenticated HTTP request with the POST method to the specified Meltwater endpoint.
            Handles the error, if any, or returns the created resource in JSON format.
            """
            response = requests.post(url, 
                                     headers={
                                         "Content-Type": "application/json",
                                         "Accept": "application/json",
                                         "user-key": self.__user_key__,
                                         "Authorization": f"Bearer {self.__access_token__}"
                                     },
                                     data=json.dumps(payload))
            if not response.ok:
                print("Search error")
                print(f"{response.status_code}: {response.reason}")
                print(response.json())
                raise Exception("API request failed")
            else:
                return response.json()
            
        def put(self, url, payload):
            """
            Sends an authenticated HTTP request with the PUT method to the specified Meltwater endpoint.
            Handles the error, if any.
            """
            response = requests.put(url, 
                                    headers={
                                        "Content-Type": "application/json",
                                        "Accept": "application/json",
                                        "user-key": self.__user_key__,
                                        "Authorization": f"Bearer {self.__access_token__}"
                                    },
                                    data=json.dumps(payload))
            if not response.ok:
                print("Search error")
                print(f"{response.status_code}: {response.reason}")
                print(response.json())
                raise Exception("API request failed")
                
            
    class Searches(RequestHandler):
        
        def __init__(self, api_version, user_key, access_token):
            self.endpoint_url = f"https://api.meltwater.com/export/v{api_version}/searches"
            self.api_version = api_version
            super().__init__(self.endpoint_url, user_key, access_token)
        
        def __validate_search_id__(self, search_id) -> None:
            """Validates the ID of the requested search."""
            if not isinstance(search_id, int):
                raise TypeError("The ID must be an integer.")
            if search_id <= 0:
                raise ValueError("The ID must be a positive integer.")
                
        def get(self, search_id: int=None) -> object:
            """Get an individual search if search_id is specified, returns a list of all searches otherwise."""
            endpoint_url = self.endpoint_url
            if self.api_version == 1:
                if search_id != None:
                    self.__validate_search_id__(search_id)
                    endpoint_url = f"{endpoint_url}/{search_id}"
                return super().get(endpoint_url)
            else:
                raise Exception(f"API version {self.api_version} not supported.")
        
        def delete(self, search_id: int) -> None: 
            """Delete an individual search."""
            endpoint_url = self.endpoint_url
            self.__validate_search_id__(search_id)
            if self.api_version == 1:
                endpoint_url = f"{endpoint_url}/{search_id}"
                super().delete(endpoint_url)
            else:
                raise Exception(f"API version {self.api_version} not supported.")
        
        def create(self, params: object, dry_run: bool=False) -> object:
            """Create a search."""
            endpoint_url = self.endpoint_url
            if self.api_version == 1:
                if dry_run:
                    endpoint_url = f"{endpoint_url}?dry_run=true"
                return super().post(endpoint_url, payload=params)
            else:
                raise Exception(f"API version {self.api_version} not supported.")
                
        def update(self, search_id: str, params: object, dry_run: bool=False) -> object:
            """Update an individual search."""
            endpoint_url = self.endpoint_url
            self.__validate_search_id__(search_id)
            if self.api_version == 1:
                endpoint_url = f"{endpoint_url}/{search_id}"
                if dry_run:
                    endpoint_url = f"{endpoint_url}?dry_run=true"
                return super().put(endpoint_url, payload=params)
            else:
                raise Exception(f"API version {self.api_version} not supported.")
        
        def count(self, search_id: int) -> object:
            """Get an approximate count of results for the search over a particual period."""
            endpoint_url = self.endpoint_url
            if self.api_version == 1:
                self.__validate_search_id__(search_id)
                endpoint_url = f"{endpoint_url}/{search_id}/count"
                response = super().get(endpoint_url)
                return response["count"]["total"]
            else:
                raise Exception(f"API version {self.api_version} not supported.")
        
        
    class Exports(RequestHandler):
        
        def __init__(self, api_version, user_key, access_token):
            self.endpoint_url = f"https://api.meltwater.com/export/v{api_version}/exports/one-time"
            self.api_version = api_version
            super().__init__(self.endpoint_url, user_key, access_token)
        
        def __validate_export_id__(self, export_id: int) -> None:
            if not isinstance(export_id, int):
                raise TypeError("The ID must be an integer.")
            if export_id <= 0:
                raise ValueError("The ID must be a positive integer.")
       
        def get(self, export_id: int=None) -> object:
            """Get details about a one-time export if export_id is specified, returns a list of all one-time exports otherwise."""
            endpoint_url = self.endpoint_url
            if self.api_version == 1:
                if export_id != None:
                    self.__validate_export_id__(export_id)
                    endpoint_url = f"{endpoint_url}/{export_id}"
                return super().get(endpoint_url)
            else:
                raise Exception(f"API version {self.api_version} not supported.")
                
        def delete(self, export_id: int) -> None: 
            """Removes an existing recurring export."""
            endpoint_url = self.endpoint_url
            self.__validate_export_id__(export_id)
            if self.api_version == 1:
                endpoint_url = f"{endpoint_url}/{export_id}"
                super().delete(endpoint_url)
            else:
                raise Exception(f"API version {self.api_version} not supported.")
                
        def create(self, params: object) -> object:
            """Creates a new one-time export."""
            endpoint_url = self.endpoint_url
            if self.api_version == 1:
                return super().post(endpoint_url, payload=params)
            else:
                raise Exception(f"API version {self.api_version} not supported.")
                
        def load(self, export_id: int) -> pd.DataFrame:
            """
            Loads a one-time export in a Pandas dataframe.
            The one-time export must have the status 'FINISHED'.
            """
            self.__validate_export_id__(export_id)
            # Define temporary file to save the downloaded data
            filename_raw_data = "to_be_deleted"
            if isfile(filename_raw_data):
                remove(filename_raw_data)
            # Get the one-time export details from Meltwater
            response = self.get(export_id)
            # Get the URL of the data file
            if not "onetime_export" in response or not "status" in response["onetime_export"] or response["onetime_export"]["status"] != "FINISHED":
                raise Exception("The export job is not finished.")
            data_url = response["onetime_export"]["data_url"]
            # Download the data
            wget.download(data_url, filename_raw_data)
            with open(filename_raw_data, "r", encoding="utf-8") as f:
                data = json.loads(f.read())
            # Remove the temporary file
            if isfile(filename_raw_data):
                remove(filename_raw_data)
            # Return the data in a Pandas DataFrame
            return pd.DataFrame.from_records(data["data"])
        
        
    def __init__(self, params: object):
        
        schema = {
            "type" : "object",
            "properties" : {
                "user_key": {"type": "string"},
                "client_id": {"type": "string"},
                "client_secret": {"type": "string"},
                "api_url": {"type": "string"},
                "version": {"type": "number"},
                "access_token": {"type": "string"},
            },
            "required": ["user_key", "client_id", "client_secret", "version"]
        }
        validate(instance=params, schema=schema)

        self.__user_key__ = params["user_key"]
        self.__client_id__ = params["client_id"]
        self.__client_secret__ = params["client_secret"]
        if "api_url" in params:
            self.__api_url__ = params["api_url"]
        else:
            self.__api_url__ = "https://api.meltwater.com"
        self.__version__ = params["version"]
        if "access_token" in params:
            self.__access_token__ = params["access_token"]
        else:
            self.__auth__()
        
        self.searches = self.Searches(self.__version__, self.__user_key__, self.__access_token__)
        self.exports = self.Exports(self.__version__, self.__user_key__, self.__access_token__)
    
    def __auth__(self):
        """Authenticate to Meltwater."""
        oauth_url = f"{self.__api_url__}/oauth2/access_token"
        response = requests.post(oauth_url, 
                                 auth=(self.__client_id__, self.__client_secret__),
                                 headers={
                                     "content-type": "application/x-www-form-urlencoded",
                                     "user-key": self.__user_key__
                                 },
                                 data="grant_type=client_credentials&scope=search")
        if not response.ok:
            print("Authentication error")
            print(f"{response.status_code}: {response.reason}")
            print(response.json())
        else:
            response = response.json()
            self.__access_token__ = response["access_token"]
            print(response)