# Meltwater Python Client

This package provides an interface to access the export API of Meltwater. 
Note that not all the endpoints have been implemented at this point.

To test the package, simply do::

```python
import meltwater
meltwater = MeltWaterClient({
    "user_key": <user_key>,
    "client_id": <client_id>,
    "client_secret": <client_secret>,
    "version": 1
})
meltwater.searches.get()
```

If you already have an access token, you can re-use it by simply doing::

```python
import meltwater
meltwater = MeltWaterClient({
    "user_key": <user_key>,
    "client_id": <client_id>,
    "client_secret": <client_secret>,
    "version": 1,
    "access_token": <access-token>
})
meltwater.searches.get()
```

## Endpoints:

- Searches API
    - meltwater.searches.get()
    - meltwater.searches.create()
    - meltwater.searches.count()
    - meltwater.searches.update()
    - meltwater.searches.delete()

- One-Time Export API
    - meltwater.exports.get()
    - meltwater.exports.create()
    - meltwater.exports.delete()
    - meltwater.exports.load()


## Test the Searches API with the client:

Get a list of all your searches:

```python
meltwater.searches.get()
```

Create a new search

```python
new_search_obj = {
    "search": {
        "type": "news",
        "query": {
            "type": "boolean",
            "source_selection_id": 1,
            "case_sensitivity": "no",
            "boolean": "Tesla OR (Volvo NEAR electric)"
        },
        "name": "TEST SEARCH 123"
    }
}
new_search = meltwater.searches.create(new_search_obj)
print("New search:", new_search)
new_search_id = new_search["search"]["id"]
```

Get an individual search

```python
meltwater.searches.get(new_search_id)
```

Get an approximate count of results for the search over a particual period

```python
meltwater.searches.count(new_search_id)
```
Update an individual search

```python
updated_search_obj ={
    "search": {
        "type": "news",
        "query": {
            "type": "boolean",
            "source_selection_id": 1,
            "case_sensitivity": "no",
            "boolean": "Tesla OR (Volvo NEAR electric)"
        },
        "name": "TEST SEARCH 123 - 2"
    }
}
meltwater.searches.update(new_search_id, updated_search_obj)
```
Delete an individual search

```python
meltwater.searches.delete(new_search_id)
```

## Test the One-Time Export API with the client:

Get a list of all your one-time exports

```python
meltwater.exports.get()
```

Creates a new one-time export
```python
from datetime import datetime, timedelta
now = datetime.now().isoformat()
one_day_ago = (datetime.today() - timedelta(days=1)).isoformat()
new_export_obj = {
    "onetime_export": {
        "start_date": one_day_ago,
        "end_date": now,
        "search_ids": [new_search_id]
    }  
}
new_export = meltwater.exports.create(new_export_obj)
print("New export:", new_export)
new_export_id = new_export["onetime_export"]["id"]
```

Get details of a one-time export

```python
meltwater.exports.get(new_export_id)
```

Removes an existing one-time export

```python
meltwater.exports.delete(new_export_id)
```

Load a one-time export into a Pandas DataFrame

```python
import pandas as pd
df = meltwater.exports.load(new_export_id)
df.head()
```
