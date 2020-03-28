#!/usr/bin/python3

import urllib3
import json
from state_info import state_info

def get_state_current(state):
    return get_api("https://covidtracking.com/api/states?state="+state)

def get_state_historic(state):
    return get_api("https://covidtracking.com/api/states/daily?state="+state)

def get_api(url):
    http = urllib3.PoolManager()
    request = http.request("GET",url)
    if request.status == 200:
        response = json.loads(request.data)
    else:
        print("ERROR: Could not read API data, got status " + str(hr.status) + " " + hr.reason)
        response = {}
    return response

def save_data(name,data):
    with open("data/" + name + ".json","w") as fp:
        json.dump(data,fp)

if __name__ == "__main__":
    si = state_info()
    for state in si.get_states():
        data = get_state_historic(state)
        save_data(state + "_historic", data)
