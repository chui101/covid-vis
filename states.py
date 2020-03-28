import json

class state_info:
    def __init__(self, datafile = "data/states.json"):
        with open(datafile,'r') as fp:
            self.data = json.load(fp)

    def get_states(self):
        return self.data.keys()

    def get_name(self, state):
        return self.data[state]['name']

    def get_area(self, state):
        return self.data[state]['area']

    def get_population(self, state):
        return self.data[state]['population']

    def get_density(self, state):
        return self.get_population(state)/self.get_area(state)

class state_historic_data:
    def __init__(self, state):
        self.state = state
        with open("data/" + state + "_historic.json",'r') as fp:
            self.data = json.load(fp)

    def get_latest(self):
        return self.data[-1]

    def get_latest_n(self, n):
        return self.data[-n:]

    def get_range(self, begin, end):
        return list(filter(lambda x: x['date'] >= begin and x['date'] <= end))
