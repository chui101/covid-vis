import json

class state_info:
    def __init__(self, datafile = "data/states.json"):
        with open(datafile,'r') as fp:
            self.data = json.load(fp)

    def get_states(self):
        """Gets a list of two letter state codes present in states.json"""
        return self.data.keys()

    def get_name(self, state):
        """Get the full name of the state from the two letter code"""
        return self.data[state]['name']

    def get_area(self, state):
        """Get the area of a state"""
        return self.data[state]['area']

    def get_population(self, state):
        """Get the population of a state (2019 estimate)"""
        return self.data[state]['population']

    def get_density(self, state):
        """Get the calculated density of a state"""
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

    def get_date_range(self, begin, end):
        return list(filter(lambda x: x['date'] >= begin and x['date'] <= end))

    def get_after_n_cases(self, n):
        """Get a set of data points after the state meets or exceeds the threshold number of cases.
        Searches backwards until it finds the last point positives meets or exceeds the threshold"""
        # find the index where the state meets the threshold
        last_index = len(self.data)-1
        for i in range(len(self.data)-1,-1,-1):
            if self.data[i]['positive'] >= n:
                last_index = i
        # slice array and return
        return self.data[last_index:]

