#!/usr/bin/python3
import numpy
from states import state_info, state_historic_data

def case_growth_rate(data):
    positives = map(lambda x: 0 if x['positive'] is None else x['positive'], data)
    nonzero = filter(lambda x: x > 0, positives)
    y = numpy.array(list(nonzero))
    return weighted_exponential_fit(numpy.arange(len(y)), y)

def death_growth_rate(data):
    deaths = map(lambda x: 0 if x['death'] is None else x['death'], data)
    nonzero = filter(lambda x: x > 0, deaths)
    y = numpy.array(list(nonzero))
    return weighted_exponential_fit(numpy.arange(len(y)), y)

def exponential_fit(x, y):
    return numpy.polyfit(x,numpy.log(y),1)

def weighted_exponential_fit(x, y):
    return numpy.polyfit(x,numpy.log(y),1,w=numpy.sqrt(y))

def to_exponential_function(fit):
    return "y=" + str(numpy.exp(fit[1])) +  "*exp(" + str(fit[0]) + "*t)"

def doubling_time(fit):
    return numpy.log(2)/fit[0]

if __name__ == "__main__":
    si = state_info()
    print("Average case doubling time in days for past 7 days of data")
    for state in si.get_states():
        data = state_historic_data(state)
        latest_data = data.get_latest_n(7)
        fit = case_growth_rate(latest_data)
        print(state + "," + str(doubling_time(fit)))

    print("Average death doubling time in days for past 7 days of data")
    for state in si.get_states():
        data = state_historic_data(state)
        latest_data = data.get_latest_n(7)
        fit = death_growth_rate(latest_data)
        print(state + "," + str(doubling_time(fit)))

    print("Average case doubling time in days since 10th confirmed case")
    for state in si.get_states():
        data = state_historic_data(state)
        latest_data = data.get_after_n_cases(10)
        fit = death_growth_rate(latest_data)
        print(state + "," + str(doubling_time(fit)))
