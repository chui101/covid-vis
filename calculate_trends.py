#!/usr/bin/python3
import numpy
from states import state_info, state_historic_data
from matplotlib import pyplot

def growth_rate(data, data_name='positive'):
    values = map(lambda x: 0 if (data_name not in x or x[data_name] is None) else x[data_name], data)
    nonzero = filter(lambda x: x > 0, values)
    y = numpy.array(list(nonzero))
    return weighted_exponential_fit(numpy.arange(len(y)), y)

def death_growth_rate(data):
    deaths = map(lambda x: 0 if ('death' not in x or x['death'] is None) else x['death'], data)
    nonzero = filter(lambda x: x > 0, deaths)
    y = numpy.array(list(nonzero))
    return weighted_exponential_fit(numpy.arange(len(y)), y)

def exponential_fit(x, y):
    return numpy.polyfit(x,numpy.log(y),1)

def weighted_exponential_fit(x, y):
    return numpy.polyfit(x,numpy.log(y),1,w=numpy.sqrt(y))

def to_exponential_function(fit):
    return "y=" + str(numpy.exp(fit[1])) +  "*exp(" + str(fit[0]) + "*t)"

def doubling_time(data, data_name='positive'):
    fit = growth_rate(data, data_name)
    return numpy.log(2)/fit[0]

if __name__ == "__main__":
    si = state_info()
    states = si.get_states()

    # calculate doubling time for last seven days for all tracked states
    doubling_time_last_seven = []
    for state in states:
        data = state_historic_data(state).get_latest_n(7)
        doubling_time_last_seven.append(doubling_time(data))
    pyplot.bar(range(len(states)), doubling_time_last_seven, tick_label=list(states))
    pyplot.title("Average doubling time over past 7 days")
    pyplot.ylabel("Average doubling time (days)")
    fig = pyplot.gcf()
    fig.set_size_inches(10, 7)
    pyplot.show()
    pyplot.close()

    # calculate doubling time for each seven day window for all tracked states
    for state in ['KY',"TN","NY","IN","LA","OH"]:
        data = state_historic_data(state).data
        doubling_time_last_seven = []
        if len(data) > 7:
            for i in range(len(data)-6):
                doubling_time_last_seven.append(doubling_time(data[i:i+7]))
            pyplot.plot(range(3,len(doubling_time_last_seven)+3), doubling_time_last_seven, label=state)
    pyplot.title("Average doubling time (7 day moving average, higher is better)")
    pyplot.ylabel("Average case doubling time (days)")
    pyplot.xlabel("Days since first reported case")
    pyplot.legend()
    pyplot.grid(b=True)
    fig = pyplot.gcf()
    fig.set_size_inches(10, 7)
    pyplot.show()
    pyplot.close()


    print("Average death doubling time in days for past 7 days of data")
    for state in states:
        data = state_historic_data(state)
        latest_data = data.get_latest_n(7)
        print(state + "," + str(doubling_time(latest_data,'death')))

    print("Average case doubling time in days since 10th confirmed case")
    for state in states:
        data = state_historic_data(state)
        latest_data = data.get_after_n_cases(10)
        print(state + "," + str(doubling_time(latest_data)))
