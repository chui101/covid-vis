from datetime import datetime
import numpy
import calculate_trends
import matplotlib.pyplot as plt
from states import state_historic_data, state_info

def plot_states_trend(states, data_name='positive', trendline=True, logarithmic=True, pop_adjusted=False, days=0, filename=None):
    index = 0
    si = state_info()
    for state in states:
        data_handle = state_historic_data(state)
        if days > 0:
            data_points = data_handle.get_latest_n(days)
        else:
            data_points = data_handle.get_after_n_cases(1)

        dates = list(map(lambda x: datetime.strptime(str(x['date']),"%Y%m%d").date(), data_points))
        cases = list(map(lambda x: x[data_name], data_points))

        if pop_adjusted:
            population = si.get_population(state)
            cases = list(map(lambda x: x/population * 10000, cases))

        (r,a) = calculate_trends.weighted_exponential_fit(numpy.arange(len(cases)), cases)
        if trendline:
            fit_y = numpy.exp(a) * numpy.exp(r * numpy.arange(len(cases)))
            plt.plot(dates, fit_y, "C"+str(index)+"--")
            plt.plot(dates, cases, "C"+str(index)+"o", label=state + " (rate=%0.4f)"%r)
        else:
            plt.plot(dates, cases, "C"+str(index)+".-", label=state + " (rate=%0.4f)"%r)
        index += 1

    if logarithmic:
        ax = plt.gca()
        ax.set_yscale("log")
    plt.legend()
    title = "Trend in "  + data_name + " counts, "
    if days == 0:
        title += "All days"
    else:
        title += "Last " + str(days) + " days"
    if pop_adjusted:
        title += " (population adjusted)"
    plt.title(title)
    ylabel = data_name + " count"
    if pop_adjusted:
        ylabel += " (per 10,000 people)"
    plt.ylabel(ylabel)
    fig = plt.gcf()
    fig.set_size_inches(10, 7)
    plt.grid(b=True)
    if filename is None:
        plt.show()
    else:
        fig.savefig("output/" + filename, dpi=100)

def plot_states_growth(states, data_name="positive", logarithmic=True, threshold=100, filename=None):
    index = 0
    si = state_info()
    for state in states:
        data_handle = state_historic_data(state)
        data_points = data_handle.get_after_n_cases(threshold)

        dates = range(len(data_points))
        cases = list(map(lambda x: x[data_name], data_points))

        plt.plot(dates, cases, "C"+str(index)+".-", label=state)
        index += 1

    for double_time in [2,3,5,10]:
        dates = numpy.arange(0,20,double_time)
        cases = (2**(dates/double_time)) * threshold
        plt.plot(dates, cases, label="doubling every " + str(double_time) + " days", color='#999999', linestyle="--")

    if logarithmic:
        ax = plt.gca()
        ax.set_yscale("log")
    plt.legend()
    plt.title("Growth rate since 100 cases")
    plt.ylabel("Cases")
    plt.xlabel("Days since 100 cases")
    fig = plt.gcf()
    fig.set_size_inches(10, 7)
    plt.grid(b=True)
    if filename is None:
        plt.show()
    else:
        fig.savefig("output/" + filename, dpi=100)

def plot_pos_test_rate(states, threshold=.10, filename=None):
    index = 0
    si = state_info()
    for state in states:
        data_handle = state_historic_data(state)
        data_points = data_handle.get_after_n_cases(threshold)

        dates = list(map(lambda x: datetime.strptime(str(x['date']),"%Y%m%d").date(), data_points))
        positive = numpy.array(list((map(lambda x: 0 if ('positive' not in x or x['positive'] is None) else x['positive'], data_points))))
        total = numpy.array(list((map(lambda x: 0 if ('total' not in x or x['total'] is None) else x['total'], data_points))))
        pos_test_rate = positive/total
        plt.plot(dates, pos_test_rate, "C"+str(index)+".-", label=state)
        index += 1
    plt.legend()
    plt.title("Positive test rate over time")
    plt.ylabel("Proportion of tests reported positive")
    fig = plt.gcf()
    fig.set_size_inches(10, 7)
    plt.grid(b=True, which="both")
    if filename is None:
        plt.show()
    else:
        fig.savefig("output/" + filename, dpi=100)


if __name__ == "__main__":
    states = ['KY',"TN","NY","IN","LA","OH"]
    plot_states_trend(states, data_name="positive", trendline=False, logarithmic=True, pop_adjusted=True, days=0)
    plot_states_trend(states, data_name="positive", trendline=True, logarithmic=True, pop_adjusted=False, days=10)
    plot_states_growth(states, data_name="positive", logarithmic=True, threshold=100)
    plot_pos_test_rate(states)
