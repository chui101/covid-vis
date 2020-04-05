from datetime import datetime
import numpy
import calculate_trends
import matplotlib.pyplot as plt
from states import state_historic_data, state_info

def plot_states(states, data_name='positive', trendline=True, logarithmic=True, pop_adjusted=False, days=0, filename=None):
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

        plt.plot(dates, cases, "C"+str(index)+"o", label=state)
        if trendline:
            (r,a) = calculate_trends.weighted_exponential_fit(numpy.arange(len(cases)), cases)
            fit_y = numpy.exp(a) * numpy.exp(r * numpy.arange(len(cases)))
            plt.plot(dates, fit_y, "C"+str(index)+"--", label=state + "(exponential, rate=%0.4f)"%r)
        index += 1

    if logarithmic:
        ax = plt.gca()
        ax.set_yscale("log")
    plt.legend()
    fig = plt.gcf()
    fig.set_size_inches(15, 10)
    if filename is None:
        plt.show()
    else:
        fig.savefig("output/" + filename, dpi=100)

if __name__ == "__main__":
    plot_states(['KY',"TN","NY","IN","LA","OH"], data_name="positive", trendline=True, logarithmic=True, pop_adjusted=True, days=10)