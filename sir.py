from scipy.integrate import odeint
import numpy
from matplotlib import pyplot as mpl
from states import three_day_average, state_historic_data, state_info
import datetime


def sir_model(y,t,N,beta,gamma):
    S,I,R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


def sir_integrate(init, time, params):
    Sinit, Iinit, Rinit = init
    population = Sinit + Iinit + Rinit
    beta, gamma = params
    model = odeint(sir_model, init, numpy.arange(time), args=(population, beta, gamma))
    return model.T


def model_error(real_data, model_data):
    if len(real_data) < len(model_data):
        delta = numpy.array(real_data) - numpy.array(model_data[:len(real_data)])
    else:
        delta = numpy.array(real_data[:len(model_data)]) - numpy.array(model_data)
    return numpy.sum(delta**2)/100000


def sir_gradient_descent(model_init, model_time, initial_params, real_data):
    Sinit, Iinit, Rinit = model_init
    b,g = initial_params
    b_learning_rate = 0.001
    b_threshold = 0.000001
    S_learning_rate = 10000
    S_threshold = 1
    iteration = 0
    S, I, R = sir_integrate((Sinit, Iinit, Rinit), model_time, (b,g))
    error = model_error(real_data, I+R)
    while (b_learning_rate > b_threshold or S_learning_rate > S_threshold) and iteration < 10000:
        print("iteration {i:d}: beta={b:f}, Sinit={S:f} => error={e:f}".format(i=iteration, b=b, S=Sinit, e=error))
        iteration = iteration + 1

        # search cardinal directions on the b, S axis to find the next step
        S1,I1,R1 = sir_integrate((Sinit, Iinit, Rinit), model_time, (b + (b_learning_rate * error), g))
        error1 = model_error(real_data, I1+R1)
        grad1 = (error1 - error)/b_learning_rate

        # S2,I2,R2 = sir_integrate((Sinit + (S_learning_rate * error), Iinit, Rinit), model_time, (b, g))
        # error2 = model_error(real_data, I2)
        # grad2 = (error2 - error)/S_learning_rate

        S3,I3,R3 = sir_integrate((Sinit, Iinit, Rinit), model_time, (b - (b_learning_rate * error), g))
        error3 = model_error(real_data, I3+R3)
        grad3 = (error3 - error)/b_learning_rate

        # S4,I4,R4 = sir_integrate((Sinit - (S_learning_rate * error), Iinit, Rinit), model_time, (b, g))
        # error4 = model_error(real_data, I4)
        # grad4 = (error4 - error)/S_learning_rate

        # whichever one has the lowest, move that direction unless it causes beta or S to be < 0.
        # if no moves are available, halve the learning rate.
        # min_grad = numpy.min([grad1,grad2,grad3,grad4])
        min_grad = numpy.min([grad1,grad3])
        # if the other points are all worse halve the learning rate
        if min_grad > 0:
            b_learning_rate = b_learning_rate / 2
            S_learning_rate = S_learning_rate / 2
            print("iteration {i:d}: learning rate decreased to {blr:f}|{Slr:f}".format(i=iteration, blr=b_learning_rate, Slr=S_learning_rate))
        elif min_grad == grad1 and b + (b_learning_rate * error) > 0:
            b = b + (b_learning_rate * error)
            error = error1
            S, I, R = S1, I1, R1
        # elif min_grad == grad2 and Sinit + (S_learning_rate * error) > 0:
        #     Sinit = Sinit + (S_learning_rate * error)
        #     error = error2
        #     S, I, R = S2, I2, R2
        elif min_grad == grad3 and b - (b_learning_rate * error) > 0:
            b = b - (b_learning_rate * error)
            error = error3
            S, I, R = S3, I3, R3
        # elif min_grad == grad4 and Sinit - (S_learning_rate * error) > 0:
        #     Sinit = Sinit - (S_learning_rate * error)
        #     error = error4
        #     S, I, R = S4, I4, R4
        else:
            b_learning_rate = b_learning_rate / 2
            S_learning_rate = S_learning_rate / 2
            print("iteration {i:d}: learning rate decreased to {blr:f}|{Slr:f}".format(i=iteration, blr=b_learning_rate, Slr=S_learning_rate))
    print("iteration {i:d}: beta={b:f}, Sinit={S:f} => error={e:f}".format(i=iteration, b=b, S=Sinit, e=error))
    return b,Sinit,S,I,R


def sir_fit_data(dates, data, s_init, i_init, r_init, moving_average=False, additional_label_text="", plot_color="b.-", time=100):
    bootstrap_data = data
    if moving_average:
        bootstrap_data = three_day_average(bootstrap_data)
    b,Sinit,S,I,R = sir_gradient_descent((s_init, i_init, r_init), time, (0.2,0.15), bootstrap_data)
    mpl.plot(dates, I+R, plot_color + '.-', label="Predicted Cumulative Infections"+additional_label_text)
    mpl.plot(dates, I, plot_color + ',-', label="Predicted Active Infections"+additional_label_text)
    mpl.legend()
    fig = mpl.gcf()
    fig.set_size_inches(10, 7)
    return b,Sinit,S,I,R


def projection_cumulative_cases(population, cases, deaths, recovered, bootstrap_date, forecast_time=150, social_distancing_factor=0.9, logarithmic=False):
    Sinit = population - cases[0] - recovered[0]
    Iinit = cases[0] - deaths[0]
    Rinit = recovered[0] + deaths[0]

    date_list = [bootstrap_date + datetime.timedelta(days=x) for x in range(forecast_time)]
    mpl.plot(date_list[:len(cases)], cases, 'ro', label="Reported Infections")

    b,Sinit,S,I,R = sir_fit_data(date_list, cases, Sinit*(1-social_distancing_factor), Iinit, Rinit, time=forecast_time, additional_label_text=", " + str(social_distancing_factor*100) + "% social distancing", plot_color="C0")
    mpl.title("KY Predicted infections over time \nUsing SIR model, β=%0.2f, γ=0.15"%b)
    mpl.grid(b=True)
    if logarithmic:
        mpl.gca().set_yscale("log")
    mpl.show()
    mpl.close()


def projection_undertesting_cases(population, cases, deaths, recovered, bootstrap_date, forecast_time=150, social_distancing_factor=0.9, testing_coverage=0.2, logarithmic=False):
    cumulative_infected = numpy.array(cases)/testing_coverage
    Sinit = population - cumulative_infected[0] - (recovered[0]/testing_coverage)
    Iinit = cumulative_infected[0] - deaths[0]
    Rinit = recovered[0] + deaths[0]

    date_list = [bootstrap_date + datetime.timedelta(days=x) for x in range(forecast_time)]
    mpl.plot(date_list[:len(cumulative_infected)], cumulative_infected, 'ro', label="Estimated Infections")
    mpl.plot(date_list[:len(cases)], cases, 'g-', label="Reported Infections")

    b,Sinit,S,I,R = sir_fit_data(date_list, cumulative_infected, Sinit*(1-social_distancing_factor), Iinit, Rinit, time=forecast_time, additional_label_text=", " + str(social_distancing_factor*100) + "% social distancing", plot_color="C0")
    mpl.title("KY Predicted infections over time assuming %d%% testing coverage\nUsing SIR model, β=%0.2f, γ=0.15"%(testing_coverage*100,b))
    mpl.grid(b=True)
    if logarithmic:
        mpl.gca().set_yscale("log")
    mpl.show()
    mpl.close()


def projection_deaths(population, deaths, recovered, bootstrap_date, forecast_time=150, social_distancing_factor=0.9, mortality_rate=0.05, logarithmic=False):
    cumulative_infected = numpy.array(deaths)/mortality_rate
    Sinit = population - cumulative_infected[0] - recovered[0]
    Iinit = cumulative_infected[0] - deaths[0]
    Rinit = recovered[0] + deaths[0]

    date_list = [bootstrap_date + datetime.timedelta(days=x) for x in range(forecast_time)]
    mpl.plot(date_list[:len(cumulative_infected)], cumulative_infected, 'ro', label="Estimated Infections")
    mpl.plot(date_list[:len(deaths)], deaths, 'g-', label="Reported Deaths")

    b,Sinit,S,I,R=sir_fit_data(date_list, cumulative_infected, Sinit*(1-social_distancing_factor), Iinit, Rinit, time=forecast_time, additional_label_text=", " + str(social_distancing_factor*100) + "% social distancing", plot_color="C0")
    mpl.title("KY Predicted infections over time based on death count, assuming %.1f%% mortality\nUsing SIR model, β=%0.2f, γ=0.15"%(mortality_rate*100,b))
    mpl.grid(b=True)
    if logarithmic:
        mpl.gca().set_yscale("log")
    mpl.show()
    mpl.close()

if __name__ == "__main__":
    state = "KY"
    si = state_info()
    pop = si.get_population(state)
    data = state_historic_data(state).get_latest_n(45)
    bootstrap_date = datetime.datetime.strptime(str(data[0]['date']),'%Y%m%d').date()
    positive = list(map(lambda x: 0 if 'positive' not in x or x['positive'] is None else x['positive'], data))
    death = list(map(lambda x: 0 if 'death' not in x or x['death'] is None else x['death'], data))
    recovered = list(map(lambda x: 0 if 'recovered' not in x or x['recovered'] is None else x['recovered'], data))

    projection_cumulative_cases(pop, positive, death, recovered, bootstrap_date, social_distancing_factor=0.98)
    projection_undertesting_cases(pop, positive, death, recovered, bootstrap_date, social_distancing_factor=0.985, testing_coverage=0.1)
    projection_undertesting_cases(pop, positive, death, recovered, bootstrap_date, social_distancing_factor=0.98, testing_coverage=0.1)
    projection_deaths(pop, death, recovered, bootstrap_date, social_distancing_factor=0.98, mortality_rate=0.02)