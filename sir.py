from scipy.integrate import odeint
import numpy
from matplotlib import pyplot as mpl
from states import three_day_average
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
    return numpy.sum(delta**2)/100

def sir_gradient_descent(model_init, model_time, initial_params, real_data):
    Sinit, Iinit, Rinit = model_init
    b,g = initial_params
    b_learning_rate = 0.0001
    b_threshold = 0.000001
    S_learning_rate = 10000
    S_threshold = 1
    iteration = 0
    S, I, R = sir_integrate((Sinit, Iinit, Rinit), model_time, (b,g))
    error = model_error(real_data, I)
    while (b_learning_rate > b_threshold or S_learning_rate > S_threshold) and iteration < 10000:
        print("iteration {i:d}: beta={b:f}, Sinit={S:f} => error={e:f}".format(i=iteration, b=b, S=Sinit, e=error))
        iteration = iteration + 1

        # search cardinal directions on the b, S axis to find the next step
        S1,I1,R1 = sir_integrate((Sinit, Iinit, Rinit), model_time, (b + (b_learning_rate * error), g))
        error1 = model_error(real_data, I1)
        grad1 = (error1 - error)/b_learning_rate

        # S2,I2,R2 = sir_integrate((Sinit + (S_learning_rate * error), Iinit, Rinit), model_time, (b, g))
        # error2 = model_error(real_data, I2)
        # grad2 = (error2 - error)/S_learning_rate

        S3,I3,R3 = sir_integrate((Sinit, Iinit, Rinit), model_time, (b - (b_learning_rate * error), g))
        error3 = model_error(real_data, I3)
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

def sir_fit_data(dates, data, s_init, i_init, r_init, moving_average=False, additional_label_text="", plot_format="b.=", time=100):
    bootstrap_data = data
    if moving_average:
        bootstrap_data = three_day_average(bootstrap_data)
    b,g,S,I,R = sir_gradient_descent((s_init, i_init, r_init), time, (0.3,0.15), bootstrap_data)
    mpl.plot(dates, I, plot_format, label="Predicted Infections"+additional_label_text)
    mpl.legend()
    fig = mpl.gcf()
    fig.set_size_inches(10, 7)


if __name__ == "__main__":
    Sinit, Iinit, Rinit =(4467572,99,2)
    ky_pos = [99,104,124,157,198,248,302,394,439,480,591,680,770,831,917,955,1008,1149,1346,1452]
    forecast_time = 100

    bootstrap_date = datetime.date(year=2020, month=3, day=22)
    date_list = [bootstrap_date + datetime.timedelta(days=x) for x in range(forecast_time)]
    mpl.plot(date_list[:len(ky_pos)], ky_pos, 'ro', label="Reported Infections")

    sir_fit_data(date_list, ky_pos, Sinit*.5, Iinit, Rinit, time=forecast_time, additional_label_text=", 50% social distancing", plot_format="C0.-")
    sir_fit_data(date_list, ky_pos, Sinit*.25, Iinit, Rinit, time=forecast_time, additional_label_text=", 75% social distancing", plot_format="C1.-")
    sir_fit_data(date_list, ky_pos, Sinit*.1, Iinit, Rinit, time=forecast_time, additional_label_text=", 90% social distancing", plot_format="C2.-")
    sir_fit_data(date_list, ky_pos, Sinit*.05, Iinit, Rinit, time=forecast_time, additional_label_text=", 95% social distancing", plot_format="C3.-")
    mpl.title("KY Predicted infections over time using SIR model, γ=0.15\nGradient descent on β to fit existing data")
    mpl.grid(b=True)
    mpl.show()
    mpl.close()