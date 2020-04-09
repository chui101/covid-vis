from scipy.integrate import odeint
import numpy
from matplotlib import pyplot as mpl

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
        delta = numpy.sqrt(numpy.array(real_data)) - numpy.sqrt(numpy.array(model_data[:len(real_data)]))
    else:
        delta = numpy.sqrt(numpy.array(real_data[:len(model_data)])) - numpy.sqrt(numpy.array(model_data))
    return numpy.sqrt(numpy.sum(delta**2) / len(delta))

def sir_gradient_descent(model_init, model_time, initial_params, real_data):
    Sinit, Iinit, Rinit = model_init
    b,g = initial_params
    b_learning_rate = 0.0001
    b_threshold = 0.000001
    S_learning_rate = 100000
    S_threshold = 10
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

        S2,I2,R2 = sir_integrate((Sinit + (S_learning_rate * error), Iinit, Rinit), model_time, (b, g))
        error2 = model_error(real_data, I2)
        grad2 = (error2 - error)/S_learning_rate

        S3,I3,R3 = sir_integrate((Sinit, Iinit, Rinit), model_time, (b - (b_learning_rate * error), g))
        error3 = model_error(real_data, I3)
        grad3 = (error3 - error)/b_learning_rate

        S4,I4,R4 = sir_integrate((Sinit - (S_learning_rate * error), Iinit, Rinit), model_time, (b, g))
        error4 = model_error(real_data, I4)
        grad4 = (error4 - error)/S_learning_rate

        # whichever one has the lowest, move that direction unless it causes beta or S to be < 0.
        # if no moves are available, halve the learning rate.
        min_grad = numpy.min([grad1,grad2,grad3,grad4])
        # if the other points are all worse halve the learning rate
        if min_grad > 0:
            b_learning_rate = b_learning_rate / 2
            S_learning_rate = S_learning_rate / 2
            print("iteration {i:d}: learning rate decreased to {blr:f}|{Slr:f}".format(i=iteration, blr=b_learning_rate, Slr=S_learning_rate))
        elif min_grad == grad1 and b + (b_learning_rate * error) > 0:
            b = b + (b_learning_rate * error)
            error = error1
            S, I, R = S1, I1, R1
        elif min_grad == grad2 and Sinit + (S_learning_rate * error) > 0:
            Sinit = Sinit + (S_learning_rate * error)
            error = error2
            S, I, R = S2, I2, R2
        elif min_grad == grad3 and b - (b_learning_rate * error) > 0:
            b = b - (b_learning_rate * error)
            error = error3
            S, I, R = S3, I3, R3
        elif min_grad == grad4 and Sinit - (S_learning_rate * error) > 0:
            Sinit = Sinit - (S_learning_rate * error)
            error = error4
            S, I, R = S4, I4, R4
        else:
            b_learning_rate = b_learning_rate / 2
            S_learning_rate = S_learning_rate / 2
            print("iteration {i:d}: learning rate decreased to {blr:f}|{Slr:f}".format(i=iteration, blr=b_learning_rate, Slr=S_learning_rate))
    print("iteration {i:d}: beta={b:f}, Sinit={S:f} => error={e:f}".format(i=iteration, b=b, S=Sinit, e=error))
    return b,Sinit,S,I,R

if __name__ == "__main__":
    init = (4467572,99,2)
    ky_pos = [99, 104, 124, 157, 198, 248, 302, 394, 439, 480, 591, 680, 770, 831, 917, 955, 1008, 1149]
    b,g,S,I,R = sir_gradient_descent(init, 100, (0.5,1/14), ky_pos)
    mpl.plot(range(len(ky_pos)), ky_pos, 'ro', label="Positive")
    mpl.plot(range(len(I)), I, 'b.-', label="Predicted")
    mpl.title("KY Predicted infections over time using SIR model")
    mpl.xlabel("Days since 100 cases")
    mpl.show()
    mpl.close()
