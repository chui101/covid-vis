from scipy.integrate import odeint
import numpy

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
    b,g = initial_params
    learning_rate = 0.0001
    threshold = 0.000001
    iteration = 0
    S, I, R = sir_integrate(model_init, model_time, (b,g))
    error = model_error(real_data, I)
    while learning_rate > threshold and iteration < 1000:
        print("iteration {i:d}: beta={b:f}, gamma={g:f} => error={e:f}".format(i=iteration, b=b, g=g, e=error))

        iteration = iteration + 1

        # search E, N, W, S on the b, g axis to find the next step
        S1,I1,R1 = sir_integrate(model_init, model_time, (b + (learning_rate * error), g))
        error1 = model_error(real_data, I1)
        grad1 = error1 - error

        # S2,I2,R2 = sir_integrate(model_init, model_time, (b, g + (learning_rate * error)))
        # error2 = model_error(real_data, I2)
        # grad2 = error2 - error

        S3,I3,R3 = sir_integrate(model_init, model_time, (b - (learning_rate * error), g))
        error3 = model_error(real_data, I3)
        grad3 = error3 - error

        # S4,I4,R4 = sir_integrate(model_init, model_time, (b, g - (learning_rate * error)))
        # error4 = model_error(real_data, I4)
        # grad4 = error4 - error

        # whichever one has the lowest, move that direction unless it causes beta or gamma to be < 0.
        # if no moves are available, halve the learning rate.
        # min_grad = numpy.min([grad1,grad2,grad3,grad4])
        min_grad = numpy.min([grad1,grad3])
        # if the other points are all worse halve the learning rate
        if min_grad > 0:
            learning_rate = learning_rate / 2
            print("iteration {i:d}: learning rate decreased to {lr:f}".format(i=iteration, lr=learning_rate))
        elif min_grad == grad1 and b + (learning_rate * error) > 0:
            b = b + (learning_rate * error)
            error = error1
            S, I, R = S1, I1, R1
        # elif min_grad == grad2 and g + (learning_rate * error) > 0:
        #     g = g + (learning_rate * error)
        #     error = error2
        #     S, I, R = S2, I2, R2
        elif min_grad == grad3 and b - (learning_rate * error) > 0:
            b = b - (learning_rate * error)
            error = error3
            S, I, R = S3, I3, R3
        # elif min_grad == grad4 and g - (learning_rate * error) > 0:
        #     g = g - (learning_rate * error)
        #     error = error4
        #     S, I, R = S4, I4, R4
        else:
            learning_rate = learning_rate / 2
            print("iteration {i:d}: learning rate decreased to {lr:f}".format(i=iteration, lr=learning_rate))
    print("iteration {i:d}: beta={b:f}, gamma={g:f} => error={e:f}".format(i=iteration, b=b, g=g, e=error))
    return b,g,S,I,R

if __name__ == "__main__":
    init = (4467572,99,2)
    ky_pos = [99, 104, 124, 157, 198, 248, 302, 394, 439, 480, 591, 680, 770]
    b,g,S,I,R = sir_gradient_descent(init, 100, (0.5,0.1), ky_pos)
    print(numpy.max(I))
