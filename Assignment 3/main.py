from scipy.integrate import odeint as ode
import matplotlib.pyplot as plt
import numpy as np
import math

def main(case):

    # constants
    b1 = 1
    b3 = 1

    # initial conditions
    x_0          = 1
    x_0_dot      = 0

    if case == 2:
        x_0 = 1.5

    if case == 3:
        x_0 = 2

    if case == 4:
        x_0 = 1.5
        b1 = -1

    if case == 5:
        x_0 = 2
        b1 = -1


    E = 1/2*x_0_dot*x_0_dot + 1/2*b1*x_0*x_0 + 1/4*b3*x_0*x_0*x_0*x_0

    def xDotDot(x):
        return -b1*x - b3*x*x*x

    y_0 = [x_0, x_0_dot]

    def dy(y, t, par):

        x=y[0]
        x_dot = y[1]
        x_dotDot = xDotDot(x)
        return [x_dot, x_dotDot]


    # time parameters
    ti = 0.0
    tf = 5
    h = 0.01
    t = np.arange(ti, tf, h)

    y = ode(dy, y_0, t, args=([],))

    # graphs
    plt.figure(1)
    plt.plot(y[:,0], y[:,1])
    plt.title("Phase space x\n"+ "E="+str(E) +", beta1="+str(b1) +", beta3="+str(b3))
    plt.xlabel("x")
    plt.ylabel("xDot")
    plt.grid()
    plt.savefig("plots/case"+str(case)+".png")


main(1)