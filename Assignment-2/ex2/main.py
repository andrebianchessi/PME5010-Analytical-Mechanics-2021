from scipy.integrate import odeint as ode
import matplotlib.pyplot as plt
import numpy as np
import math

def main(simulationCase):
    # time parameters
    ti = 0.0
    tf = 20
    h = 0.05

    # constants
    K = 3
    m = 0.2
    a = 1
    l = 0.8


    # initial conditions
    x0 = 0.1
    xDot0 = 0
    y0 = 0 
    yDot0 = 0

    if (simulationCase == 2):
        y0 = 0.1

    y_0 = [x0, xDot0, y0, yDot0]

    def XDotDot(x, y):
        sqrt1 = math.sqrt((a+y)*(a+y) + x*x)
        sqrt2 = math.sqrt((a-y)*(a-y) + x*x)
        return K*x/m * (
            l*(1/sqrt1+1/sqrt2)
            -2
        )
    def YDotDot(x, y):
        sqrt1 = math.sqrt((a-y)*(a-y) + x*x)
        sqrt2 = math.sqrt((a+y)*(a+y) + x*x)
        return K/m * (
            (a-y)*(sqrt1-l)/sqrt1
            - (a+y)*(sqrt2-l)/sqrt2
        )

    def dy(Y, t, par):
        x = Y[0]
        xDot = Y[1]
        y = Y[2]
        yDot = Y[3]

        xDotDot = XDotDot(x, y)
        yDotDot = YDotDot(x,y)
        return [xDot, xDotDot, yDot, yDotDot]

    t = np.arange(ti, tf, h)
    Y = ode(dy, y_0, t, args=([],))
    
    # Plots
    initialConds = '\nx0='+str(x0)+ ', xDot0='+str(xDot0) + ', y0='+str(y0)+ ', yDot0='+str(yDot0)+ ', m='+str(m)+ ', L='+str(l) + ', a='+str(a) + ', K='+str(K)

    plt.figure()
    plt.plot(t, Y[:,0])
    plt.title("Time series of x"+initialConds)
    plt.xlabel("time (s)")
    plt.ylabel("x (m)")
    plt.grid()
    plt.savefig('plots/ex2Case'+str(simulationCase)+'x.png')

    plt.figure()
    plt.plot(t, Y[:,1])
    plt.title("Time series of xDot"+initialConds)
    plt.xlabel("time (s)")
    plt.ylabel("xDot (m/s)")
    plt.grid()
    plt.savefig('plots/ex2Case'+str(simulationCase)+'xDot.png')

    plt.figure()
    plt.plot(t, Y[:,2])
    plt.title("Time series of y"+initialConds)
    plt.xlabel("time (s)")
    plt.ylabel("y (m)")
    plt.grid()
    plt.savefig('plots/ex2Case'+str(simulationCase)+'y.png')

    plt.figure()
    plt.plot(t, Y[:,3])
    plt.title("Time series of xDot"+initialConds)
    plt.xlabel("time (s)")
    plt.ylabel("yDot (m/s)")
    plt.grid()
    plt.savefig('plots/ex2Case'+str(simulationCase)+'yDot.png')

    plt.figure()
    plt.plot(Y[:,0], Y[:,2])
    plt.title("Time series of mass position"+initialConds)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.grid()
    plt.savefig('plots/ex2Case'+str(simulationCase)+'xy.png')



main(1)
main(2)