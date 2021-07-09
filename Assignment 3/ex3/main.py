from scipy.integrate import odeint as ode
import matplotlib.pyplot as plt
import numpy as np
import math

def main(case):

    # constants
    l = 10
    m = 1
    g = 9.8

    # initial conditions
    theta0 = math.pi/6
    thetaDot0 = 0
    phi0 = 0
    phiDot0 = 0

    if case == 2:
        phiDot0 = 1

    def ThetaDotDot(theta, phiDot):
        sTheta = math.sin(theta)
        cTheta = math.cos(theta)
        return sTheta*cTheta*phiDot*phiDot - g*sTheta/l
    
    def PhiDotDot(theta, thetaDot, phiDot):
        sTheta = math.sin(theta)
        cTheta = math.cos(theta)
        return -2*thetaDot*phiDot*cTheta/sTheta

    y_0 = [theta0, thetaDot0, phi0, phiDot0]

    def dy(y, t, par):
        theta = y[0]
        thetaDot = y[1]
        phi = y[2]
        phiDot = y[3]
        return [thetaDot, ThetaDotDot(theta, phiDot), phiDot, PhiDotDot(theta, thetaDot, phiDot)]


    # time parameters
    ti = 0.0
    tf = 10
    h = 0.01
    t = np.arange(ti, tf, h)

    y = ode(dy, y_0, t, args=([],))

    initConds = "theta0=" + str(round(theta0/math.pi,2)) + "*pi, thetaDot0="+str(thetaDot0) + ", phiDot0="+str(phiDot0)
    # graphs
    plt.figure()
    plt.scatter(y[:,0], y[:,1])
    plt.title("Phase space\n"+initConds)
    plt.xlabel("theta")
    plt.ylabel("thetaDot")
    plt.grid()
    plt.savefig("plots/ex3ThetaCase"+str(case)+".png")

    plt.figure()
    plt.scatter(y[:,2], y[:,3])
    plt.title("Phase space\n"+initConds)
    plt.xlabel("phi")
    plt.ylabel("phiDot")
    plt.grid()
    plt.savefig("plots/ex3PhiCase"+str(case)+".png")

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d', title = "Time series of center of mass "+initialConds)
    # ax.plot(np.array(xg),np.array(yg),np.array(zg))
    # ax.set_yscale('linear')
    # ax.set_xscale('linear')
    # ax.set_zscale('linear')
    # plt.savefig('plots/case'+str(simulationCase)+'G_3d.png')\




main(1)
main(2)