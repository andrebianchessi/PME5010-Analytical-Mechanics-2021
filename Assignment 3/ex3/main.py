from scipy.integrate import odeint as ode
import matplotlib.pyplot as plt
import numpy as np
import math

def Sqrt(x):
    return math.sqrt(x)
def Cos(x):
    return math.cos(x)
def Sin(x):
    return math.sin(x)

def main(case):
    # time parameters
    ti = 0.0
    tf = 2

    # constants
    l = 10
    g = 9.8

    # initial conditions
    theta0 = math.pi/12
    thetaDot0 = 0
    phi0 = 0
    phiDot0 = 0

    if case == 2:
        phiDot0 = 0.1

    def ThetaDotDot(theta, phiDot):
        sTheta = math.sin(theta)
        cTheta = math.cos(theta)
        return sTheta*cTheta*phiDot*phiDot - g*sTheta/l
    
    def PhiDotDot(theta, thetaDot, phiDot):
        sTheta = math.sin(theta)
        cTheta = math.cos(theta)
        return -2*thetaDot*phiDot*cTheta/sTheta
    
    def XDotDot(x,y,z):
        return x*(-1-g*z/(l*l))
    def YDotDot(x,y,z):
        return y*(-g*z/(l*l)-1)
    def ZDotDot(x,y,z):
        return -z+g*(x*x+y*y)/(l*l)

    def Xf(theta, phi):
        return l*math.sin(theta)*math.sin(phi)
    def Yf(theta, phi):
        return l*math.sin(theta)*math.cos(phi)
    def Zf(theta, phi):
        return -l*math.cos(theta)

    x0 = Xf(theta0, phi0)
    y0 = Yf(theta0, phi0)
    z0 = Zf(theta0, phi0)

    xDot0 = l*(Cos(theta0)*Sin(phi0)*thetaDot0 + Sin(theta0)*Cos(phi0)*phiDot0)
    yDot0 = l*(Cos(theta0)*Cos(phi0)*thetaDot0 - Sin(theta0)*Sin(phi0)*phiDot0)
    zDot0 = l*Sin(theta0)*thetaDot0

    u0 = [theta0, thetaDot0, phi0, phiDot0, x0, xDot0, y0, yDot0, z0, zDot0]

    def du(u, t, par):
        theta = u[0]
        thetaDot = u[1]
        phi = u[2]
        phiDot = u[3]
        x = u[4]
        xDot = u[5]
        y = u[6]
        yDot = u[7]
        z = u[8]
        zDot = u[9]
        return [
            thetaDot,
            ThetaDotDot(theta, phiDot),
            phiDot,
            PhiDotDot(theta, thetaDot, phiDot),
            xDot,
            XDotDot(x,y,z),
            yDot,
            YDotDot(x,y,z),
            zDot,
            ZDotDot(x,y,z)
        ]

    # time step
    h = 0.01
    t = np.arange(ti, tf, h)

    u = ode(du, u0, t, args=([],))

    initConds = "theta0=" + str(round(theta0/math.pi,2)) + "*pi, thetaDot0="+str(thetaDot0) + ", phiDot0="+str(phiDot0)
    # graphs
    plt.figure()
    plt.plot(u[:,0], u[:,1])
    plt.title("Phase space\n"+initConds)
    plt.xlabel("theta")
    plt.ylabel("thetaDot")
    plt.grid()
    plt.savefig("plots/ex3ThetaCase"+str(case)+".png")

    plt.figure()
    if (phiDot0 != 0):
        plt.plot(u[:,2], u[:,3])
    else:
        plt.scatter(u[:,2], u[:,3])
    plt.title("Phase space\n"+initConds)
    plt.xlabel("phi")
    plt.ylabel("phiDot")
    plt.grid()
    plt.savefig("plots/ex3PhiCase"+str(case)+".png")

    # Plot x,y,z
    X = []
    Y = []
    Z = []

    for i in range(len(u[:,0])):
        theta = u[i,0]
        phi = u[i,2]
        X.append(Xf(theta,phi))
        Y.append(Yf(theta,phi))
        Z.append(Zf(theta,phi))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', title = "3D Plot of Motion\n"+initConds)
    ax.plot(np.array(X),np.array(Y),np.array(Z))
    ax.plot(u[:,4],u[:,6], u[:,8])
    ax.set_yscale('linear')
    ax.set_xscale('linear')
    ax.set_zscale('linear')
    plt.savefig('plots/ex3Case'+str(case)+'_3d.png')
    plt.show()




main(1)