from scipy.integrate import odeint as ode
import matplotlib.pyplot as plt
import numpy as np
import math

def main(simulationCase):
    # time parameters
    ti = 0.0
    tf = 60
    h = 0.1

    # constants
    mgzg = 0.2           # Nm
    ZG = 1               # m
    I = 1                # kgm^2
    J = 2*I

    # initial conditions
    theta0 = math.pi/6
    thetaDot0 = 0        # rad/s
    phi0 = 0
    phiDot0 = 0
    psi0 = 0
    psiDot0 = 1          # rad/s

    if (simulationCase == 2):
        theta0 = math.pi/4
    if (simulationCase == 3):
        tf = 15
        theta0 = math.pi/4
        psiDot0 = 5
    if (simulationCase == 4):
        theta0 = math.pi/4
        psiDot0 = 0.5
    if (simulationCase == 5):
        theta0 = math.pi/4
        psiDot0 = 0.3

    y_0 = [theta0, thetaDot0, phi0, psi0]
    # constants of motion
    beta = J*(phiDot0*math.cos(theta0)+psiDot0)
    alpha = I*math.sin(theta0)*math.sin(theta0)*phiDot0 + math.cos(theta0)*beta

    def ThetaDotDot(theta):
        c = math.cos(theta)
        s = math.sin(theta)
        return (mgzg*s - (alpha-beta*c)*(beta-alpha*c)/(I*s*s*s))/I
    def PhiDot(theta):
        return (alpha-math.cos(theta)*beta)/(I*math.sin(theta)*math.sin(theta))
    def PsiDot(theta):
        phiD = PhiDot(theta)
        return beta/J-phiD*math.cos(theta)

    def dy(y, t, par):
        theta = y[0]
        thetaDot = y[1]
        phi = y[2]
        psi = y[3]

        thetaDotDot = ThetaDotDot(theta)
        phiDot = PhiDot(theta)
        psiDot = PsiDot(theta)
        return [thetaDot, thetaDotDot, phiDot, psiDot]


    t = np.arange(ti, tf, h)
    y = ode(dy, y_0, t, args=([],))

    phiDotList = []
    psiDotList = []
    for i in y[:,0]:
        phiDotList.append(PhiDot(i))
        psiDotList.append(PsiDot(i))

    def Xg(theta, phi):
        return ZG*math.sin(theta)*math.cos(phi)
    def Yg(theta, phi):
        return ZG*math.sin(theta)*math.sin(phi)
    def Zg(theta, phi):
        return ZG*math.cos(theta)
    xg = []
    yg = []
    zg = []
    for i in range(len(y[:,0])):
        theta = y[i,0]
        phi = y[i,2]
        xg.append(Xg(theta,phi))
        yg.append(Yg(theta,phi))
        zg.append(Zg(theta,phi))

    # Plots
    initialConds = '\ntheta0='+str(round(theta0/math.pi,3)) + '*pi, thetaDot0='+str(thetaDot0) + ', phiDot0='+str(phiDot0)+', psiDot0='+str(psiDot0)

    plt.figure()
    plt.plot(t, y[:,0])
    plt.title("Time series of theta"+initialConds)
    plt.xlabel("time (s)")
    plt.ylabel("theta (rad)")
    plt.grid()
    # plt.show()
    plt.savefig('plots/case'+str(simulationCase)+'theta.png')

    plt.figure()
    plt.plot(t, y[:,1])
    plt.title("Time series of thetaDot"+initialConds)
    plt.xlabel("time (s)")
    plt.ylabel("thetaDot (rad/s)")
    plt.grid()
    # plt.show()
    plt.savefig('plots/case'+str(simulationCase)+'thetaDot.png')


    plt.figure()
    plt.plot(t, y[:,2])
    plt.title("Time series of phi"+initialConds)
    plt.xlabel("time (s)")
    plt.ylabel("phi (rad)")
    plt.grid()
    plt.savefig('plots/case'+str(simulationCase)+'phi.png')

    plt.figure()
    plt.plot(t, phiDotList)
    plt.title("Time series of phiDot"+initialConds)
    plt.xlabel("time (s)")
    plt.ylabel("phiDot (rad/s)")
    plt.grid()
    plt.savefig('plots/case'+str(simulationCase)+'phiDot.png')

    plt.figure()
    plt.plot(t, y[:,3])
    plt.title("Time series of psi"+initialConds)
    plt.xlabel("time (s)")
    plt.ylabel("psi (rad)")
    plt.grid()
    plt.savefig('plots/case'+str(simulationCase)+'psi.png')

    plt.figure()
    plt.plot(t, psiDotList)
    plt.title("Time series of psiDot"+initialConds)
    plt.xlabel("time (s)")
    plt.ylabel("psiDot (rad/s)")
    plt.grid()
    plt.savefig('plots/case'+str(simulationCase)+'psiDot.png')

    plt.figure()
    plt.plot(t, xg)
    plt.title("Time series of XG coordinate"+initialConds)
    plt.xlabel("time (s)")
    plt.ylabel("XG (m)")
    plt.grid()
    plt.savefig('plots/case'+str(simulationCase)+'XG.png')

    plt.figure()
    plt.plot(t, yg)
    plt.title("Time series of YG coordinate"+initialConds)
    plt.xlabel("time (s)")
    plt.ylabel("YG (m)")
    plt.grid()
    plt.savefig('plots/case'+str(simulationCase)+'YG.png')

    plt.figure()
    plt.plot(t, zg)
    plt.title("Time series of ZG coordinate"+initialConds)
    plt.xlabel("time (s)")
    plt.ylabel("ZG (m)")
    plt.grid()
    plt.savefig('plots/case'+str(simulationCase)+'ZG.png')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', title = "Time series of center of mass "+initialConds)
    ax.plot(np.array(xg),np.array(yg),np.array(zg))
    ax.set_yscale('linear')
    ax.set_xscale('linear')
    ax.set_zscale('linear')
    plt.savefig('plots/case'+str(simulationCase)+'G_3d.png')\

main(1)
main(2)
main(3)
main(4)
main(5)