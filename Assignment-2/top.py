from scipy.integrate import odeint as ode
import matplotlib.pyplot as plt
import numpy as np
import math

# constants
mgzg = 0.2           # Nm
I = 1                # kgm^2
J = 2*I
thetaBar = math.pi/6 # rad   

# initial conditions
theta0 = thetaBar
thetaDot0 = 0        # rad/s
phiDot0 = 0
psiDot0 = 1          # rad/s
y_0 = [theta0, thetaDot0]

# constants of motion
beta = J*(phiDot0*math.cos(theta0)+psiDot0)
alpha = I*math.sin(theta0)*math.sin(theta0)*beta

def dy(y, t, par):
    theta = y[0]
    thetaDot = y[1]

    c = math.cos(theta)
    s = math.sin(theta)
    thetaDotDot = (mgzg*c - (alpha-beta*c)*(beta-alpha*c)/(I*s*s*s))/I
    return [thetaDot, thetaDotDot]


# time parameters
ti = 0.0
tf = 5.0
h = 0.01
t = np.arange(ti, tf, h)

y = ode(dy, y_0, t, args=([],))

initialConds = '\ntheta0='+str(round(theta0,2)) + ', thetaDot0='+str(thetaDot0) + ', phiDot0='+str(phiDot0)+', psiDot0='+str(psiDot0)
# graphs
plt.figure(1)
plt.plot(t, y[:,0])
plt.title("Time series of theta coordinate"+initialConds)
plt.xlabel("time (s)")
plt.ylabel("theta (rad)")
plt.grid()
plt.show()

def phiDot(theta):
    return (alpha-math.cos(theta)*beta)/(I*math.sin(theta)*math.sin(theta))
def psiDot(theta):
    phiD = phiDot(alpha)
    return (beta-J*phiD*math.cos(theta))/J

phiDotList = []
psiDotList = []
for i in y[:,0]:
    phiDotList.append(phiDot(i))
    psiDotList.append(psiDot(i))

plt.figure(2)
plt.plot(t, phiDotList)
plt.title("Time series of phiDot coordinate"+initialConds)
plt.xlabel("time (s)")
plt.ylabel("phiDot (rad/s)")
plt.grid()
plt.show()

plt.figure(3)
plt.plot(t, psiDotList)
plt.title("Time series of psiDot coordinate"+initialConds)
plt.xlabel("time (s)")
plt.ylabel("psiDot (rad/s)")
plt.grid()
plt.show()