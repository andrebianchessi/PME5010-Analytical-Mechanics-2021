from scipy.integrate import odeint as ode
import matplotlib.pyplot as plt
import numpy as np
import math

# time parameters
ti = 0.0
tf = 80.0
h = 0.1

# constants
mgzg = 0.2           # Nm
ZG = 1               # m
I = 1                # kgm^2
J = 2*I
thetaBar = math.pi/6 # rad   

# initial conditions
theta0 = thetaBar
thetaDot0 = 0        # rad/s
phi0 = 0
phiDot0 = 0
psi0 = 0
psiDot0 = 1         # rad/s
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
    phiD = PhiDot(alpha)
    return (beta-J*phiD*math.cos(theta))/J

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

# # Plots
# initialConds = '\ntheta0='+str(round(theta0,2)) + ', thetaDot0='+str(thetaDot0) + ', phiDot0='+str(phiDot0)+', psiDot0='+str(psiDot0)

# plt.figure(1)
# plt.plot(t, y[:,0])
# plt.title("Time series of theta coordinate"+initialConds)
# plt.xlabel("time (s)")
# plt.ylabel("theta (rad)")
# plt.grid()
# plt.show()


# plt.figure(2)
# plt.plot(t, y[:,2])
# plt.title("Time series of phi coordinate"+initialConds)
# plt.xlabel("time (s)")
# plt.ylabel("phi (rad)")
# plt.grid()
# plt.show()

# plt.figure(3)
# plt.plot(t, phiDotList)
# plt.title("Time series of phiDot coordinate"+initialConds)
# plt.xlabel("time (s)")
# plt.ylabel("phiDot (rad/s)")
# plt.grid()
# plt.show()

# plt.figure(4)
# plt.plot(t, y[:,3])
# plt.title("Time series of psi coordinate"+initialConds)
# plt.xlabel("time (s)")
# plt.ylabel("psi (rad)")
# plt.grid()
# plt.show()

# plt.figure(5)
# plt.plot(t, psiDotList)
# plt.title("Time series of psiDot coordinate"+initialConds)
# plt.xlabel("time (s)")
# plt.ylabel("psiDot (rad/s)")
# plt.grid()
# plt.show()

# plt.figure(6)
# plt.plot(t, xg)
# plt.title("Time series of XG coordinate"+initialConds)
# plt.xlabel("time (s)")
# plt.ylabel("XG (rad/s)")
# plt.grid()
# plt.show()

# plt.figure(7)
# plt.plot(t, yg)
# plt.title("Time series of YG coordinate"+initialConds)
# plt.xlabel("time (s)")
# plt.ylabel("YG (rad/s)")
# plt.grid()
# plt.show()

# plt.figure(8)
# plt.plot(t, zg)
# plt.title("Time series of ZG coordinate"+initialConds)
# plt.xlabel("time (s)")
# plt.ylabel("YG (rad/s)")
# plt.grid()
# plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(np.array(xg),np.array(yg),np.array(zg))
ax.set_yscale('linear')
ax.set_xscale('linear')
ax.set_zscale('linear')
plt.show()