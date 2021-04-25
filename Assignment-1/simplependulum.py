from scipy.integrate import odeint as ode
import matplotlib.pyplot as plt
import numpy as np
import math

# constants
R = 0.25         # m
m = 0.1          # kg
mu = 0.096581    # kg/m
K = 10           # N/m
K_3 =1000        # N/m^3
C = 0.2          # N/(m/s)
T0 = 100         # Nm
B = 5            # Nm/(rad/s)
beta = 0.60482   # s^-1
def T(t):
    return T0*math.tanh(beta*t)

# initial conditions
x_0          = 0.12
x_0_dot      = 0
theta_0      = 0
theta_0_dot  = 0
y_0 = [x_0, x_0_dot, theta_0, theta_0_dot]

def dy(y, t, par):
    C = par[0]
    K = par[1]
    K_3 = par[2]
    m = par[3]
    B = par[4]
    mu = par[5]
    R = par[6]
    tau = par[7](t)

    x=y[0]
    x_dot = y[1]
    theta = y[2]
    theta_dot = y[3]
    x_dot_dot = -1*(C*x_dot + x*x*x*K_3+x*K-x*theta_dot*theta_dot*m)/m
    theta_dot_dot = (tau -2*B*theta_dot-2*m*x*x_dot*theta_dot)/(m*x*x+mu*R*R*R*(math.pi+2/3))
    return [x_dot, x_dot_dot, theta_dot, theta_dot_dot]


# time parameters
ti = 0.0
tf = 10.0
h = 0.01
t = np.arange(ti, tf, h)

# # integrador da funcao pendulo
y = ode(dy, y_0, t, args=([C, K, K_3, m, B, mu, R, T],))

initialConds = '\nB='+str(B)+', x0='+str(x_0)+', v0='+str(x_0_dot)+', theta0='+str(theta_0)+', w0='+str(theta_0_dot)
# graphs
plt.figure(1)
plt.plot(t, y[:,0])
plt.title("Time series of x coordinate"+initialConds)
plt.xlabel("time (s)")
plt.ylabel("x (m)")
plt.grid()
plt.show()

plt.figure(2)
plt.plot(t, y[:,1])
plt.title("Time series of x speed"+initialConds)
plt.xlabel("time (s)")
plt.ylabel("dot(d) (m/s)")
plt.grid()
plt.show()

plt.figure(3)
plt.plot(t, y[:,2])
plt.title("Time series of theta"+initialConds)
plt.xlabel("time (s)")
plt.ylabel("theta (rad)")
plt.grid()
plt.show()

plt.figure(4)
plt.plot(t, y[:,3])
plt.title("Time series of angular speed"+initialConds)
plt.xlabel("time (s)")
plt.ylabel("dot(theta) (rad/s)")
plt.grid()
plt.show()
