from scipy.integrate import odeint as ode
import matplotlib.pyplot as plt
import numpy as np

# parametros do sistema
g = 9.8
l = 1.0
W = np.sqrt(g/l)
z = 0.0

# condicoes iniciais
theta0 = np.pi - 0.01
thetadot0 = 0.0
y0 = [theta0,thetadot0]

# funcao pendulo
def pendulo(y, t, par):
  W = par[0]
  z = par[1]
  dy = [y[1], - W * W * np.sin(y[0]) - 2 * z * W * y[1]]
  return dy

# parametros do tempo
ti = 0.0
tf = 10.0
h = 0.01
t = np.arange(ti, tf, h)

# integrador da funcao pendulo
y = ode(pendulo, y0, t, args=([W, z],))

# graficos
plt.figure(1)
plt.plot(t, y[:,0])
plt.title("Serie temporal da posicao angular")
plt.xlabel("tempo (s)")
plt.ylabel("posicao angular (rad)")
plt.grid()
plt.show()

plt.figure(2)
plt.plot(t, y[:,1])
plt.title("Serie temporal da velocidade angular")
plt.xlabel("tempo (s)")
plt.ylabel("velocidade angular (rad/s)")
plt.grid()
plt.show()

plt.figure(3)
plt.plot(y[:,0], y[:,1])
plt.title("Espaco de fase")
plt.xlabel("posicao angular (rad)")
plt.ylabel("velocidade angular (rad/s)")
plt.grid()
plt.show()
