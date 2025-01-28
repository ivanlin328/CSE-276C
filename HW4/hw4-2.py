import numpy as np
import matplotlib.pyplot as plt

def lotka_volterra(x,b, p, r, d):
    
    x1,x2=x
    dx1_dt=(b - p*x2) * x1
    dx2_dt=(r * x1 - d) * x2
    
    return np.array([dx1_dt,dx2_dt])

def fourth_order_Runge_Kutta(fx,x0,h,t_span,params):  
    t_start,t_end=t_span
    t = np.arange(t_start, t_end + h, h)
    x = np.zeros((len(t), len(x0)))
    x[0] = x0
    # Implement fourth_order_Runge_Kutta
    for i in range(1, len(t)):
        k1 = h * fx(x[i-1], *params)
        k2 = h * fx(x[i-1] + 0.5*k1, *params)
        k3 = h * fx(x[i-1] + 0.5*k2, *params)
        k4 = h * fx(x[i-1] + k3, *params)
        
        x[i] = x[i-1] + (k1 + 2*k2 + 2*k3 + k4) / 6
    
    return t, x
        
x0=([0.3,0.2])
h=0.01
time_span=(0,20)
t,solution=fourth_order_Runge_Kutta(lotka_volterra,x0,h,time_span,(1,1,1,1))
print(solution)
plt.plot(t, solution[:, 0], label='Prey Population')
plt.plot(t, solution[:, 1], label='Predator Population')
plt.title('Lotka-Volterra Predator-Prey Dynamics')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.grid(True)
plt.show()


