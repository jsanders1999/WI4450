import numpy as np
import matplotlib.pyplot as plt


x1 = np.logspace(-8, 3, 12, base = 2)
x2 = np.logspace(3, 6, 4, base = 2)
x = np.append(x1, x2)
print(x)

y1 = np.logspace(0, 12, 12, base = 2)
y2 = 4096*np.ones(4)
y = np.append(y1, y2)
print(y)

R_max_data = [6.998, 18.94, 13.1, 3.33, 13.04, 15.28, 13.08]
I_data = [0.2375, 0.2500, 0.1875, 0.7500, 0.1250, 0.1875, 0.1875]
labels = ["total", "rho=<r,r>", "p=r+alpha*p", "q=op*p", "beta=<p,q>", "x=x+alpha*p", "r=r-alpha*q"]



fig, ax = plt.subplots()
ax.plot(x,y)
ax.scatter(I_data, R_max_data)
for i, txt in enumerate(labels):
    if i ==4:
        ax.annotate(txt, (I_data[i], R_max_data[i]), xytext= (I_data[i]*0.35, R_max_data[i]), fontsize=5)
    elif i ==2:
        ax.annotate(txt, (I_data[i], R_max_data[i]), xytext= (I_data[i]*1.15, R_max_data[i]*0.85), fontsize=5)
    else:
        ax.annotate(txt, (I_data[i], R_max_data[i]), xytext= (I_data[i]*1.15, R_max_data[i]), fontsize=5)
ax.set_xscale("log", base=2)
ax.set_yscale("log", base=2)
ax.set_xlabel("I [Flops/Byte]")
ax.set_ylabel("R_max [GFlops/s]")
plt.show()