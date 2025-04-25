from matplotlib import pyplot as plt
import numpy as np

x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)
plt.figure()
plt.subplot(121)
plt.bar(y[:10],np.arange(10))
plt.subplot(122)
plt.plot(x+1, y, 'g--', lw=4, label='sin(x)')
plt.show()