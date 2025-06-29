#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent

#%%
def z_function(x, y):
    return np.sin(5*x) * np.cos(5*y) / 5


def z_derivative(x, y):
    return np.cos(5*x) * np.cos(5*y), -np.sin(5*y)*np.sin(5*x)

stop_animation = False
def on_key(event: KeyEvent):
    global stop_animation
    if event.key == 'q':
        stop_animation = True
# %%
x = np.arange(-1, 1, 0.05)
y = np.arange(-1, 1, 0.05)

X, Y = np.meshgrid(x, y)
Z = z_function(X, Y)

x0 = 0.7
y0 = 0.4
current_pos = x0, y0, z_function(x0, y0)
learning_rate = 0.01

#%%
ax = plt.subplot(projection='3d', computed_zorder=False)
plt.connect('key_press_event', on_key)

for _ in range(1000):
    if stop_animation:
        break

    X_derivative, Y_derivative = z_derivative(current_pos[0], current_pos[1])
    X_new = current_pos[0] - learning_rate * X_derivative
    Y_new = current_pos[1] - learning_rate * Y_derivative
    current_pos = X_new, Y_new, z_function(X_new, Y_new)

    ax.plot_surface(X, Y, Z, cmap='viridis', zorder=0)
    ax.scatter(current_pos[0], current_pos[1], current_pos[2], color='magenta', zorder=1)

    plt.pause(0.001)
    ax.clear()

print(current_pos)

# plt.show()
# %%
