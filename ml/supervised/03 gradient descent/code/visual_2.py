import numpy as np
import matplotlib.pyplot as plt

def y_fn(x):
    return np.sin(x)

def y_drv(x):
    return np.cos(x)

stop_animation = False
def on_key(event):
    global stop_animation
    if event.key == 'q':
        stop_animation = True

x = np.arange(-5, 5, 0.01)
y = y_fn(x)

fig, ax = plt.subplots()
fig.canvas.mpl_connect('key_press_event', on_key)

x0 = 2
current_pos = x0, y_fn(x0)

learning_rate = 0.01

for _ in range(1000):
    if stop_animation: break

    new_x = current_pos[0] - learning_rate * y_drv(current_pos[0])
    new_y = y_fn(new_x)
    current_pos = new_x, new_y

    ax.clear()
    ax.plot(x, y)
    ax.scatter(new_x, new_y, color='red')
    plt.pause(0.001)

plt.show()

