import numpy as np
import matplotlib.pyplot as plt

def y_function(x):
    return x ** 2

def y_derivative(x):
    return 2 * x


x = np.arange(-100, 100, 0.1)
y = y_function(x)

learning_rate = 0.01

current_pos = (80, y_function(80))

stop_animation = False
def on_key(event):
    global stop_animation
    if event.key == 'q':
        stop_animation = True

fig, ax = plt.subplots()
fig.canvas.mpl_connect('key_press_event', on_key)


for _ in range(500):
    if stop_animation:
        break

    new_x = current_pos[0] - learning_rate * y_derivative(current_pos[0])
    new_y = y_function(new_x)
    current_pos = new_x, new_y

    ax.clear()
    ax.plot(x, y, label='$y=x^2$')
    ax.scatter(current_pos[0], current_pos[1], color='red', label='Current Position')
    ax.legend()
    ax.set_title("Press 'q' to stop animation")
    plt.pause(0.001)

plt.show()