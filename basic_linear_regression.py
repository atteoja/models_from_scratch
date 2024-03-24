import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(x, y, m, b, a = 0.001):

    times = len(x)

    dloss_m = 0
    dloss_b = 0

    for i in range(times):
        yhat = m * x[i] + b
        dloss_m += (-2) * (y[i] - yhat) * x[i]
        dloss_b += (-2) * (y[i] - yhat)

    new_m = m - dloss_m * a
    new_b = b - dloss_b * a

    return new_m, new_b

def get_loss(x, y, m, b):

    times = len(x)

    loss = 0

    for i in range(times):
        yhat = m * x[i] + b
        loss += (y[i] - yhat) ** 2

    return loss

def main():

    a = 0.002

    x = np.random.rand(10) * 10
    y = np.random.rand(10) * 3 + 4

    m = np.random.rand()
    b = np.random.rand()

    i = 0

    plt.ion()

    fig, ax = plt.subplots()

    # Plot initial points
    ax.plot(x, y, 'o')

    losses = []

    last_loss = 0

    while True:

        loss = get_loss(x, y, m, b)
        losses.append(loss)

        # Clear the axis
        ax.clear()
        
        # Set axis limits
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)

        # Plot points
        ax.plot(x, y, 'o')

        # Plot line
        ax.plot(x, m * x + b)

        # show m and b values 2 decimal
        ax.text(0.5, 0.5, 'm = ' + str(round(m, 2)) + ' b = ' + str(round(b, 2)))

        # Draw the plot
        plt.draw()

        plt.pause(0.1)

        print(loss)

        # Update m and b
        m, b = gradient_descent(x, y, m, b, a)

        i += 1

        if abs(last_loss - loss) > 0.0001 and i < 1000:
            last_loss = loss
            continue
        else:
            plt.ioff()
            break
            
    plt.show()

    # Plot loss
    plt.plot(losses)
    plt.title('Loss')
    plt.show()

if __name__ == '__main__':
    main()
