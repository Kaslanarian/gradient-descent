import numpy as np
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import seaborn as sns


def mse_loss(X, y, theta):
    return np.sum((X @ theta - y)**2 / 2)


def grad(X, y, theta):
    return X.T @ (X @ theta - y)


def BGD(X, y, theta=np.random.rand(13), lr=1e-4, epoches=100):
    loss_list = [mse_loss(X, y, theta)]
    for epoch in range(epoches):
        grad_theta = grad(X, y, theta)
        theta -= lr * grad_theta
        loss = mse_loss(X, y, theta)
        loss_list.append(loss)

    return loss_list


def SGD(X, y, theta=np.random.rand(13), lr=1e-4, epoches=100):
    L = len(X)
    loss_list = [mse_loss(X, y, theta)]
    for epoch in range(epoches):
        c = np.random.choice(L, L)
        X, y = X[c], y[c]

        for i in range(L):
            grad_theta = grad(X[i:i + 1], y[i], theta)
            theta -= lr * grad_theta

        loss = mse_loss(X, y, theta)
        loss_list.append(loss)

    return loss_list


def mini_batch(X,
               y,
               theta=np.random.rand(13),
               lr=1e-4,
               epoches=100,
               batch_size=8):
    L = len(X)
    loss_list = [mse_loss(X, y, theta)]
    for epoch in range(epoches):
        c = np.random.choice(L, L)
        X, y = X[c], y[c]

        # 数据分割
        X_split = np.split(X, range(0, L, batch_size)[1:])
        y_split = np.split(y, range(0, L, batch_size)[1:])

        for batch in range(len(X_split)):
            batch_X, batch_y = X_split[batch], y_split[batch]
            grad_theta = grad(batch_X, batch_y, theta)
            theta -= lr * grad_theta

        loss = mse_loss(X, y, theta)
        loss_list.append(loss)

    return loss_list


def momentum(X, y, theta=np.random.rand(13), lr=1e-4, epoches=100, gamma=0.9):
    L = len(X)
    loss_list = [mse_loss(X, y, theta)]

    for epoch in range(epoches):
        c = np.random.choice(L, L)
        X, y = X[c], y[c]

        v_theta = lr * grad(X[0:1], y[0], theta)

        for i in range(1, L):
            grad_theta = grad(X[i:i + 1], y[i], theta)

            v_theta = gamma * v_theta + lr * grad_theta
            theta -= v_theta

        loss = mse_loss(X, y, theta)
        loss_list.append(loss)

    return loss_list


def nestrov(X, y, theta=np.random.rand(13), lr=1e-4, epoches=100, gamma=0.9):
    L = len(X)
    loss_list = [mse_loss(X, y, theta)]

    for epoch in range(epoches):
        c = np.random.choice(L, L)
        X, y = X[c], y[c]

        v_theta = lr * grad(X[0:1], y[0], theta)

        for i in range(1, L):
            grad_theta = grad(X[i:i + 1], y[i], theta - gamma * v_theta)
            v_theta = gamma * v_theta + lr * grad_theta
            theta -= v_theta

        loss = mse_loss(X, y, theta)
        loss_list.append(loss)

    return loss_list


def adagrad(X,
            y,
            theta=np.random.rand(13),
            lr=1e-4,
            epoches=100,
            epsilon=1e-8):
    loss_list = [mse_loss(X, y, theta)]
    G_theta = np.zeros(len(theta)) + epsilon
    for epoch in range(epoches):
        grad_theta = grad(X, y, theta)
        G_theta += grad_theta**2
        theta -= lr * grad_theta / np.sqrt(G_theta)

        loss = mse_loss(X, y, theta)
        loss_list.append(loss)

    return loss_list


def adadelta(X,
             y,
             theta=np.random.rand(13),
             lr=1e-4,
             epoches=100,
             gamma=0.9,
             epsilon=1e-8):
    loss_list = [mse_loss(X, y, theta)]
    G_theta = np.zeros(len(theta))
    for epoch in range(EPOCHES):
        grad_theta = grad(X, y, theta)
        G_theta = (1 - gamma) * grad_theta**2 + gamma * G_theta
        theta -= lr * grad_theta / (np.sqrt(G_theta) + epsilon)

        loss = mse_loss(X, y, theta)
        loss_list.append(loss)

    return loss_list


def adam(X,
         y,
         theta=np.random.rand(13),
         lr=1e-4,
         epoches=100,
         beta1=0.9,
         beta2=0.999,
         epsilon=1e-8):
    m_theta = np.zeros(len(theta))
    v_theta = np.zeros(len(theta))
    loss_list = [mse_loss(X, y, theta)]
    for t in range(1, epoches + 1):
        grad_theta = grad(X, y, theta)
        m_theta = beta1 * m_theta + (1 - beta1) * grad_theta
        v_theta = beta2 * v_theta + (1 - beta2) * grad_theta**2

        theta -= lr / (epsilon +
                       np.sqrt(v_theta /
                               (1 - beta2**t))) * m_theta / (1 - beta1**t)

        loss = mse_loss(X, y, theta)
        loss_list.append(loss)

    return loss_list


if __name__ == "__main__":
    X, y = load_boston(return_X_y=True)
    X = (X - X.mean(0)) / X.std(0)
    y = (y - y.mean(0)) / y.std(0)
    theta = np.random.rand(13)
    EPOCHES = 100

    sgd_list = SGD(X, y, np.copy(theta), epoches=EPOCHES)
    bgd_list = BGD(X, y, np.copy(theta), epoches=EPOCHES)
    mini_list = mini_batch(X, y, np.copy(theta), epoches=EPOCHES)
    mom_list = momentum(X, y, np.copy(theta), epoches=EPOCHES)
    nest_list = nestrov(X, y, np.copy(theta), epoches=EPOCHES)
    ada_list = adagrad(X, y, np.copy(theta), lr=0.1, epoches=EPOCHES)
    delta_list = adadelta(X, y, np.copy(theta), lr=0.1, epoches=EPOCHES)
    adam_list = adam(X, y, np.copy(theta), lr=0.1, epoches=EPOCHES)
    plt.plot(sgd_list, label="SGD")
    plt.plot(bgd_list, label="BGD")
    plt.plot(mini_list, label="mini-batch")
    plt.plot(mom_list, label="Momentum")
    plt.plot(nest_list, label="Nesterov")
    plt.plot(ada_list, label="AdaGrad lr=0.1")
    plt.plot(delta_list, label="AdaDelta lr=0.1")
    plt.plot(adam_list, label="Adam lr=0.1")
    plt.legend()
    plt.show()