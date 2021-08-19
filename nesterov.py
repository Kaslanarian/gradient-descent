import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 数据预处理
    X, y = load_boston(return_X_y=True)
    X = (X - X.mean(0)) / X.std(0)
    y = (y - y.mean(0)) / y.std(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

    theta = np.random.randn(13)
    b = np.random.randn(1)

    lr = 1e-4
    EPOCHES = 100
    gamma = 0.9
    L = X_train.shape[0]

    train_loss = []

    for epoch in range(EPOCHES):
        c = np.random.choice(L, L)
        X_train, y_train = X_train[c], y_train[c]

        f = X_train[0:1] @ theta + b - y_train[0]
        v_theta = lr * 2 * X_train[0:1].T @ f
        v_b = lr * f.mean()

        for i in range(1, L):
            f = X_train[i:i + 1] @ (theta - gamma * v_theta) + (
                b - gamma * v_b) - y_train[i]
            grad_theta = 2 * X_train[i:i + 1].T @ f
            grad_b = f.mean()

            v_theta = gamma * v_theta + lr * grad_theta
            v_b = gamma * v_b + lr * grad_b

            theta -= v_theta
            b -= v_b

        loss = mean_squared_error(y_train, X_train @ theta + b)
        train_loss.append(loss)

    # 测试误差
    print(mean_squared_error(y_test, X_test @ theta + b))

    # 绘制图像
    plt.plot(train_loss, label="train loss")
    plt.legend()
    plt.show()
