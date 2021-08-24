import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

if __name__ == "__main__":
    '''
    使用Boston房价数据集进行线性回归的批梯度下降
    '''
    # 数据预处理
    X, y = load_boston(return_X_y=True)
    X = (X - X.mean(0)) / X.std(0)
    y = (y - y.mean(0)) / y.std(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

    theta = np.random.randn(13)
    b = np.random.randn(1)

    lr = 1.
    EPOCHES = 100
    epsilon = 1e-8
    G_theta = np.zeros(X_train.shape[1]) + epsilon
    G_b = np.zeros(1) + epsilon

    train_loss = []

    for epoch in range(EPOCHES):
        f = X_train @ theta + b - y_train  # (506, )
        grad_theta = 2 * X_train.T @ f
        grad_b = f.mean()

        G_theta += grad_theta**2
        G_b += grad_b**2

        theta -= lr / np.sqrt(G_theta) * grad_theta
        b -= lr / np.sqrt(G_b) * grad_b

        loss = mean_squared_error(y_train, X_train @ theta + b)
        train_loss.append(loss)

    # 测试误差
    print(mean_squared_error(y_test, X_test @ theta + b))

    # 绘制图像
    plt.plot(train_loss, label="train loss")
    plt.legend()
    plt.show()
