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

    lr = 1e-3
    L = X_train.shape[0]
    batch_size = 64

    train_loss = []

    for i in range(100):
        # 打乱数据
        c = np.random.choice(L, L)
        X_train, y_train = X_train[c], y_train[c]

        # 数据分割
        X_split = np.split(X_train, range(0, L, batch_size)[1:])
        y_split = np.split(y_train, range(0, L, batch_size)[1:])

        for batch in range(len(X_split)):
            batch_X, batch_y = X_split[batch], y_split[batch]
            f = batch_X @ theta + b - batch_y  # (506, )
            grad_theta = 2 * batch_X.T @ f
            grad_b = f.mean()
            theta -= lr * grad_theta
            b -= lr * grad_b

        loss = mean_squared_error(y_train, X_train @ theta + b)
        train_loss.append(loss)

    # 测试误差
    print(mean_squared_error(y_test, X_test @ theta + b))

    # 绘制图像
    plt.plot(train_loss, label="train loss")
    plt.legend()
    plt.show()
