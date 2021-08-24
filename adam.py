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

    lr = 1e-1
    EPOCHES = 1000
    epsilon = 1e-8
    beta1 = 0.9
    beta2 = 0.999
    m_theta = np.zeros(X_train.shape[1])
    v_theta = np.zeros(X_train.shape[1])
    m_b = np.zeros(1)
    v_b = np.zeros(1)

    train_loss = []

    for t in range(1, EPOCHES + 1):
        f = X_train @ theta + b - y_train  # (506, )
        grad_theta = 2 * X_train.T @ f
        grad_b = f.mean()

        m_theta = beta1 * m_theta + (1 - beta1) * grad_theta
        v_theta = beta2 * v_theta + (1 - beta2) * grad_theta**2
        m_b = beta1 * m_b + (1 - beta1) * grad_b
        v_b = beta2 * v_b + (1 - beta2) * grad_b**2

        theta -= lr * m_theta / (1 - beta1**t) / (epsilon +
                                                  np.sqrt(v_theta /
                                                          (1 - beta2**t)))
        b -= lr * m_b / (1 - beta1**t) / (epsilon + np.sqrt(v_b /
                                                            (1 - beta2**t)))

        loss = mean_squared_error(y_train, X_train @ theta + b)
        train_loss.append(loss)

    # 测试误差
    print(mean_squared_error(y_test, X_test @ theta + b))

    # 绘制图像
    plt.plot(train_loss, label="train loss")
    plt.legend()
    plt.show()
