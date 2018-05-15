import numpy as np

def softmax(x):
    """
        Softmax 函数
    """
    assert len(x.shape) > 1, "Softmax的得分向量要求维度高于1"
    print(x)
    x -= np.max(x, axis=0, keepdims=True)
    print(x)
    x = np.exp(x) / np.sum(np.exp(x), axis=0, keepdims=True)

    return x


def softmax1(x):
    """
        Softmax 函数
    """
    assert len(x.shape) > 1, "Softmax的得分向量要求维度高于1"
    print(x)
    x -= np.max(x, axis=1, keepdims=True)
    print(x)
    x = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    return x


if __name__ == "__main__":
    x = np.array([[1, 2, 6], [3, 4, 9]])
    print(softmax(x))
    print(x)
    print(softmax1(x))
