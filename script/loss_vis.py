import matplotlib.pyplot as plt

def plot_loss_curves(loss_curves, labels):
    """
    绘制多个数组对应的多个损失函数下降曲线。

    参数：
    - loss_curves: 一个列表，包含多个数组，每个数组代表一个损失函数下降曲线的值。
    - labels: 一个列表，包含每个损失函数对应的标签。

    注意：loss_curves 和 labels 的长度应相等。
    """

    # 创建图形和子图
    fig, ax = plt.subplots()

    # 设置横坐标和标签
    x = range(len(loss_curves[0]))
    ax.set_xlabel('epoch')

    # 设置纵坐标和标签
    ax.set_ylabel('loss')

    # 绘制每个损失函数的曲线
    for loss_curve, label in zip(loss_curves, labels):
        ax.plot(x, loss_curve, label=label)


    # 添加图例
    ax.legend()
    plt.title('Title')
    # 显示图形
    plt.show()
    plt.savefig("loss.jpg")


if __name__ == "__main__":
    import numpy as np

    # 假设有两个损失函数下降曲线
    loss_curve1 = np.load("./checkpoints/gpstg-pems-bay-15/train_loss.npy")
    loss_curve2 = np.load("./checkpoints/gpstg-pems-bay-15/val_loss.npy")
    loss_curve3 = [0.5 for i in range(172)]
    print(loss_curve1[:10], loss_curve2[:10])
    # 损失函数的标签
    labels = ['Train loss', 'Val loss', '3']

    # 调用函数进行绘图
    plot_loss_curves([loss_curve1, loss_curve2, loss_curve3], labels)
