import os.path
import numpy as np
import matplotlib.pyplot as plt


# x 轴的数据（可根据实际情况调整）
def visualize(pred_data, true_data, start=-1, end=-1):
    pred_data = pred_data[start: end: 4]
    true_data = true_data[start: end: 4]
    x = range(len(pred_data))

    # 创建一个图形窗口和坐标轴对象
    fig, ax = plt.subplots()

    # 绘制预测数据曲线，使用红色线条
    ax.plot(x, pred_data, color='red', label='Predicted Data')

    # 绘制真实数据曲线，使用蓝色线条
    ax.plot(x, true_data, color='blue', label='True Data')

    # 添加图例
    ax.legend()

    # 设置标题和坐标轴标签
    ax.set_title('Predicted vs True Data')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # 显示图形
    plt.show()
    if not os.path.exists('./result'):
        os.mkdir('result')
    # plt.savefig('./result/line_plot.png')


if __name__ == "__main__":
    # 预测数据和真实数据的列表
    y_pred_list = np.loadtxt('../result/y_pred_list.txt')
    y_list = np.loadtxt('../result/y_list.txt')
    visualize(y_pred_list, y_list, 0, 500)
