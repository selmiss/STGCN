import numpy as np
from sklearn.decomposition import PCA


def pca_analysis(pca, input_data):
    pca.fit(input_data)
    result = pca.transform(input_data)
    # explained_variance_ratio = pca.explained_variance_ratio_
    # for i, ratio in enumerate(explained_variance_ratio):
    #     print(f"主成分{i+1}的方差贡献度: {ratio:.4f}")
    print(result)
    print(pca.inverse_transform(result))
    return result


def pca_graph(pca, graph):
    pca.fit(graph)
    result = pca.transform(graph)
    # explained_variance_ratio = pca.explained_variance_ratio_
    # for i, ratio in enumerate(explained_variance_ratio):
    #     print(f"主成分{i+1}的方差贡献度: {ratio:.4f}")
    print(pca.inverse_transform(result))
    return result

if __name__ == "__main__":
    # 创建示例数据
    input_arr = np.array([i for i in range(4)])
    pca = PCA(n_components=4)

    X = np.array([input_arr, input_arr - 10, input_arr + 1, input_arr**2])  # 假设有100个样本，每个样本有50个特征
    pca_analysis(pca, X)
