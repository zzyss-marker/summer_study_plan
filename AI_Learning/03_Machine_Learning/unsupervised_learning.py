"""
无监督学习算法实现
从零开始实现聚类和降维算法
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']

class UnsupervisedLearning:
    """无监督学习算法实现类"""
    
    def __init__(self):
        self.algorithms_implemented = []
        print("🔍 无监督学习算法实现系统")
        print("=" * 50)
    
    def kmeans_implementation(self):
        """K-means聚类算法实现"""
        print("🎯 K-means聚类算法实现")
        print("=" * 30)
        
        class KMeans:
            def __init__(self, k=3, max_iters=100, random_state=42):
                self.k = k
                self.max_iters = max_iters
                self.random_state = random_state
                
            def fit(self, X):
                np.random.seed(self.random_state)
                n_samples, n_features = X.shape
                
                # 随机初始化聚类中心
                self.centroids = X[np.random.choice(n_samples, self.k, replace=False)]
                
                self.history = {'centroids': [self.centroids.copy()], 'labels': []}
                
                for iteration in range(self.max_iters):
                    # 分配样本到最近的聚类中心
                    distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
                    labels = np.argmin(distances, axis=0)
                    
                    self.history['labels'].append(labels.copy())
                    
                    # 更新聚类中心
                    new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.k)])
                    
                    # 检查收敛
                    if np.allclose(self.centroids, new_centroids):
                        print(f"K-means在第{iteration+1}次迭代后收敛")
                        break
                    
                    self.centroids = new_centroids
                    self.history['centroids'].append(self.centroids.copy())
                
                self.labels_ = labels
                return self
            
            def predict(self, X):
                distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
                return np.argmin(distances, axis=0)
            
            def inertia(self, X):
                """计算簇内平方和"""
                total_inertia = 0
                for i in range(self.k):
                    cluster_points = X[self.labels_ == i]
                    if len(cluster_points) > 0:
                        total_inertia += np.sum((cluster_points - self.centroids[i])**2)
                return total_inertia
        
        # 生成聚类数据
        X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, 
                              random_state=42)
        
        # 训练K-means
        kmeans = KMeans(k=4, random_state=42)
        kmeans.fit(X)
        
        # 评估结果
        silhouette_avg = silhouette_score(X, kmeans.labels_)
        ari_score = adjusted_rand_score(y_true, kmeans.labels_)
        inertia = kmeans.inertia(X)
        
        print(f"轮廓系数: {silhouette_avg:.4f}")
        print(f"调整兰德指数: {ari_score:.4f}")
        print(f"簇内平方和: {inertia:.2f}")
        
        # 肘部法则 - 确定最佳k值
        print(f"\n肘部法则确定最佳k值:")
        k_range = range(1, 11)
        inertias = []
        
        for k in k_range:
            kmeans_k = KMeans(k=k, random_state=42)
            kmeans_k.fit(X)
            inertias.append(kmeans_k.inertia(X))
        
        # 可视化结果
        plt.figure(figsize=(15, 10))
        
        # K-means聚类结果
        plt.subplot(2, 3, 1)
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        for i in range(kmeans.k):
            cluster_points = X[kmeans.labels_ == i]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                       c=colors[i], alpha=0.6, label=f'簇 {i+1}')
        
        plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], 
                   c='black', marker='x', s=200, linewidths=3, label='聚类中心')
        plt.title('K-means聚类结果')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 真实标签
        plt.subplot(2, 3, 2)
        plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.6)
        plt.title('真实聚类')
        plt.grid(True, alpha=0.3)
        
        # 肘部法则
        plt.subplot(2, 3, 3)
        plt.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('聚类数 k')
        plt.ylabel('簇内平方和')
        plt.title('肘部法则')
        plt.grid(True, alpha=0.3)
        
        # K-means收敛过程
        plt.subplot(2, 3, 4)
        plt.scatter(X[:, 0], X[:, 1], c='lightgray', alpha=0.5)
        
        # 显示前几次迭代的聚类中心
        for i, centroids in enumerate(kmeans.history['centroids'][:5]):
            alpha = 0.3 + 0.7 * i / 4  # 透明度递增
            plt.scatter(centroids[:, 0], centroids[:, 1], 
                       c='red', marker='x', s=100, alpha=alpha, 
                       label=f'迭代 {i+1}' if i < 3 else None)
        
        plt.title('K-means收敛过程')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 轮廓分析
        plt.subplot(2, 3, 5)
        from sklearn.metrics import silhouette_samples
        silhouette_vals = silhouette_samples(X, kmeans.labels_)
        
        y_lower = 10
        for i in range(kmeans.k):
            cluster_silhouette_vals = silhouette_vals[kmeans.labels_ == i]
            cluster_silhouette_vals.sort()
            
            size_cluster_i = cluster_silhouette_vals.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = colors[i]
            plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals,
                             facecolor=color, edgecolor=color, alpha=0.7)
            
            plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10
        
        plt.axvline(x=silhouette_avg, color="red", linestyle="--", 
                   label=f'平均轮廓系数 = {silhouette_avg:.3f}')
        plt.xlabel('轮廓系数')
        plt.ylabel('簇标签')
        plt.title('轮廓分析')
        plt.legend()
        
        # 不同初始化的比较
        plt.subplot(2, 3, 6)
        
        # 多次运行K-means，比较结果稳定性
        inertias_multiple = []
        for seed in range(10):
            kmeans_temp = KMeans(k=4, random_state=seed)
            kmeans_temp.fit(X)
            inertias_multiple.append(kmeans_temp.inertia(X))
        
        plt.bar(range(10), inertias_multiple, alpha=0.7)
        plt.xlabel('随机种子')
        plt.ylabel('簇内平方和')
        plt.title('不同初始化的结果')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        self.algorithms_implemented.append("K-means聚类")
    
    def hierarchical_clustering_implementation(self):
        """层次聚类算法实现"""
        print("\n🌳 层次聚类算法实现")
        print("=" * 30)
        
        class HierarchicalClustering:
            def __init__(self, linkage='single'):
                self.linkage = linkage  # 'single', 'complete', 'average'
                
            def fit(self, X):
                n_samples = X.shape[0]
                
                # 计算距离矩阵
                self.distance_matrix = self._compute_distance_matrix(X)
                
                # 初始化：每个点为一个簇
                clusters = [[i] for i in range(n_samples)]
                self.merge_history = []
                
                while len(clusters) > 1:
                    # 找到最近的两个簇
                    min_dist = float('inf')
                    merge_i, merge_j = -1, -1
                    
                    for i in range(len(clusters)):
                        for j in range(i + 1, len(clusters)):
                            dist = self._cluster_distance(clusters[i], clusters[j])
                            if dist < min_dist:
                                min_dist = dist
                                merge_i, merge_j = i, j
                    
                    # 合并簇
                    new_cluster = clusters[merge_i] + clusters[merge_j]
                    self.merge_history.append((clusters[merge_i].copy(), 
                                             clusters[merge_j].copy(), min_dist))
                    
                    # 更新簇列表
                    clusters = [clusters[k] for k in range(len(clusters)) 
                               if k != merge_i and k != merge_j] + [new_cluster]
                
                return self
            
            def _compute_distance_matrix(self, X):
                n_samples = X.shape[0]
                dist_matrix = np.zeros((n_samples, n_samples))
                
                for i in range(n_samples):
                    for j in range(i + 1, n_samples):
                        dist = np.sqrt(np.sum((X[i] - X[j])**2))
                        dist_matrix[i, j] = dist_matrix[j, i] = dist
                
                return dist_matrix
            
            def _cluster_distance(self, cluster1, cluster2):
                distances = []
                for i in cluster1:
                    for j in cluster2:
                        distances.append(self.distance_matrix[i, j])
                
                if self.linkage == 'single':
                    return min(distances)  # 单链接
                elif self.linkage == 'complete':
                    return max(distances)  # 全链接
                elif self.linkage == 'average':
                    return np.mean(distances)  # 平均链接
            
            def get_clusters(self, n_clusters):
                """获取指定数量的簇"""
                if n_clusters <= 0:
                    return []
                
                # 从合并历史中重建簇
                n_samples = len(self.merge_history) + 1
                clusters = [[i] for i in range(n_samples)]
                
                merges_to_do = n_samples - n_clusters
                for i in range(merges_to_do):
                    cluster1, cluster2, _ = self.merge_history[i]
                    
                    # 找到对应的簇并合并
                    idx1 = idx2 = -1
                    for j, cluster in enumerate(clusters):
                        if set(cluster) == set(cluster1):
                            idx1 = j
                        elif set(cluster) == set(cluster2):
                            idx2 = j
                    
                    if idx1 != -1 and idx2 != -1:
                        new_cluster = clusters[idx1] + clusters[idx2]
                        clusters = [clusters[k] for k in range(len(clusters)) 
                                   if k != idx1 and k != idx2] + [new_cluster]
                
                return clusters
        
        # 生成测试数据
        X_hier, _ = make_blobs(n_samples=50, centers=3, cluster_std=1.0, 
                              random_state=42)
        
        # 测试不同的链接方法
        linkage_methods = ['single', 'complete', 'average']
        
        plt.figure(figsize=(15, 5))
        
        for i, linkage in enumerate(linkage_methods):
            hc = HierarchicalClustering(linkage=linkage)
            hc.fit(X_hier)
            
            # 获取3个簇
            clusters = hc.get_clusters(3)
            
            # 创建标签数组
            labels = np.zeros(len(X_hier))
            for cluster_id, cluster in enumerate(clusters):
                for point_id in cluster:
                    labels[point_id] = cluster_id
            
            plt.subplot(1, 3, i + 1)
            plt.scatter(X_hier[:, 0], X_hier[:, 1], c=labels, cmap='viridis', alpha=0.7)
            plt.title(f'{linkage.capitalize()} Linkage')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("层次聚类特点:")
        print("• Single Linkage: 容易产生链状簇")
        print("• Complete Linkage: 倾向于产生紧凑的球状簇")
        print("• Average Linkage: 介于两者之间")
        
        self.algorithms_implemented.append("层次聚类")
    
    def pca_implementation(self):
        """主成分分析(PCA)实现"""
        print("\n📊 主成分分析(PCA)实现")
        print("=" * 30)
        
        class PCA:
            def __init__(self, n_components=2):
                self.n_components = n_components
                
            def fit(self, X):
                # 数据中心化
                self.mean_ = np.mean(X, axis=0)
                X_centered = X - self.mean_
                
                # 计算协方差矩阵
                cov_matrix = np.cov(X_centered.T)
                
                # 特征值分解
                eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
                
                # 按特征值大小排序
                idx = np.argsort(eigenvalues)[::-1]
                self.eigenvalues_ = eigenvalues[idx]
                self.eigenvectors_ = eigenvectors[:, idx]
                
                # 选择前n_components个主成分
                self.components_ = self.eigenvectors_[:, :self.n_components].T
                
                # 计算方差解释比例
                self.explained_variance_ratio_ = self.eigenvalues_ / np.sum(self.eigenvalues_)
                
                return self
            
            def transform(self, X):
                X_centered = X - self.mean_
                return np.dot(X_centered, self.components_.T)
            
            def fit_transform(self, X):
                return self.fit(X).transform(X)
            
            def inverse_transform(self, X_transformed):
                return np.dot(X_transformed, self.components_) + self.mean_
        
        # 生成高维数据
        np.random.seed(42)
        n_samples = 200
        
        # 创建相关的高维数据
        X_high = np.random.randn(n_samples, 5)
        # 添加相关性
        X_high[:, 1] = X_high[:, 0] + 0.5 * np.random.randn(n_samples)
        X_high[:, 2] = -X_high[:, 0] + 0.3 * np.random.randn(n_samples)
        X_high[:, 3] = 0.5 * X_high[:, 1] + 0.4 * np.random.randn(n_samples)
        X_high[:, 4] = np.random.randn(n_samples)  # 独立维度
        
        # 应用PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_high)
        
        print(f"原始数据维度: {X_high.shape}")
        print(f"PCA后维度: {X_pca.shape}")
        print(f"主成分方差解释比例: {pca.explained_variance_ratio_[:2]}")
        print(f"累计方差解释比例: {np.sum(pca.explained_variance_ratio_[:2]):.4f}")
        
        # 可视化PCA结果
        plt.figure(figsize=(15, 10))
        
        # 原始数据的前两个维度
        plt.subplot(2, 3, 1)
        plt.scatter(X_high[:, 0], X_high[:, 1], alpha=0.6)
        plt.xlabel('特征1')
        plt.ylabel('特征2')
        plt.title('原始数据 (前两个维度)')
        plt.grid(True, alpha=0.3)
        
        # PCA降维结果
        plt.subplot(2, 3, 2)
        plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, color='red')
        plt.xlabel('第一主成分')
        plt.ylabel('第二主成分')
        plt.title('PCA降维结果')
        plt.grid(True, alpha=0.3)
        
        # 方差解释比例
        plt.subplot(2, 3, 3)
        plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
                pca.explained_variance_ratio_, alpha=0.7)
        plt.xlabel('主成分')
        plt.ylabel('方差解释比例')
        plt.title('各主成分方差解释比例')
        plt.grid(True, alpha=0.3)
        
        # 累计方差解释比例
        plt.subplot(2, 3, 4)
        cumsum_var = np.cumsum(pca.explained_variance_ratio_)
        plt.plot(range(1, len(cumsum_var) + 1), cumsum_var, 'bo-', linewidth=2)
        plt.axhline(y=0.95, color='r', linestyle='--', label='95%阈值')
        plt.xlabel('主成分数量')
        plt.ylabel('累计方差解释比例')
        plt.title('累计方差解释比例')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 主成分载荷图
        plt.subplot(2, 3, 5)
        feature_names = [f'特征{i+1}' for i in range(X_high.shape[1])]
        
        for i, feature in enumerate(feature_names):
            plt.arrow(0, 0, pca.components_[0, i], pca.components_[1, i], 
                     head_width=0.05, head_length=0.05, fc='blue', ec='blue')
            plt.text(pca.components_[0, i] * 1.1, pca.components_[1, i] * 1.1, 
                    feature, fontsize=10)
        
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.xlabel('第一主成分载荷')
        plt.ylabel('第二主成分载荷')
        plt.title('主成分载荷图')
        plt.grid(True, alpha=0.3)
        
        # 重构误差分析
        plt.subplot(2, 3, 6)
        n_components_range = range(1, X_high.shape[1] + 1)
        reconstruction_errors = []
        
        for n_comp in n_components_range:
            pca_temp = PCA(n_components=n_comp)
            X_transformed = pca_temp.fit_transform(X_high)
            X_reconstructed = pca_temp.inverse_transform(X_transformed)
            error = np.mean((X_high - X_reconstructed)**2)
            reconstruction_errors.append(error)
        
        plt.plot(n_components_range, reconstruction_errors, 'go-', linewidth=2)
        plt.xlabel('主成分数量')
        plt.ylabel('重构误差')
        plt.title('重构误差 vs 主成分数量')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        self.algorithms_implemented.append("主成分分析")
    
    def dbscan_implementation(self):
        """DBSCAN密度聚类实现"""
        print("\n🔍 DBSCAN密度聚类实现")
        print("=" * 30)
        
        class DBSCAN:
            def __init__(self, eps=0.5, min_samples=5):
                self.eps = eps
                self.min_samples = min_samples
                
            def fit(self, X):
                n_samples = X.shape[0]
                self.labels_ = np.full(n_samples, -1)  # -1表示噪声点
                
                cluster_id = 0
                visited = np.zeros(n_samples, dtype=bool)
                
                for i in range(n_samples):
                    if visited[i]:
                        continue
                    
                    visited[i] = True
                    neighbors = self._get_neighbors(X, i)
                    
                    if len(neighbors) < self.min_samples:
                        # 标记为噪声点
                        self.labels_[i] = -1
                    else:
                        # 开始新簇
                        self._expand_cluster(X, i, neighbors, cluster_id, visited)
                        cluster_id += 1
                
                return self
            
            def _get_neighbors(self, X, point_idx):
                distances = np.sqrt(np.sum((X - X[point_idx])**2, axis=1))
                return np.where(distances <= self.eps)[0]
            
            def _expand_cluster(self, X, point_idx, neighbors, cluster_id, visited):
                self.labels_[point_idx] = cluster_id
                
                i = 0
                while i < len(neighbors):
                    neighbor_idx = neighbors[i]
                    
                    if not visited[neighbor_idx]:
                        visited[neighbor_idx] = True
                        new_neighbors = self._get_neighbors(X, neighbor_idx)
                        
                        if len(new_neighbors) >= self.min_samples:
                            neighbors = np.concatenate([neighbors, new_neighbors])
                    
                    if self.labels_[neighbor_idx] == -1:
                        self.labels_[neighbor_idx] = cluster_id
                    
                    i += 1
        
        # 生成包含噪声的数据
        X_circles, _ = make_circles(n_samples=300, factor=0.3, noise=0.1, random_state=42)
        
        # 添加一些噪声点
        noise_points = np.random.uniform(-2, 2, (50, 2))
        X_with_noise = np.vstack([X_circles, noise_points])
        
        # 应用DBSCAN
        dbscan = DBSCAN(eps=0.3, min_samples=10)
        dbscan.fit(X_with_noise)
        
        # 统计结果
        n_clusters = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
        n_noise = list(dbscan.labels_).count(-1)
        
        print(f"聚类数量: {n_clusters}")
        print(f"噪声点数量: {n_noise}")
        
        # 可视化DBSCAN结果
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(X_with_noise[:, 0], X_with_noise[:, 1], alpha=0.6)
        plt.title('原始数据（包含噪声）')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        unique_labels = set(dbscan.labels_)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # 噪声点用黑色表示
                col = 'black'
                marker = 'x'
                alpha = 0.5
                label = '噪声'
            else:
                marker = 'o'
                alpha = 0.7
                label = f'簇 {k}'
            
            class_member_mask = (dbscan.labels_ == k)
            xy = X_with_noise[class_member_mask]
            plt.scatter(xy[:, 0], xy[:, 1], c=[col], marker=marker, 
                       alpha=alpha, s=50, label=label)
        
        plt.title(f'DBSCAN聚类结果\n(eps={dbscan.eps}, min_samples={dbscan.min_samples})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("DBSCAN特点:")
        print("• 可以发现任意形状的簇")
        print("• 自动确定簇的数量")
        print("• 能够识别噪声点")
        print("• 对参数eps和min_samples敏感")
        
        self.algorithms_implemented.append("DBSCAN聚类")
    
    def run_all_algorithms(self):
        """运行所有无监督学习算法"""
        print("🔍 无监督学习算法完整实现")
        print("=" * 60)
        
        self.kmeans_implementation()
        self.hierarchical_clustering_implementation()
        self.pca_implementation()
        self.dbscan_implementation()
        
        print(f"\n🎉 无监督学习算法实现完成！")
        print(f"已实现的算法: {', '.join(self.algorithms_implemented)}")
        
        print(f"\n📚 算法总结:")
        print("1. K-means - 基于距离的聚类，需要预设簇数")
        print("2. 层次聚类 - 构建聚类树，不需要预设簇数")
        print("3. PCA - 线性降维，保持最大方差")
        print("4. DBSCAN - 密度聚类，可发现任意形状的簇")

def main():
    """主函数"""
    unsupervised = UnsupervisedLearning()
    unsupervised.run_all_algorithms()
    
    print("\n💡 算法选择指南:")
    print("1. K-means - 球状簇，已知簇数")
    print("2. 层次聚类 - 探索性分析，构建分类体系")
    print("3. PCA - 数据可视化，特征降维")
    print("4. DBSCAN - 任意形状簇，有噪声数据")

if __name__ == "__main__":
    main()
