"""
敏感度分析脚本
每次修改一个参数，分别保存去噪和整合的结果到不同的CSV文件
"""

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import sys
import os
from sklearn.metrics import (
    normalized_mutual_info_score, mutual_info_score, adjusted_mutual_info_score,
    v_measure_score, homogeneity_score, completeness_score,
    adjusted_rand_score, fowlkes_mallows_score
)

# 添加项目根目录到 Python 路径（从根目录的codes文件夹导入）
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入CANDIES相关模块（从根目录的codes文件夹导入）
from codes1.DiTs import seed_everything
from codes1.sampler import *
from codes1.train_diff import *
from codes1.ZINB_encoder import *
from codes1.preprocess1 import *
from codes1.AutoEncoder import *
from codes1.get_graph import *
from codes1.integration import *


# 默认参数值
DEFAULT_SPATIAL_K = 3
DEFAULT_FEATURE_K = 20
DEFAULT_DIFFUSION_STEP = 800
DEFAULT_ALPHA = 2.0
DEFAULT_BETA = 0.1
DEFAULT_LAMBDA_CL = 0.5  # 对比学习损失权重
DEFAULT_LAMBDA_REC = 0.5  # 重构损失权重（lambda_cl + lambda_rec = 1）

# 参数测试范围（每次只修改一个参数，其他保持默认值）
PARAM_RANGES = {
    'spatial_k': [1, 3, 5, 8, 10],  # 空间图的k值
    'feature_k': [10, 15, 20, 25, 30],  # 特征图的k值
    'diffusion_step': [300, 500, 800, 1000, 1200],  # Diffusion步数
    'alpha': [1.0, 1.5, 2.0, 2.5, 3.0],  # 编码阶段重构损失权重
    'beta': [0.05, 0.1, 0.15, 0.2, 0.3],  # 编码阶段ZINB损失权重
    'lambda_cl': [0.1, 0.3, 0.5, 0.7, 0.9]  # 整合阶段对比学习损失权重（lambda_rec = 1 - lambda_cl）
}

# 是否运行所有参数组合（True: 遍历所有参数范围, False: 只运行默认值）
RUN_ALL_PARAMS = True

# 数据路径
file_fold = 'E:/yan0/ours/data/simulation/'

# 结果保存路径
RESULT_DIR = "E:/yan0/ours/CANDIES_v2/sensitivity_analysis/parameters_analysis/results"
os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================================
# 辅助函数
# ============================================================================

class ConditionalDiffusionDataset():
    def __init__(self, adata_omics1, adata_omics2):
        self.adata_omics1 = adata_omics1
        self.adata_omics2 = adata_omics2
        self.st_sample = torch.tensor(self.adata_omics1, dtype=torch.float32)
        self.con_sample = torch.tensor(self.adata_omics2, dtype=torch.float32)
        self.con_data = torch.tensor(self.adata_omics2, dtype=torch.float32)

    def __len__(self):
        return len(self.adata_omics1)

    def __getitem__(self, idx):
        return self.st_sample[idx], self.con_sample[idx], self.con_data


def calculate_metrics(ground_truth, predictions):
    """计算所有评估指标"""
    metrics = {
        'Mutual Information': mutual_info_score(ground_truth, predictions),
        'NMI': normalized_mutual_info_score(ground_truth, predictions),
        'AMI': adjusted_mutual_info_score(ground_truth, predictions),
        'V-measure': v_measure_score(ground_truth, predictions),
        'Homogeneity': homogeneity_score(ground_truth, predictions),
        'Completeness': completeness_score(ground_truth, predictions),
        'ARI': adjusted_rand_score(ground_truth, predictions),
        'FMI': fowlkes_mallows_score(ground_truth, predictions)
    }
    return metrics


def print_metrics(metrics, method_name):
    """打印评估指标"""
    print(f'\n{method_name}\n')
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")


# ============================================================================
# 主流程
# ============================================================================

def load_and_preprocess_data():
    """数据读取和预处理（只需要执行一次）"""
    print("\n" + "=" * 80)
    print("数据读取和预处理（一次性执行）")
    print("=" * 80)
    
    # ========================================================================
    # 1. 数据读取
    # ========================================================================
    print("\n[1/2] 读取数据...")
    adata_omics1 = sc.read_h5ad(file_fold + 'rna(std3).h5ad')
    adata_omics2 = sc.read_h5ad(file_fold + 'adt(std1).h5ad')
    adata_omics1.var_names_make_unique()
    adata_omics2.var_names_make_unique()
    
    # 对齐空间坐标
    spatial1 = pd.DataFrame(adata_omics1.obsm['spatial'], columns=['x', 'y'])
    spatial2 = pd.DataFrame(adata_omics2.obsm['spatial'], columns=['x', 'y'])
    spatial1['index1'] = spatial1.index
    spatial2['index2'] = spatial2.index
    merged = pd.merge(spatial1, spatial2, on=['x', 'y'], how='inner')
    sorted_index1 = merged['index1'].values
    sorted_index2 = merged['index2'].values
    adata_omics1 = adata_omics1[sorted_index1]
    adata_omics2 = adata_omics2[sorted_index2]
    
    labels = pd.read_csv(file_fold + 'simulation_ground_truth.csv')
    labels.index = adata_omics1.obs.index
    adata_omics1.obs['ground_truth'] = labels['leiden1']
    adata_omics2.obs['ground_truth'] = labels['leiden1']
    
    # ========================================================================
    # 2. 数据预处理
    # ========================================================================
    print("\n[2/2] 数据预处理...")
    # RNA
    print("  - RNA: 过滤基因...")
    sc.pp.filter_genes(adata_omics1, min_cells=50)
    sc.pp.filter_genes(adata_omics1, min_counts=10)
    print("  - RNA: 归一化...")
    sc.pp.normalize_total(adata_omics1, target_sum=1e6)
    print("  - RNA: 寻找高变基因（这步可能较慢）...")
    sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=2000)
    print("  - RNA: 标准化...")
    sc.pp.scale(adata_omics1)
    print("  - RNA: PCA降维...")
    adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
    n_comps_rna = min(adata_omics2.n_vars - 1, 50)  # 限制PCA组件数量，避免过大
    print(f"    RNA数据形状: {adata_omics1_high.shape}, PCA组件数: {n_comps_rna}")
    
    # 检查数据是否是稀疏矩阵
    import scipy.sparse as sp
    is_sparse = sp.issparse(adata_omics1_high.X)
    print(f"    数据是否为稀疏矩阵: {is_sparse}")
    
    try:
        print("    尝试使用scanpy的PCA...")
        # 使用scanpy的PCA可能更高效
        sc.tl.pca(adata_omics1_high, n_comps=n_comps_rna, use_highly_variable=False, svd_solver='arpack')
        adata_omics1.obsm['feat'] = adata_omics1_high.obsm['X_pca']
        print("    scanpy PCA完成")
    except Exception as e:
        # 如果scanpy失败，使用原来的pca函数
        print(f"    scanpy PCA失败: {e}")
        print("    使用自定义PCA函数...")
        if is_sparse:
            print("    转换稀疏矩阵为密集矩阵（这可能需要一些时间）...")
            X_dense = adata_omics1_high.X.toarray() if is_sparse else adata_omics1_high.X
            print("    转换完成，开始PCA...")
            from sklearn.decomposition import PCA
            pca_model = PCA(n_components=n_comps_rna)
            adata_omics1.obsm['feat'] = pca_model.fit_transform(X_dense)
        else:
            adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=n_comps_rna)
        print("    自定义PCA完成")
    print("  - RNA: 完成")
    
    # Protein
    print("  - Protein: 标准化...")
    sc.pp.scale(adata_omics2)
    print("  - Protein: PCA降维...")
    n_comps_protein = min(adata_omics2.n_vars - 1, 50)  # 限制PCA组件数量
    print(f"    Protein数据形状: {adata_omics2.shape}, PCA组件数: {n_comps_protein}")
    
    is_sparse_protein = sp.issparse(adata_omics2.X)
    print(f"    数据是否为稀疏矩阵: {is_sparse_protein}")
    
    try:
        print("    尝试使用scanpy的PCA...")
        sc.tl.pca(adata_omics2, n_comps=n_comps_protein, use_highly_variable=False, svd_solver='arpack')
        adata_omics2.obsm['feat'] = adata_omics2.obsm['X_pca']
        print("    scanpy PCA完成")
    except Exception as e:
        print(f"    scanpy PCA失败: {e}")
        print("    使用自定义PCA函数...")
        if is_sparse_protein:
            print("    转换稀疏矩阵为密集矩阵（这可能需要一些时间）...")
            X_dense_protein = adata_omics2.X.toarray() if is_sparse_protein else adata_omics2.X
            print("    转换完成，开始PCA...")
            from sklearn.decomposition import PCA
            pca_model = PCA(n_components=n_comps_protein)
            adata_omics2.obsm['feat'] = pca_model.fit_transform(X_dense_protein)
        else:
            adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=n_comps_protein)
        print("    自定义PCA完成")
    print("  - Protein: 完成")
    
    print("\n数据读取和预处理完成！")
    return adata_omics1, adata_omics2


def run_single_experiment(adata_omics1, adata_omics2, spatial_k, feature_k, diffusion_step, alpha, beta, lambda_cl, lambda_rec):
    """运行单次实验"""
    # 识别当前修改的参数类型（用于生成文件名）
    param_type = None
    param_value = None
    if spatial_k != DEFAULT_SPATIAL_K:
        param_type = "spatial_k"
        param_value = spatial_k
    elif feature_k != DEFAULT_FEATURE_K:
        param_type = "feature_k"
        param_value = feature_k
    elif diffusion_step != DEFAULT_DIFFUSION_STEP:
        param_type = "diffusion_step"
        param_value = diffusion_step
    elif alpha != DEFAULT_ALPHA:
        param_type = "alpha"
        param_value = alpha
    elif beta != DEFAULT_BETA:
        param_type = "beta"
        param_value = beta
    elif lambda_cl != DEFAULT_LAMBDA_CL:
        param_type = "lambda_cl"
        param_value = lambda_cl
    else:
        param_type = "default"
        param_value = "default"
    
    # 生成参数名称（用于显示）
    param_name_parts = []
    if spatial_k != DEFAULT_SPATIAL_K:
        param_name_parts.append(f"spatial_k_{spatial_k}")
    if feature_k != DEFAULT_FEATURE_K:
        param_name_parts.append(f"feature_k_{feature_k}")
    if diffusion_step != DEFAULT_DIFFUSION_STEP:
        param_name_parts.append(f"diffusion_step_{diffusion_step}")
    if alpha != DEFAULT_ALPHA:
        param_name_parts.append(f"alpha_{alpha}")
    if beta != DEFAULT_BETA:
        param_name_parts.append(f"beta_{beta}")
    if lambda_cl != DEFAULT_LAMBDA_CL:
        param_name_parts.append(f"lambda_cl_{lambda_cl}")
    
    if not param_name_parts:
        current_param_name = "default"
    else:
        current_param_name = "_".join(param_name_parts)
    
    print("=" * 80)
    print(f"开始敏感度分析 - 参数: {current_param_name}")
    print(f"  spatial_k={spatial_k}, feature_k={feature_k}, diffusion_step={diffusion_step}")
    print(f"  alpha={alpha}, beta={beta}, lambda_cl={lambda_cl}, lambda_rec={lambda_rec}")
    print("=" * 80)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 复制数据，避免修改原始数据
    adata_omics1 = adata_omics1.copy()
    adata_omics2 = adata_omics2.copy()
    
    # ========================================================================
    # 1. 构建图（使用参数化的spatial_k和feature_k）
    # ========================================================================
    print(f"\n[1/5] 构建图 (spatial_k={spatial_k}, feature_k={feature_k})...")
    adata_omics1, adata_omics2 = construct_neighbor_graph(
        adata_omics1, adata_omics2,
        spatial_k=spatial_k,
        feature_k=feature_k
    )
    
    adj = adjacent_matrix_preprocessing(adata_omics1, adata_omics2)
    adj_spatial_omics1 = adj['adj_spatial_omics1'].to(device)
    adj_spatial_omics2 = adj['adj_spatial_omics2'].to(device)
    
    # ========================================================================
    # 2. 编码阶段（使用参数化的alpha和beta）
    # ========================================================================
    print(f"\n[2/5] 编码阶段 (alpha={alpha}, beta={beta})...")
    ae_model = encoder_ZINB(
        adata=adata_omics1,
        device=torch.device(device),
        epochs=300,
        dim_output=128,
        alpha=alpha,
        beta=beta
    )
    adata_omics1.obsm['emb_ZINB'], adj_mat = ae_model.train()
    
    seed = 2024
    seed_everything(seed)
    train_model(adata_omics1, adata_omics2, adj_spatial_omics1, adj_spatial_omics2, epochs=200)
    
    # ========================================================================
    # 3. 去噪阶段（使用参数化的diffusion_step）
    # ========================================================================
    print(f"\n[3/5] 去噪阶段 (diffusion_step={diffusion_step})...")
    # 对齐embeddings
    slices_omics1_spatial = adata_omics1.obsm['spatial']
    slices_omics2_spatial = adata_omics2.obsm['spatial']
    emb_latent_omics1 = adata_omics1.obsm['emb_ZINB']
    emb_latent_omics2 = adata_omics2.obsm['emb_latent_omics2']
    
    df_omics1 = pd.DataFrame(emb_latent_omics1, index=[tuple(coord) for coord in slices_omics1_spatial])
    df_omics2 = pd.DataFrame(emb_latent_omics2, index=[tuple(coord) for coord in slices_omics2_spatial])
    df_omics2_aligned = df_omics2.reindex(df_omics1.index)
    aligned_emb_latent_omics1 = df_omics1.to_numpy()
    aligned_emb_latent_omics2 = df_omics2_aligned.to_numpy()
    
    # Diffusion
    dataset = ConditionalDiffusionDataset(aligned_emb_latent_omics1, aligned_emb_latent_omics2)
    seed = 42
    seed_everything(seed)
    
    com_mtx = run_diff(
        dataset,
        k=3,
        batch_size=512,
        hidden_size=512,
        learning_rate=1e-3,
        num_epoch=1000,
        diffusion_step=diffusion_step,  # 使用参数
        depth=6,
        head=16,
        device=device,
        classes=6,
        patience=40,
        bias=1
    )
    
    adata_omics1.obsm['denoise_emb'] = com_mtx
    
    # 聚类和评估
    tool = 'mclust'
    clustering(adata_omics1, key='denoise_emb', add_key='Denoise', n_clusters=5, end=1.2, method=tool, use_pca=True)
    
    # 计算去噪阶段的指标
    denoise_metrics = calculate_metrics(adata_omics1.obs['ground_truth'], adata_omics1.obs['Denoise'])
    print_metrics(denoise_metrics, '去噪结果 (ADT_denoised)')
    
    # 保存去噪结果（追加到同一类参数的文件中）
    denoise_results = {
        'Parameter': [current_param_name],
        'spatial_k': [spatial_k],
        'feature_k': [feature_k],
        'diffusion_step': [diffusion_step],
        'alpha': [alpha],
        'beta': [beta],
        'lambda_cl': [lambda_cl],
        'lambda_rec': [lambda_rec],
        **denoise_metrics
    }
    denoise_df = pd.DataFrame(denoise_results)
    denoise_csv_file = os.path.join(RESULT_DIR, f"denoise_results_{param_type}.csv")
    
    # 追加模式：如果文件存在则追加，否则创建新文件
    if os.path.exists(denoise_csv_file):
        denoise_df.to_csv(denoise_csv_file, mode='a', header=False, index=False)
        print(f"去噪结果已追加到: {denoise_csv_file}")
    else:
        denoise_df.to_csv(denoise_csv_file, index=False)
        print(f"去噪结果已保存到: {denoise_csv_file}")
    
    # ========================================================================
    # 4. 整合阶段（使用参数化的lambda_cl和lambda_rec）
    # ========================================================================
    print(f"\n[4/5] 整合阶段 (lambda_cl={lambda_cl}, lambda_rec={lambda_rec})...")
    adata1 = adata_omics1.copy()
    adata2 = adata_omics2.copy()
    
    adata1.obsm['feat'] = adata1.obsm['denoise_emb']
    adata2.obsm['feat'] = adata2.obsm['emb_latent_omics2']
    adata1, adata2 = construct_neighbor_graph(adata1, adata2, spatial_k=spatial_k, feature_k=feature_k)
    
    adj = adjacent_matrix_preprocessing(adata1, adata2)
    adj_spatial_omics1 = adj['adj_spatial_omics1'].to(device)
    adj_spatial_omics2 = adj['adj_spatial_omics2'].to(device)
    adj_feature_omics1 = adj['adj_feature_omics1'].to(device)
    adj_feature_omics2 = adj['adj_feature_omics2'].to(device)
    
    features_omics1 = torch.FloatTensor(adata1.obsm['feat'].copy()).to(device)
    features_omics2 = torch.FloatTensor(adata2.obsm['feat'].copy()).to(device)
    
    seed = 2025
    seed_everything(seed)
    
    result = train_and_infer(
        features_omics1=features_omics1,
        features_omics2=features_omics2,
        adj_spatial_omics1=adj_spatial_omics1,
        adj_feature_omics1=adj_feature_omics1,
        adj_spatial_omics2=adj_spatial_omics2,
        adj_feature_omics2=adj_feature_omics2,
        device=device,
        epochs=100,
        lambda_cl=lambda_cl,  # 使用参数
        lambda_rec=lambda_rec  # 使用参数
    )
    
    adata = adata1.copy()
    candies_emb = result['emb_latent_combined'].detach().cpu().numpy().copy()
    adata.obsm['CANDIES'] = candies_emb
    
    # 聚类和评估
    tool = 'mclust'
    clustering(adata, key='CANDIES', add_key='CANDIES', n_clusters=5, end=1, method=tool, use_pca=True)
    
    # 计算整合阶段的指标
    integration_metrics = calculate_metrics(adata.obs['ground_truth'], adata.obs['CANDIES'])
    print_metrics(integration_metrics, '整合结果 (CANDIES)')
    
    # 保存整合结果（追加到同一类参数的文件中）
    integration_results = {
        'Parameter': [current_param_name],
        'spatial_k': [spatial_k],
        'feature_k': [feature_k],
        'diffusion_step': [diffusion_step],
        'alpha': [alpha],
        'beta': [beta],
        'lambda_cl': [lambda_cl],
        'lambda_rec': [lambda_rec],
        'Method': ['CANDIES(ADT_denoise)'],
        **integration_metrics
    }
    integration_df = pd.DataFrame(integration_results)
    integration_csv_file = os.path.join(RESULT_DIR, f"integration_results_{param_type}.csv")
    
    # 追加模式：如果文件存在则追加，否则创建新文件
    if os.path.exists(integration_csv_file):
        integration_df.to_csv(integration_csv_file, mode='a', header=False, index=False)
        print(f"整合结果已追加到: {integration_csv_file}")
    else:
        integration_df.to_csv(integration_csv_file, index=False)
        print(f"整合结果已保存到: {integration_csv_file}")
    
    # ========================================================================
    # 5. 完成
    # ========================================================================
    print("\n" + "=" * 80)
    print(f"敏感度分析完成 - 参数: {current_param_name}")
    print("=" * 80)
    print(f"\n结果文件:")
    print(f"  去噪结果: {denoise_csv_file}")
    print(f"  整合结果: {integration_csv_file}")
    
    return current_param_name


def main():
    """主函数：遍历所有参数组合"""
    print("=" * 80)
    print("敏感度分析 - 批量运行所有参数组合")
    print("=" * 80)
    
    # 一次性加载和预处理数据
    adata_omics1, adata_omics2 = load_and_preprocess_data()
    
    if not RUN_ALL_PARAMS:
        # 只运行默认参数
        print("\n只运行默认参数...")
        run_single_experiment(
            adata_omics1, adata_omics2,
            DEFAULT_SPATIAL_K, DEFAULT_FEATURE_K, DEFAULT_DIFFUSION_STEP,
            DEFAULT_ALPHA, DEFAULT_BETA, DEFAULT_LAMBDA_CL, DEFAULT_LAMBDA_REC
        )
    else:
        # 遍历所有参数（每次只改一个参数）
        total_experiments = sum(len(v) for v in PARAM_RANGES.values()) - len(PARAM_RANGES) + 1  # 减去重复的默认值
        current_exp = 0
        
        # 2. 遍历 spatial_k
        for spatial_k in PARAM_RANGES['spatial_k']:
            if spatial_k == DEFAULT_SPATIAL_K:
                continue
            current_exp += 1
            print(f"\n[{current_exp}/{total_experiments}] 测试 spatial_k={spatial_k}...")
            run_single_experiment(
                adata_omics1, adata_omics2,
                spatial_k, DEFAULT_FEATURE_K, DEFAULT_DIFFUSION_STEP,
                DEFAULT_ALPHA, DEFAULT_BETA, DEFAULT_LAMBDA_CL, DEFAULT_LAMBDA_REC
            )
        
        # 3. 遍历 feature_k
        for feature_k in PARAM_RANGES['feature_k']:
            if feature_k == DEFAULT_FEATURE_K:
                continue
            current_exp += 1
            print(f"\n[{current_exp}/{total_experiments}] 测试 feature_k={feature_k}...")
            run_single_experiment(
                adata_omics1, adata_omics2,
                DEFAULT_SPATIAL_K, feature_k, DEFAULT_DIFFUSION_STEP,
                DEFAULT_ALPHA, DEFAULT_BETA, DEFAULT_LAMBDA_CL, DEFAULT_LAMBDA_REC
            )
        
        # 4. 遍历 diffusion_step
        for diffusion_step in PARAM_RANGES['diffusion_step']:
            if diffusion_step == DEFAULT_DIFFUSION_STEP:
                continue
            current_exp += 1
            print(f"\n[{current_exp}/{total_experiments}] 测试 diffusion_step={diffusion_step}...")
            run_single_experiment(
                adata_omics1, adata_omics2,
                DEFAULT_SPATIAL_K, DEFAULT_FEATURE_K, diffusion_step,
                DEFAULT_ALPHA, DEFAULT_BETA, DEFAULT_LAMBDA_CL, DEFAULT_LAMBDA_REC
            )
        
        # 5. 遍历 alpha
        for alpha in PARAM_RANGES['alpha']:
            if alpha == DEFAULT_ALPHA:
                continue
            current_exp += 1
            print(f"\n[{current_exp}/{total_experiments}] 测试 alpha={alpha}...")
            run_single_experiment(
                adata_omics1, adata_omics2,
                DEFAULT_SPATIAL_K, DEFAULT_FEATURE_K, DEFAULT_DIFFUSION_STEP,
                alpha, DEFAULT_BETA, DEFAULT_LAMBDA_CL, DEFAULT_LAMBDA_REC
            )
        
        # 6. 遍历 beta
        for beta in PARAM_RANGES['beta']:
            if beta == DEFAULT_BETA:
                continue
            current_exp += 1
            print(f"\n[{current_exp}/{total_experiments}] 测试 beta={beta}...")
            run_single_experiment(
                adata_omics1, adata_omics2,
                DEFAULT_SPATIAL_K, DEFAULT_FEATURE_K, DEFAULT_DIFFUSION_STEP,
                DEFAULT_ALPHA, beta, DEFAULT_LAMBDA_CL, DEFAULT_LAMBDA_REC
            )
        
        # 7. 遍历 lambda_cl（lambda_rec自动计算为1-lambda_cl）
        for lambda_cl in PARAM_RANGES['lambda_cl']:
            if lambda_cl == DEFAULT_LAMBDA_CL:
                continue
            lambda_rec = 1.0 - lambda_cl
            current_exp += 1
            print(f"\n[{current_exp}/{total_experiments}] 测试 lambda_cl={lambda_cl}, lambda_rec={lambda_rec}...")
            run_single_experiment(
                adata_omics1, adata_omics2,
                DEFAULT_SPATIAL_K, DEFAULT_FEATURE_K, DEFAULT_DIFFUSION_STEP,
                DEFAULT_ALPHA, DEFAULT_BETA, lambda_cl, lambda_rec
            )
        
        print("\n" + "=" * 80)
        print(f"所有实验完成！共运行 {current_exp} 个参数组合")
        print("=" * 80)


if __name__ == "__main__":
    main()
