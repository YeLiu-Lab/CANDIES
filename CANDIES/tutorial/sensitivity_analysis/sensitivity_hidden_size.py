"""
敏感性分析：测试不同 hidden_size 对模型性能的影响

条件模态MLP的输出维度 = hidden_size × 2
通过改变 hidden_size，测试不同条件嵌入维度的影响
计算评估指标（ARI, NMI等）并保存到CSV
"""
import sys
import os
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from codes_v2.train_diff import ConditionalDiffusionDataset, run_diff
from codes_v2.DiTs import seed_everything
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score, 
    adjusted_mutual_info_score, v_measure_score,
    homogeneity_score, completeness_score, fowlkes_mallows_score
)

def run_leiden(adata1, n_cluster, use_rep="embeddings", key_added="Nleiden", range_min=0, range_max=3, max_steps=30, tolerance=0):
    """运行Leiden聚类，找到指定数量的聚类"""
    adata = adata1.copy()
    sc.pp.neighbors(adata, use_rep=use_rep)
    this_step = 0
    this_min = float(range_min)
    this_max = float(range_max)
    while this_step < max_steps:
        this_resolution = this_min + ((this_max-this_min)/2)
        sc.tl.leiden(adata, resolution=this_resolution)
        this_clusters = adata.obs['leiden'].nunique()

        if this_clusters > n_cluster+tolerance:
            this_max = this_resolution
        elif this_clusters < n_cluster-tolerance:
            this_min = this_resolution
        else:
            print("Succeed to find %d clusters at resolution %.3f"%(n_cluster, this_resolution))
            adata1.obs[key_added] = adata.obs["leiden"]
            return adata1
        
        this_step += 1
    
    print('Cannot find the number of clusters')
    adata1.obs[key_added] = adata.obs["leiden"]
    return adata1

def calculate_metrics(ground_truth, predicted):
    """计算所有评估指标"""
    metrics = {
        'ARI': adjusted_rand_score(ground_truth, predicted),
        'NMI': normalized_mutual_info_score(ground_truth, predicted),
        'AMI': adjusted_mutual_info_score(ground_truth, predicted),
        'V_measure': v_measure_score(ground_truth, predicted),
        'Homogeneity': homogeneity_score(ground_truth, predicted),
        'Completeness': completeness_score(ground_truth, predicted),
        'FMI': fowlkes_mallows_score(ground_truth, predicted)
    }
    return metrics

def sensitivity_analysis(
    aligned_emb_latent_omics1,
    aligned_emb_latent_omics2,
    adata_omics1,
    hidden_sizes=[128, 192, 256, 320, 384],
    output_dir='sensitivity_analysis/results',
    n_clusters=5,  # 聚类数量（根据你的ground_truth调整）
    k=3,
    batch_size=512,
    learning_rate=1e-3,
    num_epoch=1000,
    diffusion_step=800,
    depth=6,
    head=16,
    device='cuda:0',
    classes=6,
    patience=40,
    bias=0.5
):
    """
    运行敏感性分析
    
    参数:
    - aligned_emb_latent_omics1: 对齐后的模态1嵌入
    - aligned_emb_latent_omics2: 对齐后的模态2嵌入
    - adata_omics1: 原始数据（用于保存结果和计算指标）
    - hidden_sizes: 要测试的隐空间维度列表
    - output_dir: 结果保存目录
    - n_clusters: 聚类数量
    - 其他参数: 训练参数
    """
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查是否有ground_truth
    has_ground_truth = 'ground_truth' in adata_omics1.obs
    
    # 存储所有结果
    all_results = []
    
    print(f"\n开始敏感性分析，测试 {len(hidden_sizes)} 个不同的隐空间维度...")
    print(f"隐空间维度列表: {hidden_sizes}")
    print(f"对应的条件模态MLP输出维度: {[h*2 for h in hidden_sizes]}")
    if has_ground_truth:
        print(f"将计算评估指标（需要ground_truth）")
    print("=" * 80)
    
    for i, hidden_size in enumerate(hidden_sizes):
        print(f"\n[{i+1}/{len(hidden_sizes)}] 测试 hidden_size = {hidden_size} (条件维度 = {hidden_size * 2})")
        print("-" * 80)
        
        try:
            # 创建数据集
            dataset = ConditionalDiffusionDataset(aligned_emb_latent_omics1, aligned_emb_latent_omics2)
            
            # 设置随机种子
            seed = 2024
            seed_everything(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            
            # 运行训练和推理
            com_mtx = run_diff(
                dataset,
                k=k,
                batch_size=batch_size,
                hidden_size=hidden_size,  # 关键参数：改变隐空间维度
                learning_rate=learning_rate,
                num_epoch=num_epoch,
                diffusion_step=diffusion_step,
                depth=depth,
                head=head,
                device=device,
                classes=classes,
                patience=patience,
                bias=bias
            )
            
            # 保存去噪后的嵌入
            adata_result = adata_omics1.copy()
            adata_result.obsm['denoise_emb'] = com_mtx
            
            # 进行聚类
            print(f"进行聚类（n_clusters={n_clusters}）...")
            adata_result = run_leiden(
                adata_result, 
                n_cluster=n_clusters, 
                use_rep="denoise_emb", 
                key_added="Denoise"
            )
            
            # 计算评估指标
            result_dict = {
                'hidden_size': hidden_size,
                'cond_mlp_dim': hidden_size * 2
            }
            
            if has_ground_truth:
                ground_truth = adata_result.obs['ground_truth']
                predicted = adata_result.obs['Denoise']
                
                metrics = calculate_metrics(ground_truth, predicted)
                result_dict.update(metrics)
                
                print(f"评估指标:")
                print(f"  ARI: {metrics['ARI']:.6f}")
                print(f"  NMI: {metrics['NMI']:.6f}")
                print(f"  AMI: {metrics['AMI']:.6f}")
            else:
                print("警告: 未找到ground_truth，跳过评估指标计算")
            
            # 保存结果文件
            result_file = os.path.join(output_dir, f'denoised_hidden{hidden_size}.h5ad')
            adata_result.write(result_file)
            result_dict['result_file'] = result_file
            result_dict['status'] = 'success'
            
            print(f"✓ 完成！结果已保存到: {result_file}")
            
            all_results.append(result_dict)
            
        except Exception as e:
            print(f"✗ 错误: {str(e)}")
            import traceback
            traceback.print_exc()
            all_results.append({
                'hidden_size': hidden_size,
                'cond_mlp_dim': hidden_size * 2,
                'status': f'error: {str(e)}'
            })
    
    # 保存结果到CSV
    results_df = pd.DataFrame(all_results)
    csv_file = os.path.join(output_dir, 'sensitivity_analysis_results.csv')
    results_df.to_csv(csv_file, index=False)
    
    print("\n" + "=" * 80)
    print("敏感性分析完成！")
    print(f"结果已保存到: {csv_file}")
    print("=" * 80)
    
    # 显示结果摘要
    if has_ground_truth:
        print("\n结果摘要（按ARI排序）:")
        print(results_df[['hidden_size', 'cond_mlp_dim', 'ARI', 'NMI']].to_string(index=False))
    
    return results_df
