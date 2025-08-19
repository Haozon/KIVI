import torch
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from models.llama_kivi import LlamaForCausalLM_KIVI
from transformers import LlamaConfig, AutoTokenizer, LlamaForCausalLM
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def analyze_kv_distribution():
    """复现论文Figure 2: KV cache分布分析"""
    
    # 加载模型
    config = LlamaConfig.from_pretrained("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hdpmlpserving/LLMs/LLMs_HF/llama-2-7b")
    model = LlamaForCausalLM.from_pretrained("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hdpmlpserving/LLMs/LLMs_HF/llama-2-7b")
    tokenizer = AutoTokenizer.from_pretrained("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hdpmlpserving/LLMs/LLMs_HF/llama-2-7b")
    
    # 准备输入 - 使用更长的序列来获得更好的可视化效果
    text = "The quick brown fox jumps over the lazy dog. " * 100
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    
    # 收集KV cache
    kv_caches = []
    
    # 推理获取KV cache
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, output_attentions=False)
    
    # 从模型输出获取KV cache
    if hasattr(outputs, 'past_key_values') and outputs.past_key_values is not None:
        past_kv = outputs.past_key_values
        print(f"Found past_key_values of type: {type(past_kv)}")
        
        # 处理不同类型的past_key_values
        if hasattr(past_kv, 'key_cache') and hasattr(past_kv, 'value_cache'):
            # 新版本Cache类
            for i in range(len(past_kv.key_cache)):
                kv_caches.append({
                    'layer': i,
                    'key_states': past_kv.key_cache[i],
                    'value_states': past_kv.value_cache[i]
                })
        elif isinstance(past_kv, (list, tuple)):
            # 旧版本格式：list of (key, value) tuples
            for i, layer_kv in enumerate(past_kv):
                if isinstance(layer_kv, tuple) and len(layer_kv) >= 2:
                    kv_caches.append({
                        'layer': i,
                        'key_states': layer_kv[0],
                        'value_states': layer_kv[1]
                    })
        
        print(f"Extracted {len(kv_caches)} KV caches from model outputs")
    
    if len(kv_caches) == 0:
        print("No KV caches collected.")
        return
    
    # 创建3D可视化 - 类似论文中的图
    def plot_3d_kv_cache(cache_data, layer_idx, head_idx=0, cache_type='key'):
        """绘制3D KV cache分布图"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        if cache_type == 'key':
            data = cache_data['key_states']
            title_prefix = "Key Cache"
        else:
            data = cache_data['value_states']
            title_prefix = "Value Cache"
        
        # 数据形状: [batch_size, num_heads, seq_len, head_dim]
        cache_tensor = data[0, head_idx].float().cpu().numpy()  # [seq_len, head_dim]
        seq_len, head_dim = cache_tensor.shape
        
        # 创建网格
        X, Y = np.meshgrid(np.arange(seq_len), np.arange(head_dim))
        Z = np.abs(cache_tensor).T  # 转置以匹配网格
        
        # 3D表面图 - 使用更好的颜色映射
        surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.9, 
                              linewidth=0, antialiased=True, rcount=50, ccount=50)
        
        # 设置标签和标题
        ax.set_xlabel('Token', fontsize=12)
        ax.set_ylabel('Channel', fontsize=12)
        ax.set_zlabel('Absolute Value', fontsize=12)
        ax.set_title(f'Llama-2-7B Layer {layer_idx+1}\nHead {head_idx} {title_prefix}', fontsize=14)
        
        # 调整视角 - 更像论文中的角度
        ax.view_init(elev=30, azim=-60)  # 改变倾斜角度
        
        # 设置坐标轴范围和刻度
        ax.set_xlim(0, seq_len)
        ax.set_ylim(0, head_dim)
        
        # 移除颜色条以保持简洁
        # fig.colorbar(surf, shrink=0.5, aspect=5)
        
        return fig, ax
    
    # 绘制多个层和多个head的KV cache分布
    layers_to_plot = [0, 7, 15, 23, 30]  # 绘制更多层：第1, 8, 16, 24, 31层
    layers_to_plot = [i for i in layers_to_plot if i < len(kv_caches)]
    heads_to_plot = [0, 1, 2, 3]  # 绘制前4个head
    
    print(f"Will plot {len(layers_to_plot)} layers and {len(heads_to_plot)} heads")
    
    # 为每个层创建多head对比图
    for layer_idx in layers_to_plot:
        cache = kv_caches[layer_idx]
        
        if cache['key_states'] is not None:
            num_heads = cache['key_states'].shape[1]
            actual_heads = [h for h in heads_to_plot if h < num_heads]
            
            print(f"Layer {layer_idx+1} - Total heads: {num_heads}, plotting heads: {actual_heads}")
            
            # 创建2x4的子图布局 (Key和Value各一行，4个head)
            fig, axes = plt.subplots(2, len(actual_heads), figsize=(5*len(actual_heads), 10), 
                                   subplot_kw={'projection': '3d'})
            
            if len(actual_heads) == 1:
                axes = axes.reshape(2, 1)
            
            for head_idx, actual_head in enumerate(actual_heads):
                # Key Cache
                cache_tensor = cache['key_states'][0, actual_head].float().cpu().numpy()
                seq_len, head_dim = cache_tensor.shape
                X, Y = np.meshgrid(np.arange(seq_len), np.arange(head_dim))
                Z = np.abs(cache_tensor).T
                
                surf = axes[0, head_idx].plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.9, 
                                                    linewidth=0, antialiased=True, rcount=40, ccount=40)
                axes[0, head_idx].set_xlabel('Token')
                axes[0, head_idx].set_ylabel('Channel')
                axes[0, head_idx].set_zlabel('Absolute Value')
                axes[0, head_idx].set_title(f'Llama-2-7B Layer {layer_idx+1}\nHead {actual_head} Key Cache')
                axes[0, head_idx].view_init(elev=30, azim=-60)
                
                # Value Cache
                cache_tensor = cache['value_states'][0, actual_head].float().cpu().numpy()
                Z = np.abs(cache_tensor).T
                
                surf = axes[1, head_idx].plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.9, 
                                                    linewidth=0, antialiased=True, rcount=40, ccount=40)
                axes[1, head_idx].set_xlabel('Token')
                axes[1, head_idx].set_ylabel('Channel')
                axes[1, head_idx].set_zlabel('Absolute Value')
                axes[1, head_idx].set_title(f'Llama-2-7B Layer {layer_idx+1}\nHead {actual_head} Value Cache')
                axes[1, head_idx].view_init(elev=30, azim=-60)
            
            plt.tight_layout()
            plt.savefig(f'analysis/layer_{layer_idx+1}_multi_head_kv_cache.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    # 创建层间对比图 - 固定head 0
    if len(layers_to_plot) >= 4:
        # Key Cache层间对比
        fig, axes = plt.subplots(1, len(layers_to_plot), figsize=(5*len(layers_to_plot), 6), 
                               subplot_kw={'projection': '3d'})
        
        if len(layers_to_plot) == 1:
            axes = [axes]
        
        for i, layer_idx in enumerate(layers_to_plot):
            cache = kv_caches[layer_idx]
            if cache['key_states'] is not None:
                cache_tensor = cache['key_states'][0, 0].float().cpu().numpy()  # Head 0
                seq_len, head_dim = cache_tensor.shape
                X, Y = np.meshgrid(np.arange(seq_len), np.arange(head_dim))
                Z = np.abs(cache_tensor).T
                
                surf = axes[i].plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.9, 
                                          linewidth=0, antialiased=True, rcount=40, ccount=40)
                axes[i].set_xlabel('Token')
                axes[i].set_ylabel('Channel')
                axes[i].set_zlabel('Absolute Value')
                axes[i].set_title(f'Llama-2-7B Layer {layer_idx+1}\nHead 0 Key Cache')
                axes[i].view_init(elev=30, azim=-60)
        
        plt.tight_layout()
        plt.savefig('analysis/key_cache_layer_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Value Cache层间对比
        fig, axes = plt.subplots(1, len(layers_to_plot), figsize=(5*len(layers_to_plot), 6), 
                               subplot_kw={'projection': '3d'})
        
        if len(layers_to_plot) == 1:
            axes = [axes]
        
        for i, layer_idx in enumerate(layers_to_plot):
            cache = kv_caches[layer_idx]
            if cache['value_states'] is not None:
                cache_tensor = cache['value_states'][0, 0].float().cpu().numpy()  # Head 0
                seq_len, head_dim = cache_tensor.shape
                X, Y = np.meshgrid(np.arange(seq_len), np.arange(head_dim))
                Z = np.abs(cache_tensor).T
                
                surf = axes[i].plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.9, 
                                          linewidth=0, antialiased=True, rcount=40, ccount=40)
                axes[i].set_xlabel('Token')
                axes[i].set_ylabel('Channel')
                axes[i].set_zlabel('Absolute Value')
                axes[i].set_title(f'Llama-2-7B Layer {layer_idx+1}\nHead 0 Value Cache')
                axes[i].view_init(elev=30, azim=-60)
        
        plt.tight_layout()
        plt.savefig('analysis/value_cache_layer_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 创建head间对比图 - 固定某一层（如第16层）
    if len(layers_to_plot) > 2:
        middle_layer = layers_to_plot[len(layers_to_plot)//2]  # 选择中间层
        cache = kv_caches[middle_layer]
        
        if cache['key_states'] is not None:
            num_heads = cache['key_states'].shape[1]
            heads_to_compare = list(range(min(8, num_heads)))  # 最多比较8个head
            
            # Key Cache head间对比
            fig, axes = plt.subplots(2, 4, figsize=(20, 10), subplot_kw={'projection': '3d'})
            axes = axes.flatten()
            
            for head_idx in heads_to_compare:
                cache_tensor = cache['key_states'][0, head_idx].float().cpu().numpy()
                seq_len, head_dim = cache_tensor.shape
                X, Y = np.meshgrid(np.arange(seq_len), np.arange(head_dim))
                Z = np.abs(cache_tensor).T
                
                surf = axes[head_idx].plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.9, 
                                                 linewidth=0, antialiased=True, rcount=30, ccount=30)
                axes[head_idx].set_xlabel('Token')
                axes[head_idx].set_ylabel('Channel')
                axes[head_idx].set_zlabel('Absolute Value')
                axes[head_idx].set_title(f'Layer {middle_layer+1} Head {head_idx}\nKey Cache')
                axes[head_idx].view_init(elev=30, azim=-60)
            
            plt.tight_layout()
            plt.savefig(f'analysis/layer_{middle_layer+1}_key_cache_head_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Value Cache head间对比
            fig, axes = plt.subplots(2, 4, figsize=(20, 10), subplot_kw={'projection': '3d'})
            axes = axes.flatten()
            
            for head_idx in heads_to_compare:
                cache_tensor = cache['value_states'][0, head_idx].float().cpu().numpy()
                seq_len, head_dim = cache_tensor.shape
                X, Y = np.meshgrid(np.arange(seq_len), np.arange(head_dim))
                Z = np.abs(cache_tensor).T
                
                surf = axes[head_idx].plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.9, 
                                                 linewidth=0, antialiased=True, rcount=30, ccount=30)
                axes[head_idx].set_xlabel('Token')
                axes[head_idx].set_ylabel('Channel')
                axes[head_idx].set_zlabel('Absolute Value')
                axes[head_idx].set_title(f'Layer {middle_layer+1} Head {head_idx}\nValue Cache')
                axes[head_idx].view_init(elev=30, azim=-60)
            
            plt.tight_layout()
            plt.savefig(f'analysis/layer_{middle_layer+1}_value_cache_head_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    # 统计分析 - 按层和head分析
    print("\nDetailed KV Cache Distribution Analysis:")
    for layer_idx in layers_to_plot[:3]:  # 分析前3层
        cache = kv_caches[layer_idx]
        if cache['key_states'] is not None:
            num_heads = cache['key_states'].shape[1]
            print(f"\nLayer {layer_idx+1} (Total {num_heads} heads):")
            
            for head_idx in range(min(4, num_heads)):  # 分析前4个head
                key_cache = cache['key_states'][0, head_idx].float().cpu()
                value_cache = cache['value_states'][0, head_idx].float().cpu()
                
                # Key cache分析
                key_channel_var = torch.var(key_cache, dim=0)  # 每个channel的方差
                key_token_var = torch.var(key_cache, dim=1)    # 每个token的方差
                
                # Value cache分析
                value_channel_var = torch.var(value_cache, dim=0)
                value_token_var = torch.var(value_cache, dim=1)
                
                print(f"  Head {head_idx}:")
                print(f"    Key - Channel var ratio: {(key_channel_var.max()/key_channel_var.min()):.2f}, "
                      f"Token var ratio: {(key_token_var.max()/key_token_var.min()):.2f}")
                print(f"    Value - Channel var ratio: {(value_channel_var.max()/value_channel_var.min()):.2f}, "
                      f"Token var ratio: {(value_token_var.max()/value_token_var.min()):.2f}")

if __name__ == "__main__":
    analyze_kv_distribution()