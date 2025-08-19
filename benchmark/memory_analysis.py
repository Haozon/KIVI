import torch
import psutil
import GPUtil
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class MemoryProfiler:
    def __init__(self):
        self.baseline_gpu_mem = None
        self.baseline_cpu_mem = None
    
    def start_profiling(self):
        """开始内存profiling"""
        torch.cuda.reset_peak_memory_stats()
        self.baseline_gpu_mem = torch.cuda.memory_allocated()
        self.baseline_cpu_mem = psutil.virtual_memory().used
        
    def get_memory_usage(self):
        """获取当前内存使用"""
        gpu_mem = torch.cuda.memory_allocated()
        gpu_peak = torch.cuda.max_memory_allocated()
        cpu_mem = psutil.virtual_memory().used
        
        return {
            'gpu_current': gpu_mem / 1024**3,  # GB
            'gpu_peak': gpu_peak / 1024**3,
            'gpu_delta': (gpu_mem - self.baseline_gpu_mem) / 1024**3,
            'cpu_delta': (cpu_mem - self.baseline_cpu_mem) / 1024**3
        }

def benchmark_memory_usage():
    """对比KIVI与baseline的内存使用"""
    
    # 确保输出目录存在
    os.makedirs('benchmark', exist_ok=True)
    
    profiler = MemoryProfiler()
    
    # 测试配置
    configs = [
        {'k_bits': 16, 'v_bits': 16, 'name': 'Baseline (FP16)'},
        {'k_bits': 4, 'v_bits': 4, 'name': 'KIVI-4bit'},
        {'k_bits': 2, 'v_bits': 2, 'name': 'KIVI-2bit'}
    ]
    
    results = {}
    model_path = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hdpmlpserving/LLMs/LLMs_HF/llama-2-7b"
    
    for config in configs:
        print(f"\n=== Testing {config['name']} ===")
        
        profiler.start_profiling()
        
        # 加载模型
        if config['k_bits'] < 16:
            try:
                from models.llama_kivi import LlamaForCausalLM_KIVI
                
                model_config = LlamaConfig.from_pretrained(model_path)
                model_config.k_bits = config['k_bits']
                model_config.v_bits = config['v_bits']
                model_config.group_size = 32
                model_config.residual_length = 128
                
                model = LlamaForCausalLM_KIVI.from_pretrained(
                    model_path,
                    config=model_config,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            except ImportError:
                print(f"KIVI model not available, skipping {config['name']}")
                continue
        else:
            model = LlamaForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        
        mem_after_load = profiler.get_memory_usage()
        print(f"Memory after loading: {mem_after_load['gpu_current']:.2f}GB")
        
        # 生成长序列测试内存增长
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        batch_sizes = [1, 2, 4]  # 减少batch size避免OOM
        seq_lengths = [512, 1024]  # 减少序列长度
        
        for batch_size in batch_sizes:
            for seq_len in seq_lengths:
                try:
                    # 创建输入
                    input_text = "Hello world! This is a test sentence. " * (seq_len // 20)
                    inputs = tokenizer(
                        [input_text] * batch_size,
                        return_tensors="pt",
                        max_length=seq_len,
                        truncation=True,
                        padding=True
                    )
                    
                    # 移动到GPU
                    if torch.cuda.is_available():
                        inputs = {k: v.to('cuda') for k, v in inputs.items()}
                    
                    # 推理
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=50,  # 减少生成长度
                            do_sample=False,
                            use_cache=True,
                            pad_token_id=tokenizer.eos_token_id
                        )
                    
                    mem_usage = profiler.get_memory_usage()
                    
                    key = f"bs{batch_size}_seq{seq_len}"
                    if config['name'] not in results:
                        results[config['name']] = {}
                    
                    results[config['name']][key] = {
                        'gpu_peak': mem_usage['gpu_peak'],
                        'gpu_current': mem_usage['gpu_current']
                    }
                    
                    print(f"BS={batch_size}, Seq={seq_len}: "
                          f"Peak GPU={mem_usage['gpu_peak']:.2f}GB, "
                          f"Current GPU={mem_usage['gpu_current']:.2f}GB")
                    
                except Exception as e:
                    print(f"Error at BS={batch_size}, Seq={seq_len}: {e}")
                    break
                
                # 清理中间结果
                torch.cuda.empty_cache()
        
        # 清理模型
        del model
        torch.cuda.empty_cache()
        print(f"Finished testing {config['name']}")
    
    # 分析结果
    if results:
        analyze_memory_results(results)
    else:
        print("No results to analyze")

def analyze_memory_results(results):
    """分析内存使用结果"""
    
    # 转换为DataFrame
    data = []
    for model_name, model_results in results.items():
        for config, metrics in model_results.items():
            bs, seq = config.replace('bs', '').replace('_seq', ' ').split()
            data.append({
                'Model': model_name,
                'Batch Size': int(bs),
                'Sequence Length': int(seq),
                'Peak GPU (GB)': metrics['gpu_peak'],
                'Current GPU (GB)': metrics['gpu_current']
            })
    
    if not data:
        print("No data to analyze")
        return
        
    df = pd.DataFrame(data)
    print("\n=== Memory Usage Summary ===")
    print(df.to_string(index=False))
    
    # 绘制对比图
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 不同序列长度的内存使用
    for model in df['Model'].unique():
        model_data = df[df['Model'] == model]
        if not model_data.empty:
            seq_mem = model_data.groupby('Sequence Length')['Peak GPU (GB)'].mean()
            axes[0].plot(seq_mem.index, seq_mem.values, marker='o', label=model)
    
    axes[0].set_xlabel('Sequence Length')
    axes[0].set_ylabel('Peak GPU Memory (GB)')
    axes[0].set_title('Memory vs Sequence Length')
    axes[0].legend()
    axes[0].grid(True)
    
    # 不同batch size的内存使用
    for model in df['Model'].unique():
        model_data = df[df['Model'] == model]
        if not model_data.empty:
            batch_mem = model_data.groupby('Batch Size')['Peak GPU (GB)'].mean()
            axes[1].plot(batch_mem.index, batch_mem.values, marker='s', label=model)
    
    axes[1].set_xlabel('Batch Size')
    axes[1].set_ylabel('Peak GPU Memory (GB)')
    axes[1].set_title('Memory vs Batch Size')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('benchmark/memory_comparison.png', dpi=300, bbox_inches='tight')
    print("Memory comparison plot saved to benchmark/memory_comparison.png")
    
    # 计算内存节省比例
    baseline_data = df[df['Model'].str.contains('Baseline')]
    kivi_2bit_data = df[df['Model'].str.contains('2bit')]
    
    if not baseline_data.empty and not kivi_2bit_data.empty:
        memory_savings = {}
        for _, row in baseline_data.iterrows():
            config = f"bs{row['Batch Size']}_seq{row['Sequence Length']}"
            baseline_mem = row['Peak GPU (GB)']
            
            kivi_row = kivi_2bit_data[
                (kivi_2bit_data['Batch Size'] == row['Batch Size']) & 
                (kivi_2bit_data['Sequence Length'] == row['Sequence Length'])
            ]
            
            if not kivi_row.empty:
                kivi_mem = kivi_row.iloc[0]['Peak GPU (GB)']
                savings = (baseline_mem - kivi_mem) / baseline_mem * 100
                memory_savings[config] = savings
        
        print("\n=== Memory Savings Analysis ===")
        for config, savings in memory_savings.items():
            print(f"{config}: {savings:.1f}% memory reduction")
    
    # 保存结果到CSV
    df.to_csv('benchmark/memory_results.csv', index=False)
    print("Results saved to benchmark/memory_results.csv")

def main():
    """主函数"""
    print("Starting Memory Benchmark Analysis")
    print("=" * 50)
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        print("CUDA is not available. This benchmark requires GPU.")
        return
    
    print(f"Using GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    try:
        benchmark_memory_usage()
        print("\nBenchmark completed successfully!")
    except Exception as e:
        print(f"Benchmark failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()