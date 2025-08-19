import time
import torch
import numpy as np
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

def benchmark_throughput():
    """测试不同配置下的推理吞吐量"""
    
    configs = [
        {'bits': 16, 'name': 'Baseline', 'model_class': 'transformers'},
        {'bits': 4, 'name': 'KIVI-4bit', 'model_class': 'kivi'},
        {'bits': 2, 'name': 'KIVI-2bit', 'model_class': 'kivi'}
    ]
    
    test_cases = [
        {'batch_size': 1, 'input_len': 512, 'output_len': 128},
        {'batch_size': 4, 'input_len': 512, 'output_len': 128},
        {'batch_size': 8, 'input_len': 1024, 'output_len': 256},
        {'batch_size': 16, 'input_len': 2048, 'output_len': 512}
    ]
    
    results = {}
    
    for config in configs:
        print(f"\n=== Testing {config['name']} ===")
        
        # 加载模型
        model = load_model_by_config(config)
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        
        config_results = {}
        
        for test_case in test_cases:
            try:
                throughput = measure_throughput(
                    model, tokenizer, 
                    batch_size=test_case['batch_size'],
                    input_len=test_case['input_len'],
                    output_len=test_case['output_len']
                )
                
                test_name = f"bs{test_case['batch_size']}_in{test_case['input_len']}_out{test_case['output_len']}"
                config_results[test_name] = throughput
                
                print(f"{test_name}: {throughput:.2f} tokens/sec")
                
            except Exception as e:
                print(f"Failed {test_case}: {e}")
        
        results[config['name']] = config_results
        
        # 清理
        del model
        torch.cuda.empty_cache()
    
    # 分析和可视化结果
    analyze_throughput_results(results)

def measure_throughput(model, tokenizer, batch_size, input_len, output_len, num_runs=5):
    """测量特定配置下的吞吐量"""
    
    # 准备输入
    input_text = "The quick brown fox jumps over the lazy dog. " * (input_len // 10)
    inputs = tokenizer(
        [input_text] * batch_size,
        return_tensors="pt",
        max_length=input_len,
        truncation=True,
        padding=True
    ).to('cuda')
    
    # 预热
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=10, do_sample=False)
    
    torch.cuda.synchronize()
    
    # 测量
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=output_len,
                do_sample=False,
                use_cache=True
            )
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        times.append(end_time - start_time)
    
    # 计算吞吐量 (tokens/sec)
    avg_time = np.mean(times[1:])  # 排除第一次运行
    total_tokens = batch_size * output_len
    throughput = total_tokens / avg_time
    
    return throughput

def analyze_throughput_results(results):
    """分析吞吐量结果"""
    
    # 计算速度提升
    baseline_name = 'Baseline'
    speedups = {}
    
    for config_name, config_results in results.items():
        if config_name == baseline_name:
            continue
            
        config_speedups = {}
        for test_name, throughput in config_results.items():
            if test_name in results[baseline_name]:
                baseline_throughput = results[baseline_name][test_name]
                speedup = throughput / baseline_throughput
                config_speedups[test_name] = speedup
        
        speedups[config_name] = config_speedups
    
    # 可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 绝对吞吐量对比
    test_names = list(results[baseline_name].keys())
    x = np.arange(len(test_names))
    width = 0.25
    
    for i, (config_name, config_results) in enumerate(results.items()):
        throughputs = [config_results.get(name, 0) for name in test_names]
        ax1.bar(x + i*width, throughputs, width, label=config_name)
    
    ax1.set_xlabel('Test Configuration')
    ax1.set_ylabel('Throughput (tokens/sec)')
    ax1.set_title('Throughput Comparison')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(test_names, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 相对速度提升
    for config_name, config_speedups in speedups.items():
        speedup_values = [config_speedups.get(name, 0) for name in test_names]
        ax2.plot(x, speedup_values, marker='o', label=config_name)
    
    ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Baseline')
    ax2.set_xlabel('Test Configuration')
    ax2.set_ylabel('Speedup (×)')
    ax2.set_title('Relative Speedup vs Baseline')
    ax2.set_xticks(x)
    ax2.set_xticklabels(test_names, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('benchmarks/throughput_analysis.png', dpi=300)
    
    # 打印统计信息
    print("\n=== Speedup Analysis ===")
    for config_name, config_speedups in speedups.items():
        avg_speedup = np.mean(list(config_speedups.values()))
        max_speedup = max(config_speedups.values())
        print(f"{config_name}: Avg speedup = {avg_speedup:.2f}×, Max speedup = {max_speedup:.2f}×")
