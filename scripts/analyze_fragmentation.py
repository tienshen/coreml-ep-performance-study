#!/usr/bin/env python3
"""Analyze graph fragmentation between BERT models."""

import json
from pathlib import Path
from collections import Counter


def analyze_model_fragmentation(model_name, profile_pattern):
    print('='*70)
    print(model_name)
    print('='*70)
    
    # Find CoreML profile
    profiles = list(Path('profiles').glob(profile_pattern))
    if not profiles:
        print(f'⚠ No profile found matching {profile_pattern}')
        return None
    
    profile_path = profiles[0]
    print(f'Profile: {profile_path.name}')
    
    with open(profile_path) as f:
        data = json.load(f)
    
    # Get kernel events
    kernels = [e for e in data if e.get('cat') == 'Node' and 'kernel_time' in e.get('name', '')]
    
    # Separate CoreML vs CPU ops
    coreml_ops = [e for e in kernels if 'CoreML' in e['name']]
    cpu_ops = [e for e in kernels if 'CoreML' not in e['name']]
    
    # Count unique CoreML partitions
    coreml_partitions = set()
    for op in coreml_ops:
        op_name = op['args'].get('op_name', '')
        if 'CoreML' in op_name:
            coreml_partitions.add(op_name)
    
    # Count CPU op types
    cpu_op_types = Counter()
    for op in cpu_ops:
        op_name = op['args'].get('op_name', 'unknown')
        cpu_op_types[op_name] += 1
    
    # Calculate timing
    coreml_time = sum(e['dur'] for e in coreml_ops)
    cpu_time = sum(e['dur'] for e in cpu_ops)
    total = coreml_time + cpu_time
    
    print(f'\n📊 Graph Fragmentation Analysis:')
    print(f'  Total kernel events: {len(kernels)}')
    print(f'  CoreML partition count: {len(coreml_partitions)}')
    print(f'  CoreML kernel calls: {len(coreml_ops)}')
    print(f'  CPU fallback ops: {len(cpu_ops)}')
    print(f'  Unique CPU op types: {len(cpu_op_types)}')
    
    print(f'\n⏱ Execution Time Breakdown:')
    print(f'  CoreML time: {coreml_time/1000:.2f}ms ({(coreml_time/total)*100:.1f}%)')
    print(f'  CPU time: {cpu_time/1000:.2f}ms ({(cpu_time/total)*100:.1f}%)')
    
    if coreml_partitions:
        print(f'\n🔗 CoreML Partitions:')
        for partition in sorted(coreml_partitions):
            count = sum(1 for op in coreml_ops if op['args'].get('op_name') == partition)
            print(f'  - {partition}: {count} executions')
    
    if cpu_op_types and len(cpu_op_types) <= 10:
        print(f'\n💻 CPU Fallback Operations:')
        for op_type, count in cpu_op_types.most_common(10):
            print(f'  - {op_type}: {count}x')
    
    return {
        'total_kernels': len(kernels),
        'coreml_partitions': len(coreml_partitions),
        'coreml_ops': len(coreml_ops),
        'cpu_ops': len(cpu_ops),
        'coreml_pct': (coreml_time/total)*100 if total > 0 else 0
    }


def main():
    # Analyze both models
    results = []
    
    print('\n')
    r1 = analyze_model_fragmentation('BERT-base-uncased', 'bert-base-uncased*CoreMLExecutionProvider.json')
    if r1:
        results.append(('BERT-base', r1))
    
    print('\n\n')
    r2 = analyze_model_fragmentation('tiny-systems-bert', 'tiny-systems-bert*CoreMLExecutionProvider.json')
    if r2:
        results.append(('tiny-systems-bert', r2))
    
    # Comparison summary
    if len(results) == 2:
        print('\n\n')
        print('='*70)
        print('FRAGMENTATION COMPARISON')
        print('='*70)
        print(f"{'Metric':<35} {'BERT-base':<15} {'tiny-systems-bert':<20}")
        print('-'*70)
        print(f"{'CoreML partitions':<35} {results[0][1]['coreml_partitions']:<15} {results[1][1]['coreml_partitions']:<20}")
        print(f"{'Total kernel events':<35} {results[0][1]['total_kernels']:<15} {results[1][1]['total_kernels']:<20}")
        print(f"{'CoreML operations':<35} {results[0][1]['coreml_ops']:<15} {results[1][1]['coreml_ops']:<20}")
        print(f"{'CPU fallback operations':<35} {results[0][1]['cpu_ops']:<15} {results[1][1]['cpu_ops']:<20}")
        print(f"{'CoreML time %':<35} {results[0][1]['coreml_pct']:.1f}%{' ':<12} {results[1][1]['coreml_pct']:.1f}%")
        print('='*70)


if __name__ == '__main__':
    main()
