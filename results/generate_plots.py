import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def generate_plots(input_file, output_dir):
    """Parses benchmark JSON and generates paper-ready plots."""
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    systems = data['systems']
    system_names = list(systems.keys())
    
    # Sort systems in a logical order for presentation
    display_order = ['NEXUS_v2', 'MemGPTStyle', 'Mem0Style', 'NaiveRAG', 'FullContext']
    valid_systems = [s for s in display_order if s in system_names]
    
    # Colors for consistent styling
    colors = {
        'NEXUS_v2': '#8B5CF6',     # Purple
        'MemGPTStyle': '#3B82F6',  # Blue
        'Mem0Style': '#10B981',    # Green
        'NaiveRAG': '#F59E0B',     # Yellow/Orange
        'FullContext': '#EF4444'   # Red
    }
    
    # Extract data
    ingest_times = [systems[s]['ingest_time_seconds'] for s in valid_systems]
    f1_scores = [systems[s]['aggregate']['f1_mean'] for s in valid_systems]
    latencies = [systems[s]['aggregate']['latency_ms_mean'] / 1000.0 for s in valid_systems] # convert to seconds
    tokens = [systems[s]['aggregate']['tokens_used_mean'] for s in valid_systems]
    
    os.makedirs(output_dir, exist_ok=True)
    
    plt.style.use('ggplot')
    
    # 1. Ingestion Scalability (Log Scale Plot)
    plt.figure(figsize=(10, 6))
    bars = plt.bar(valid_systems, ingest_times, color=[colors[s] for s in valid_systems])
    plt.yscale('log')
    plt.ylabel('Ingestion Time (Seconds) - Log Scale', fontsize=12)
    plt.title('Ingestion Scalability (419 Conversation Turns)', fontsize=14, pad=20)
    plt.xticks(rotation=15)
    
    # Add exact time labels on top of bars
    for bar in bars:
        height = bar.get_height()
        label = f"{height:.1f}s"
        if height > 1000:
            label = f"{height/60:.1f}m"
        plt.text(bar.get_x() + bar.get_width()/2., height,
                label, ha='center', va='bottom', fontsize=10, fontweight='bold')
                
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig1_ingestion.png'), dpi=300)
    plt.close()
    
    
    # 2. Retrieval Efficiency (Latency vs Tokens) - Scatter Plot
    plt.figure(figsize=(10, 6))
    for s in valid_systems:
        x = systems[s]['aggregate']['latency_ms_mean'] / 1000.0
        y = systems[s]['aggregate']['tokens_used_mean']
        plt.scatter(x, y, color=colors[s], s=200, label=s, alpha=0.8, edgecolors='black', linewidth=1.5)
        
        # Annotate
        plt.annotate(s, (x, y), xytext=(10, 5), textcoords="offset points", fontweight='bold')
    
    plt.xlabel('Query Latency (Seconds)', fontsize=12)
    plt.ylabel('Context Tokens Used', fontsize=12)
    plt.title('Retrieval Efficiency: Token Usage vs. Query Latency', fontsize=14, pad=20)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Optional limits to give some breathing room
    plt.xlim(max(0, min(latencies) - 1), max(latencies) + 2)
    plt.ylim(max(0, min(tokens) - 10), max(tokens) + 15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig2_efficiency.png'), dpi=300)
    plt.close()
    
    
    # 3. Accuracy (F1 Score)
    plt.figure(figsize=(10, 6))
    bars = plt.bar(valid_systems, f1_scores, color=[colors[s] for s in valid_systems])
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('Accuracy on LoCoMo Temporal Evaluation (Mistral 7B)', fontsize=14, pad=20)
    plt.xticks(rotation=15)
    
    # Add exact time labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f"{height:.3f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
                
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig3_accuracy.png'), dpi=300)
    plt.close()
    
    print(f"Generated 3 paper plots in {output_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to benchmark JSON results file")
    parser.add_argument("--output", default="paper/figures", help="Directory to save figures")
    args = parser.parse_args()
    
    generate_plots(args.input, args.output)
