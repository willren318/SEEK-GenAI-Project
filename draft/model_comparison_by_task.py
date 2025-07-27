#!/usr/bin/env python3
"""
Script to create a comparison chart of model performance across different tasks.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def create_task_comparison_chart(data_path, output_path=None, metric='accuracy', figsize=(14, 8)):
    """
    Create a bar chart comparing model performance across different tasks.
    
    Parameters:
    -----------
    data_path : str
        Path to CSV file with model performance data
    output_path : str
        Path to save the output chart
    metric : str
        Metric to compare (e.g., 'accuracy', 'cost', 'time')
    figsize : tuple
        Figure size
    """
    # Read data
    df = pd.read_csv(data_path)
    
    # Normalize task names to avoid duplicates
    df['task_name'] = df['task_name'].str.strip().str.lower()
    
    # Get unique tasks and models
    tasks = sorted(df['task_name'].unique())  # Sort to ensure consistent order
    
    # Create standardized model order
    all_models = [
        'claude-3-7-sonnet-20250219',
        'claude-3-haiku-20240307',
        'llama4-maverick',
        'llama3.1-70b',
        'llama3.1-8b',
        'deepseek-reasoner',
        'deepseek-chat'
    ]
    
    # Filter to only include models that exist in the data
    models = [model for model in all_models if model in df['model_variant'].unique()]
    
    # Create readable model names for display
    model_display_names = []
    for model in models:
        if model.startswith('claude-3-7-sonnet'):
            display_name = 'Claude Sonnet'
        elif model.startswith('claude-3-haiku'):
            display_name = 'Claude Haiku'
        elif model.startswith('llama4'):
            display_name = 'Llama 4'
        elif '70b' in model:
            display_name = 'Llama 3.1 70B'
        elif '8b' in model:
            display_name = 'Llama 3.1 8B'
        elif model.endswith('reasoner'):
            display_name = 'DeepSeek Reasoner'
        elif model.endswith('chat'):
            display_name = 'DeepSeek Chat'
        else:
            display_name = model
        model_display_names.append(display_name)
    
    # Create a mapping from model variant to display name
    model_name_map = dict(zip(models, model_display_names))
    
    # Create readable task names for the legend
    task_display_names = []
    for task in tasks:
        if task == 'work_arrangement':
            display_name = 'Work Arrangement'
        elif task == 'salary':
            display_name = 'Salary'
        elif task == 'seniority':
            display_name = 'Seniority'
        else:
            display_name = task.capitalize()
        task_display_names.append(display_name)
    
    # Create a mapping from task to display name
    task_name_map = dict(zip(tasks, task_display_names))
    
    # Create a figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use consistent colors
    # Explicitly create a color palette with specific colors for each task
    colors = {
        'work_arrangement': '#1f77b4',  # blue
        'salary': '#ff7f0e',           # orange
        'seniority': '#2ca02c'         # green
    }
    
    # Set the width of each bar
    bar_width = 0.25
    
    # Set positions for bars
    positions = np.arange(len(model_display_names))
    
    # Filter the dataframe to only include models in our defined order
    df_filtered = df[df['model_variant'].isin(models)].copy()
    
    # Create a categorical type with our specific order
    df_filtered['model_display'] = df_filtered['model_variant'].map(model_name_map)
    df_filtered['model_variant'] = pd.Categorical(
        df_filtered['model_variant'], 
        categories=models, 
        ordered=True
    )
    
    # Sort the dataframe by model_variant to ensure consistent order
    df_filtered = df_filtered.sort_values('model_variant')
    
    # Plot bars for each task
    legend_handles = []
    
    for i, task in enumerate(tasks):
        task_data = df_filtered[df_filtered['task_name'] == task]
        color = colors.get(task, f'C{i}')
        
        # Ensure all models are represented (even if missing data)
        task_values = []
        for model in models:
            model_value = task_data[task_data['model_variant'] == model][metric].values
            if len(model_value) > 0:
                task_values.append(model_value[0])
            else:
                task_values.append(0)  # Default if model not present for this task
        
        # Calculate bar positions
        bar_pos = positions + (i - 1) * bar_width
        
        # Plot the bars for this task
        bars = ax.bar(
            bar_pos, 
            task_values, 
            width=bar_width, 
            color=color, 
            label=task_name_map[task],
            edgecolor='black',
            linewidth=0.5,
            alpha=0.8
        )
        legend_handles.append(bars)
        
        # Add value labels on top of bars
        for j, value in enumerate(task_values):
            if value > 0:  # Only add labels for non-zero values
                ax.text(
                    bar_pos[j], 
                    value + 0.01, 
                    f'{value:.2f}', 
                    ha='center', 
                    va='bottom', 
                    fontsize=8,
                    rotation=90
                )
    
    # Add some text for labels, title and axes
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{metric.capitalize()}', fontsize=12, fontweight='bold')
    ax.set_title(f'Comparison of {metric.capitalize()} Across Tasks by Model', fontsize=14, fontweight='bold')
    ax.set_xticks(positions)
    ax.set_xticklabels(model_display_names, rotation=45, ha='right')
    
    # Create custom legend with display names
    legend_labels = [task_name_map[task] for task in tasks]
    ax.legend(title='Task', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add grid for readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add horizontal line for comparison (e.g., at 0.7 accuracy)
    if metric == 'accuracy':
        ax.axhline(y=0.7, color='r', linestyle='--', alpha=0.5, label='0.7 Threshold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to {output_path}")
    else:
        plt.show()
    
    return fig

def create_multi_metric_comparison(data_path, output_path=None, figsize=(15, 12)):
    """
    Create a multi-faceted chart showing accuracy, cost, and time across tasks.
    
    Parameters:
    -----------
    data_path : str
        Path to CSV file with model performance data
    output_path : str
        Path to save the output chart
    figsize : tuple
        Figure size
    """
    # Read data
    df = pd.read_csv(data_path)
    
    # Normalize task names to avoid duplicates
    df['task_name'] = df['task_name'].str.strip().str.lower()
    
    # Get unique tasks and models
    tasks = sorted(df['task_name'].unique())  # Sort to ensure consistent order
    
    # Create standardized model order
    all_models = [
        'claude-3-7-sonnet-20250219',
        'claude-3-haiku-20240307',
        'llama4-maverick',
        'llama3.1-70b',
        'llama3.1-8b',
        'deepseek-reasoner',
        'deepseek-chat'
    ]
    
    # Filter to only include models that exist in the data
    models = [model for model in all_models if model in df['model_variant'].unique()]
    
    # Create readable model names for display
    model_display_names = []
    for model in models:
        if model.startswith('claude-3-7-sonnet'):
            display_name = 'Claude Sonnet'
        elif model.startswith('claude-3-haiku'):
            display_name = 'Claude Haiku'
        elif model.startswith('llama4'):
            display_name = 'Llama 4'
        elif '70b' in model:
            display_name = 'Llama 3.1 70B'
        elif '8b' in model:
            display_name = 'Llama 3.1 8B'
        elif model.endswith('reasoner'):
            display_name = 'DeepSeek Reasoner'
        elif model.endswith('chat'):
            display_name = 'DeepSeek Chat'
        else:
            display_name = model
        model_display_names.append(display_name)
    
    # Create a mapping from model variant to display name
    model_name_map = dict(zip(models, model_display_names))
    
    # Create readable task names for the legend
    task_display_names = []
    for task in tasks:
        if task == 'work_arrangement':
            display_name = 'Work Arrangement'
        elif task == 'salary':
            display_name = 'Salary'
        elif task == 'seniority':
            display_name = 'Seniority'
        else:
            display_name = task.capitalize()
        task_display_names.append(display_name)
    
    # Create a mapping from task to display name
    task_name_map = dict(zip(tasks, task_display_names))
    
    # Set up the figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    metrics = ['accuracy', 'cost', 'time']
    titles = ['Accuracy Comparison', 'Cost per Example Comparison', 'Latency Comparison']
    ylabels = ['Accuracy', 'Cost ($)', 'Time (seconds)']
    
    # Set up seaborn for prettier plots
    sns.set_style("whitegrid")
    
    # Use consistent colors
    # Explicitly create a color palette with specific colors for each task
    colors = {
        'work_arrangement': '#1f77b4',  # blue
        'salary': '#ff7f0e',           # orange
        'seniority': '#2ca02c'         # green
    }
    
    # Define a consistent color palette for tasks to ensure consistency across subplots
    task_palette = {task: colors.get(task, '#333333') for task in tasks}
    
    # Filter the dataframe to only include models in our defined order
    df_filtered = df[df['model_variant'].isin(models)].copy()
    
    # Create a categorical type with our specific order
    df_filtered['model_display'] = df_filtered['model_variant'].map(model_name_map)
    df_filtered['model_variant'] = pd.Categorical(
        df_filtered['model_variant'], 
        categories=models, 
        ordered=True
    )
    
    # Sort the dataframe by model_variant to ensure consistent order
    df_filtered = df_filtered.sort_values('model_variant')
    
    # Plot each metric
    for i, (metric, title, ylabel) in enumerate(zip(metrics, titles, ylabels)):
        ax = axes[i]
        
        # Create the grouped bar chart
        chart = sns.barplot(
            x='model_variant', 
            y=metric, 
            hue='task_name',
            data=df_filtered, 
            ax=ax,
            palette=task_palette,
            hue_order=tasks,  # Ensure consistent order
            order=models     # Use our specific model order
        )
        
        # Set x-tick labels to use our display names
        ax.set_xticklabels([model_name_map[model] for model in models], rotation=45, ha='right')
        
        # Customize the plot
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=10)
        
        # Remove legend from all plots (we'll add a custom one later)
        ax.get_legend().remove()
        
        # For cost and time, use log scale
        if metric in ['cost', 'time']:
            ax.set_yscale('log')
            ax.set_ylabel(f'{ylabel} (log scale)', fontsize=10)
    
    # Create better x-axis labels
    axes[-1].set_xlabel('Model', fontsize=10)
    
    # Create a cleaner legend with proper task display names
    legend_patches = []
    legend_labels = []
    
    # Use our sorted tasks list to ensure consistent order
    for task in tasks:
        color = task_palette[task]
        legend_patches.append(plt.Rectangle((0, 0), 1, 1, fc=color))
        legend_labels.append(task_name_map[task])  # Use display name
    
    # Add the legend to the figure
    fig.legend(
        legend_patches,
        legend_labels,
        title='Task',
        loc='upper right',
        bbox_to_anchor=(0.98, 0.98),
        frameon=True
    )
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Multi-metric chart saved to {output_path}")
    
    return fig

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create model comparison charts across tasks')
    parser.add_argument('--input', required=True, help='Path to combined CSV data file')
    parser.add_argument('--output', help='Path to save the chart')
    parser.add_argument('--metric', default='accuracy', choices=['accuracy', 'cost', 'time'],
                        help='Metric to compare')
    parser.add_argument('--multi-metric', action='store_true', 
                        help='Create a multi-metric comparison chart')
    parser.add_argument('--figsize', type=float, nargs=2, default=(14, 8),
                        help='Figure size (width height)')
    
    args = parser.parse_args()
    
    if args.multi_metric:
        create_multi_metric_comparison(args.input, args.output, tuple(args.figsize))
    else:
        create_task_comparison_chart(args.input, args.output, args.metric, tuple(args.figsize)) 