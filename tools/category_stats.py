import os.path
from pathlib import Path
from tqdm import tqdm
from glob import glob
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

from models.utils import get_all_sequence_dirs, action_labels
from database import Database

data_dir = Path("data/track_3.0.0")  # Should have all categories as subdirectories
image_suffix = "rgb.png"


def pie_chart(data, save_path="data/temp_pie.pdf", title=""):
    labels = list(data.keys())
    sizes = list(data.values())
    
    # Set modern style
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 11
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create color palette
    colors = sns.color_palette("viridis", len(labels))
    
    # Create labels with category name and value
    labels_with_values = [f"{label}\n{value:,}" for label, value in zip(labels, sizes)]
    
    # Create pie chart
    wedges, texts = ax.pie(
        sizes,
        labels=labels_with_values,
        startangle=140,
        colors=colors,
        shadow=False,
        textprops={'fontsize': 20},
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
    )
    
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(save_path, format="pdf", dpi=300, bbox_inches='tight')
    print("Saved pie chart to", save_path)


def bar_chart(data, save_path="data/temp_bar.pdf", title="", xlabel="Category", ylabel="Number of Frames"):
    sorted_items = sorted(data.items(), key=lambda x: x[1], reverse=True)
    categories, values = zip(*sorted_items)
    
    # Set a modern, clean style
    sns.set_theme(style="ticks", palette="deep")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 11
    
    fig, ax = plt.subplots(figsize=(8, 6))
    df = pd.DataFrame({"Category": categories, "Values": values})
    
    # Create bar plot with custom colors and thinner bars
    bars = sns.barplot(
        data=df, x="Category", y="Values", order=categories,
        palette="viridis", edgecolor='black', linewidth=0.8, ax=ax,
        width=0.7  # Make bars thinner (default is 0.8)
    )
    
    # Add value labels on top of bars
    for container in ax.containers:
        for bar in container.patches:
            height = bar.get_height()
            x = bar.get_x() + bar.get_width() / 2
            label = f"{int(height):,}"  # Add comma separator for thousands
            ax.text(x, height + max(values) * 0.01, label, 
                   ha='left', va='bottom', fontsize=14, rotation=45, rotation_mode='anchor')
    
    # Styling
    ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    
    # Format y-axis with comma separators
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Rotate x-axis labels with end of text aligned to center of bar
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=14, rotation_mode='anchor')
    plt.yticks(fontsize=14)
    
    # Add subtle grid
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.7)
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    sns.despine()
    
    plt.tight_layout()
    plt.savefig(save_path, format="pdf", dpi=300, bbox_inches='tight')
    print("Saved bar chart to", save_path)


if __name__ == "__main__":
    category_to_frames = {}
    category_to_videos = {}
    action_to_videos = {action: 0 for action in action_labels}
    for cat_dir in tqdm(list(data_dir.iterdir())):
        if not cat_dir.is_dir():
            continue
        category = cat_dir.name
        if category not in category_to_frames:
            category_to_frames[category] = 0
            category_to_videos[category] = 0
        all_sequence_dirs = get_all_sequence_dirs(cat_dir)
        category_to_videos[category] = len(all_sequence_dirs)
        for sequence_dir in all_sequence_dirs:
            num_frames = len(list(glob(f"{sequence_dir}/*{image_suffix}", recursive=True)))
            category_to_frames[category] += num_frames
            actions_file = os.path.join(sequence_dir, "actions.txt")
            with open(actions_file, 'r') as f:
                actions = f.read().splitlines()
            for action in actions:
                if action in action_to_videos:
                    action_to_videos[action] += 1
    print("Total frames:", sum(category_to_frames.values()))
    print("Total videos:", sum(category_to_videos.values()))

    for category in category_to_frames.keys():
        print(f"{category}:\t{category_to_videos[category]} tracks,\t{category_to_frames[category]} frames")
    print(f"total:\t{sum(category_to_videos.values())} tracks,\t{sum(category_to_frames.values())} frames")
    for action in action_to_videos.keys():
        print(f"{action}:\t{action_to_videos[action]} tracks")

    bar_chart(category_to_frames, save_path="data/category_bar.pdf")
    pie_chart(category_to_frames, save_path="data/category_pie.pdf")
    bar_chart(category_to_videos, save_path="data/category_bar_videos.pdf", xlabel="Category", ylabel="Number of Videos")
    pie_chart(category_to_videos, save_path="data/category_pie_videos.pdf")
    bar_chart(action_to_videos, save_path="data/action_bar.pdf", xlabel="Action", ylabel="Number of Videos")



