import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import re
from pathlib import Path

def categorize_gpu(device_name):
    """Categorize GPU into series families"""
    if "RTX 50" in device_name or "RTX 5050" in device_name or "RTX 5060" in device_name or "RTX 5070" in device_name or "RTX 5080" in device_name or "RTX 5090" in device_name:
        return "50 Series (RTX 50xx)"
    elif "RTX 40" in device_name or "RTX 4060" in device_name or "RTX 4070" in device_name or "RTX 4080" in device_name or "RTX 4090" in device_name:
        return "40 Series (RTX 40xx)"
    elif "RTX 30" in device_name or "RTX 3050" in device_name or "RTX 3060" in device_name or "RTX 3070" in device_name or "RTX 3080" in device_name or "RTX 3090" in device_name:
        return "30 Series (RTX 30xx)"
    elif "RTX 20" in device_name or "RTX 2050" in device_name or "RTX 2060" in device_name or "RTX 2070" in device_name or "RTX 2080" in device_name:
        return "20 Series (RTX 20xx)"
    elif "GTX 16" in device_name or "GTX 1650" in device_name or "GTX 1660" in device_name:
        return "16 Series (GTX 16xx)"
    elif "GTX 10" in device_name or "GTX 1030" in device_name or "GTX 1050" in device_name or "GTX 1060" in device_name or "GTX 1070" in device_name or "GTX 1080" in device_name:
        return "10 Series (GTX 10xx)"
    else:
        return "Other GPUs"

def load_and_process_data(csv_path):
    """Load CSV and categorize GPUs"""
    df = pd.read_csv(csv_path)
    df['GPU Family'] = df['Device Name'].apply(categorize_gpu)
    df['Median Score'] = pd.to_numeric(df['Median Score'], errors='coerce')
    return df

def create_chart(df, selected_families, chart_type, sort_order):
    """Create bar chart for selected GPU families"""
    # Filter by selected families
    if not selected_families:
        return None, "Please select at least one GPU family"
    
    filtered_df = df[df['GPU Family'].isin(selected_families)].copy()
    
    if filtered_df.empty:
        return None, "No data available for selected families"
    
    # Sort data
    if sort_order == "Score (High to Low)":
        filtered_df = filtered_df.sort_values('Median Score', ascending=False)
    elif sort_order == "Score (Low to High)":
        filtered_df = filtered_df.sort_values('Median Score', ascending=True)
    elif sort_order == "Name (A-Z)":
        filtered_df = filtered_df.sort_values('Device Name')
    else:  # Name (Z-A)
        filtered_df = filtered_df.sort_values('Device Name', ascending=False)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, max(8, len(filtered_df) * 0.3)))
    
    if chart_type == "Grouped by Family":
        # Group by family and plot
        families = filtered_df['GPU Family'].unique()
        colors = plt.cm.tab10(range(len(families)))
        color_map = dict(zip(families, colors))
        
        bars = ax.barh(range(len(filtered_df)), 
                       filtered_df['Median Score'],
                       color=[color_map[family] for family in filtered_df['GPU Family']])
        
        ax.set_yticks(range(len(filtered_df)))
        ax.set_yticklabels(filtered_df['Device Name'], fontsize=9)
        ax.set_xlabel('Median Benchmark Score', fontsize=11, fontweight='bold')
        ax.set_title('GPU Benchmark Scores by Family', fontsize=13, fontweight='bold', pad=20)
        
        # Add legend
        legend_elements = [plt.Rectangle((0,0),1,1, fc=color_map[family], label=family) 
                          for family in families]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
        
    else:  # Single Color
        bars = ax.barh(range(len(filtered_df)), 
                       filtered_df['Median Score'],
                       color='steelblue')
        
        ax.set_yticks(range(len(filtered_df)))
        ax.set_yticklabels(filtered_df['Device Name'], fontsize=9)
        ax.set_xlabel('Median Benchmark Score', fontsize=11, fontweight='bold')
        ax.set_title('GPU Benchmark Scores', fontsize=13, fontweight='bold', pad=20)
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(filtered_df.iterrows()):
        ax.text(row['Median Score'] + max(filtered_df['Median Score']) * 0.01, 
                i, 
                f"{row['Median Score']:.0f}", 
                va='center', 
                fontsize=8)
    
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    plt.tight_layout()
    
    stats = f"Showing {len(filtered_df)} GPUs from {len(selected_families)} families"
    return fig, stats

def create_summary_stats(df, selected_families):
    """Create summary statistics for selected families"""
    if not selected_families:
        return "Please select at least one GPU family"
    
    filtered_df = df[df['GPU Family'].isin(selected_families)]
    
    stats_text = "Summary Statistics:\n\n"
    for family in selected_families:
        family_df = filtered_df[filtered_df['GPU Family'] == family]
        if not family_df.empty:
            stats_text += f"{family}:\n"
            stats_text += f"  Count: {len(family_df)} GPUs\n"
            stats_text += f"  Average Score: {family_df['Median Score'].mean():.2f}\n"
            stats_text += f"  Max Score: {family_df['Median Score'].max():.2f} ({family_df.loc[family_df['Median Score'].idxmax(), 'Device Name']})\n"
            stats_text += f"  Min Score: {family_df['Median Score'].min():.2f} ({family_df.loc[family_df['Median Score'].idxmin(), 'Device Name']})\n\n"
    
    return stats_text

def analyze_csv(csv_file, selected_families, chart_type, sort_order):
    """Main analysis function"""
    try:
        df = load_and_process_data(csv_file.name)
        fig, chart_info = create_chart(df, selected_families, chart_type, sort_order)
        stats = create_summary_stats(df, selected_families)
        return fig, chart_info, stats
    except Exception as e:
        return None, f"Error: {str(e)}", ""

# Create Gradio interface
with gr.Blocks(title="GPU Benchmark Viewer") as demo:
    gr.Markdown("# GPU Benchmark Data Viewer")
    gr.Markdown("Upload a GPU benchmark CSV file and visualize performance by GPU family")
    
    with gr.Row():
        with gr.Column(scale=1):
            csv_input = gr.File(label="Upload CSV File", file_types=[".csv"])
            
            family_selector = gr.CheckboxGroup(
                choices=[
                    "50 Series (RTX 50xx)",
                    "40 Series (RTX 40xx)",
                    "30 Series (RTX 30xx)",
                    "20 Series (RTX 20xx)",
                    "16 Series (GTX 16xx)",
                    "10 Series (GTX 10xx)",
                    "Other GPUs"
                ],
                value=["50 Series (RTX 50xx)", "40 Series (RTX 40xx)", "30 Series (RTX 30xx)"],
                label="Select GPU Families to Display"
            )
            
            chart_type = gr.Radio(
                choices=["Grouped by Family", "Single Color"],
                value="Grouped by Family",
                label="Chart Style"
            )
            
            sort_order = gr.Radio(
                choices=["Score (High to Low)", "Score (Low to High)", "Name (A-Z)", "Name (Z-A)"],
                value="Score (High to Low)",
                label="Sort Order"
            )
            
            analyze_btn = gr.Button("Generate Chart", variant="primary")
        
        with gr.Column(scale=2):
            chart_output = gr.Plot(label="Benchmark Chart")
            chart_info = gr.Textbox(label="Chart Info", lines=1)
            stats_output = gr.Textbox(label="Summary Statistics", lines=12)
    
    analyze_btn.click(
        fn=analyze_csv,
        inputs=[csv_input, family_selector, chart_type, sort_order],
        outputs=[chart_output, chart_info, stats_output]
    )
    
    gr.Markdown("""
    ### Instructions:
    1. Upload your GPU benchmark CSV file (must have columns: Device Name, Median Score)
    2. Select which GPU families you want to visualize
    3. Choose chart style and sort order
    4. Click Generate Chart to view the results
    
    The app automatically categorizes GPUs into series families (10, 16, 20, 30, 40, 50 series).
    """)

if __name__ == "__main__":
    demo.launch(inbrowser=True)
