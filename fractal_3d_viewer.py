import numpy as np
import plotly.graph_objects as go
import gradio as gr
import os
from datetime import datetime

def iterate_fractal(x0, y0, z0, iterations):
    """
    Generate fractal points using the given equations:
    x(n+1) = x(n) + cos(y(n))*sin(z(n))
    y(n+1) = y(n) + cos(z(n))*sin(x(n))
    z(n+1) = z(n) + cos(x(n))*sin(y(n))
    """
    x = np.zeros(iterations)
    y = np.zeros(iterations)
    z = np.zeros(iterations)
    
    x[0] = x0
    y[0] = y0
    z[0] = z0
    
    for i in range(1, iterations):
        x[i] = x[i-1] + np.cos(y[i-1]) * np.sin(z[i-1])
        y[i] = y[i-1] + np.cos(z[i-1]) * np.sin(x[i-1])
        z[i] = z[i-1] + np.cos(x[i-1]) * np.sin(y[i-1])
    
    return x, y, z

def create_3d_plot(x0, y0, z0, iterations, point_size, colorscale):
    """Create an interactive 3D plot of the fractal"""
    x, y, z = iterate_fractal(x0, y0, z0, iterations)
    
    # Create color gradient based on iteration number
    colors = np.arange(len(x))
    
    # Create 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers+lines',
        marker=dict(
            size=point_size,
            color=colors,
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(title="Iteration")
        ),
        line=dict(
            color=colors,
            colorscale=colorscale,
            width=1
        ),
        text=[f'Iteration {i}<br>x: {x[i]:.3f}<br>y: {y[i]:.3f}<br>z: {z[i]:.3f}' 
              for i in range(len(x))],
        hoverinfo='text'
    )])
    
    # Update layout
    fig.update_layout(
        title=f'3D Fractal Visualization<br>Initial: ({x0:.2f}, {y0:.2f}, {z0:.2f})',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube'
        ),
        width=900,
        height=700,
        showlegend=False
    )
    
    return fig

def reset_values():
    """Reset to default values"""
    return 0.1, 0.1, 0.1, 1000, 2, "Viridis"

def save_obj(x0, y0, z0, iterations):
    """Save fractal vertices as OBJ file"""
    x, y, z = iterate_fractal(x0, y0, z0, int(iterations))
    
    # Create output directory if it doesn't exist
    os.makedirs("fractal_outputs", exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"fractal_outputs/fractal_{timestamp}.obj"
    
    # Write OBJ file
    with open(filename, 'w') as f:
        f.write(f"# Fractal OBJ Export\n")
        f.write(f"# Initial conditions: x0={x0}, y0={y0}, z0={z0}\n")
        f.write(f"# Iterations: {iterations}\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Write vertices
        for i in range(len(x)):
            f.write(f"v {x[i]:.6f} {y[i]:.6f} {z[i]:.6f}\n")
        
        # Optionally write line segments connecting consecutive points
        f.write("\n# Line segments\n")
        for i in range(len(x) - 1):
            f.write(f"l {i+1} {i+2}\n")
    
    return f"Saved to {filename} ({len(x)} vertices)"

# Create Gradio interface
with gr.Blocks(title="3D Fractal Viewer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 3D Fractal Viewer
    
    Visualize the 3D dynamical system:
    - **x(n+1) = x(n) + cos(y(n)) × sin(z(n))**
    - **y(n+1) = y(n) + cos(z(n)) × sin(x(n))**
    - **z(n+1) = z(n) + cos(x(n)) × sin(y(n))**
    
    Adjust the initial conditions and parameters to explore different fractal patterns.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Initial Conditions")
            x0 = gr.Slider(-5, 5, value=0.1, step=0.1, label="x₀ (Initial X)")
            y0 = gr.Slider(-5, 5, value=0.1, step=0.1, label="y₀ (Initial Y)")
            z0 = gr.Slider(-5, 5, value=0.1, step=0.1, label="z₀ (Initial Z)")
            
            gr.Markdown("### Visualization Parameters")
            iterations = gr.Slider(100, 10000, value=1000, step=100, label="Iterations")
            point_size = gr.Slider(0.5, 5, value=2, step=0.5, label="Point Size")
            colorscale = gr.Dropdown(
                ["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Rainbow", "Jet", "Hot", "Cool"],
                value="Viridis",
                label="Color Scale"
            )
            
            with gr.Row():
                generate_btn = gr.Button("Generate", variant="primary", size="lg")
                reset_btn = gr.Button("Reset", size="lg")
            
            save_btn = gr.Button("💾 Save as OBJ", variant="secondary", size="lg")
            save_status = gr.Textbox(label="Export Status", interactive=False)
        
        with gr.Column(scale=2):
            plot_output = gr.Plot(label="3D Fractal")
    
    # Button actions
    generate_btn.click(
        fn=create_3d_plot,
        inputs=[x0, y0, z0, iterations, point_size, colorscale],
        outputs=plot_output
    )
    
    reset_btn.click(
        fn=reset_values,
        outputs=[x0, y0, z0, iterations, point_size, colorscale]
    )
    
    save_btn.click(
        fn=save_obj,
        inputs=[x0, y0, z0, iterations],
        outputs=save_status
    )
    
    # Generate initial plot on load
    demo.load(
        fn=create_3d_plot,
        inputs=[x0, y0, z0, iterations, point_size, colorscale],
        outputs=plot_output
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
