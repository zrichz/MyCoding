import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
from datetime import datetime
import os


class Cell:
    """Represents a single cell with type, life score, and movement direction."""
    def __init__(self, cell_type, life_score=1, direction=None):
        self.cell_type = cell_type  # 'A', 'B', or 'M' (mixed)
        self.life_score = life_score
        # Direction: 'down' for A-type, 'right' for B-type, 'down'/'right' for mixed
        if direction is None:
            self.direction = 'down' if cell_type == 'A' else 'right'
        else:
            self.direction = direction


class CellularFlowSimulation:
    """Simulates A cells moving down and B cells moving right on a grid."""
    
    def __init__(self, grid_size=512):
        self.grid_size = grid_size
        self.grid = [[[] for _ in range(grid_size)] for _ in range(grid_size)]
        
    def spawn_cells(self, num_a_cells, num_b_cells):
        """Spawn A cells on row 0 and B cells on column 0."""
        # Spawn A cells randomly on row 0
        for _ in range(num_a_cells):
            col = np.random.randint(0, self.grid_size)
            self.grid[0][col].append(Cell('A', life_score=1, direction='down'))
        
        # Spawn B cells randomly on column 0
        for _ in range(num_b_cells):
            row = np.random.randint(0, self.grid_size)
            self.grid[row][0].append(Cell('B', life_score=1, direction='right'))
    
    def find_nearest_free_cell(self, row, col):
        """Find the nearest cell location that is empty using BFS."""
        visited = set()
        queue = [(row, col, 0)]  # (row, col, distance)
        visited.add((row, col))
        
        while queue:
            r, c, dist = queue.pop(0)
            
            # Check if this cell is free (and not the starting cell)
            if dist > 0 and len(self.grid[r][c]) == 0:
                return (r, c)
            
            # Add neighbors to queue
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                    if (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc, dist + 1))
        
        # If no free cell found, return original location
        return (row, col)
    
    def merge_cells_at_location(self, row, col):
        """Merge cells: one stays with combined score, others relocate with life+1."""
        cells = self.grid[row][col]
        if len(cells) <= 1:
            return
        
        # Count cells by their original direction
        down_score = sum(cell.life_score for cell in cells if cell.direction == 'down')
        right_score = sum(cell.life_score for cell in cells if cell.direction == 'right')
        
        total_score = down_score + right_score
        
        # Determine movement direction and type based on what combined
        if down_score > 0 and right_score > 0:
            # Mixed: A and B cells combined, choose direction randomly
            cell_type = 'M'
            direction = np.random.choice(['down', 'right'])
        elif down_score > 0:
            # Only A-type cells (moving down)
            cell_type = 'A'
            direction = 'down'
        else:
            # Only B-type cells (moving right)
            cell_type = 'B'
            direction = 'right'
        
        # Keep one merged cell at this location
        merged_cell = Cell(cell_type, life_score=total_score, direction=direction)
        self.grid[row][col] = [merged_cell]
        
        # Relocate the remaining cells (all but one)
        for i in range(len(cells) - 1):
            # Find nearest free location
            new_row, new_col = self.find_nearest_free_cell(row, col)
            
            # Create a cell with incremented life score
            old_cell = cells[i]
            relocated_cell = Cell(
                old_cell.cell_type, 
                life_score=old_cell.life_score + 1, 
                direction=old_cell.direction
            )
            self.grid[new_row][new_col].append(relocated_cell)
    
    def move_cells(self):
        """Move all cells according to their direction."""
        new_grid = [[[] for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                for cell in self.grid[row][col]:
                    if cell.direction == 'down':
                        # Move down to one of 3 neighboring cells
                        if row < self.grid_size - 1:
                            # Choose from (row+1, col-1), (row+1, col), (row+1, col+1)
                            choices = []
                            if col > 0:
                                choices.append((row + 1, col - 1))
                            choices.append((row + 1, col))
                            if col < self.grid_size - 1:
                                choices.append((row + 1, col + 1))
                            
                            new_row, new_col = choices[np.random.randint(len(choices))]
                            new_grid[new_row][new_col].append(cell)
                        # Cells at bottom row die
                    
                    elif cell.direction == 'right':
                        # Move right to one of 3 neighboring cells
                        if col < self.grid_size - 1:
                            # Choose from (row-1, col+1), (row, col+1), (row+1, col+1)
                            choices = []
                            if row > 0:
                                choices.append((row - 1, col + 1))
                            choices.append((row, col + 1))
                            if row < self.grid_size - 1:
                                choices.append((row + 1, col + 1))
                            
                            new_row, new_col = choices[np.random.randint(len(choices))]
                            new_grid[new_row][new_col].append(cell)
                        # Cells at rightmost column die
        
        self.grid = new_grid
    
    def merge_all_cells(self):
        """Merge cells at all locations."""
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                self.merge_cells_at_location(row, col)
    
    def step(self, num_a_cells, num_b_cells):
        """Execute one timestep of the simulation."""
        self.spawn_cells(num_a_cells, num_b_cells)
        self.move_cells()
        self.merge_all_cells()
    
    def get_visualization_array(self):
        """Convert grid to numpy array for visualization."""
        # Create arrays for A cells, B cells, mixed cells, and total
        a_array = np.zeros((self.grid_size, self.grid_size))
        b_array = np.zeros((self.grid_size, self.grid_size))
        m_array = np.zeros((self.grid_size, self.grid_size))
        total_array = np.zeros((self.grid_size, self.grid_size))
        
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                for cell in self.grid[row][col]:
                    total_array[row, col] += cell.life_score
                    if cell.cell_type == 'A':
                        a_array[row, col] += cell.life_score
                    elif cell.cell_type == 'B':
                        b_array[row, col] += cell.life_score
                    elif cell.cell_type == 'M':
                        m_array[row, col] += cell.life_score
        
        return a_array, b_array, m_array, total_array


def run_simulation(num_a_cells, num_b_cells, num_timesteps, seed, grid_size):
    """Run the cellular flow simulation."""
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Initialize simulation
    sim = CellularFlowSimulation(grid_size=grid_size)
    
    # Run simulation
    for step in range(num_timesteps):
        sim.step(num_a_cells, num_b_cells)
    
    # Get visualization arrays
    a_array, b_array, m_array, total_array = sim.get_visualization_array()
    
    # Apply log mapping for better visibility of small values
    total_log = np.log1p(total_array)  # log(1 + x) to handle zeros
    total_normalized = total_log / (total_log.max() + 1e-10)
    
    # Create combined visualization with viridis and log mapping
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Use log-normalized array with viridis for maximum contrast
    im = ax.imshow(total_normalized, cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
    ax.set_title(f'Cellular Flow Simulation\n{grid_size}x{grid_size} grid, {num_timesteps} timesteps', fontsize=16)
    ax.axis('off')
    cbar = plt.colorbar(im, ax=ax, label='Log(Life Score + 1)', fraction=0.046, pad=0.04)
    plt.tight_layout()
    
    # Save figure
    output_dir = '/home/rich/MyCoding/GENERATIVE_ART/cellular_flow_outputs'
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'cellular_flow_{grid_size}_a{num_a_cells}_b{num_b_cells}_t{num_timesteps}_s{seed}_{timestamp}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    
    # Statistics
    total_a = int(a_array.sum())
    total_b = int(b_array.sum())
    total_m = int(m_array.sum())
    total_all = int(total_array.sum())
    max_life_score = int(total_array.max())
    
    message = (f"Simulation completed\n"
               f"Grid size: {grid_size}x{grid_size}\n"
               f"Timesteps: {num_timesteps}\n"
               f"A cells spawned per step: {num_a_cells}\n"
               f"B cells spawned per step: {num_b_cells}\n"
               f"Total A life score: {total_a}\n"
               f"Total B life score: {total_b}\n"
               f"Total Mixed life score: {total_m}\n"
               f"Overall total life score: {total_all}\n"
               f"Maximum life score at single location: {max_life_score}\n"
               f"Seed: {seed}\n"
               f"Saved to: {filename}")
    
    return fig, message


# Create Gradio interface
with gr.Blocks(title="Cellular Flow Simulation") as demo:
    gr.Markdown("# Cellular Flow Simulation")
    gr.Markdown("""
    This simulation spawns two types of cells on a grid:
    - **A cells** spawn on row 0 and move downward to one of 3 neighboring cells
    - **B cells** spawn on column 0 and move rightward to one of 3 neighboring cells
    - When cells occupy the same location, they merge with combined life scores
    - **Merged cells continue moving**: A+A moves down, B+B moves right, A+B randomly chooses direction
    - **Viridis colormap** provides maximum contrast, normalized after simulation completes
    """)
    
    with gr.Row():
        with gr.Column():
            grid_size = gr.Radio(
                choices=[128, 256, 512],
                value=256,
                label="Grid Size (pixels)"
            )
            num_a_cells = gr.Slider(
                minimum=1,
                maximum=100,
                value=20,
                step=1,
                label="A Cells per Timestep"
            )
            num_b_cells = gr.Slider(
                minimum=1,
                maximum=100,
                value=20,
                step=1,
                label="B Cells per Timestep"
            )
            num_timesteps = gr.Slider(
                minimum=10,
                maximum=500,
                value=100,
                step=10,
                label="Number of Timesteps"
            )
            seed = gr.Number(
                value=42,
                label="Random Seed",
                precision=0
            )
            run_button = gr.Button("Run Simulation", variant="primary")
    
    with gr.Row():
        output_plot = gr.Plot(label="Cellular Flow Visualization")
    
    with gr.Row():
        output_message = gr.Textbox(label="Simulation Info", lines=10)
    
    run_button.click(
        fn=run_simulation,
        inputs=[num_a_cells, num_b_cells, num_timesteps, seed, grid_size],
        outputs=[output_plot, output_message]
    )
    
    gr.Markdown("""
    ### How to Use:
    1. Select grid size: 128, 256, or 512 pixels
    2. Adjust the number of A and B cells spawned per timestep
    3. Set the total number of timesteps to run
    4. Optionally change the random seed for different patterns
    5. Click "Run Simulation" to execute
    6. Results are automatically saved as PNG files
    
    ### Interpretation:
    - **Viridis colormap with log scale**: Better visibility for both small and large values
    - Dark purple = low cell density, bright yellow = high cell density
    - Shows all cells (A, B, and merged) in a single visualization
    - Life scores accumulate where cells merge, creating convergence patterns
    - Diagonal patterns emerge where A cells (down) and B cells (right) intersect
    """)


if __name__ == "__main__":
    demo.launch(inbrowser=True)
