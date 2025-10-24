"""
Predator-Prey Simulation with Agent Behavior Animation
Generates individual PNG frames showing predator-prey dynamics on a 2D plane.

Features:
- Predators hunt nearest prey within sight radius
- Prey flee from nearby predators
- Movement trajectories are visualized as lines
- Generates consecutive frames for animation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
from pathlib import Path
import random
import math
import subprocess
import os
import time
import gradio as gr
from dataclasses import dataclass
from typing import List, Tuple, Optional

# Set matplotlib to use a faster backend
import matplotlib
matplotlib.use('Agg')  # Use Anti-Grain Geometry backend for faster rendering

@dataclass
class Agent:
    """Base agent class"""
    x: float
    y: float
    sight_radius: float
    color: str
    size: float
    
class Predator(Agent):
    """Predator agent that hunts prey"""
    def __init__(self, x: float, y: float, sight_radius: float = 100.0, attack_distance: float = 25.0):
        super().__init__(x, y, sight_radius, color='red', size=8.0)
        self.last_x = x
        self.last_y = y
        self.has_moved = False
        self.attack_distance = attack_distance  # Maximum distance predator can move per turn
        
    def hunt(self, prey_array: np.ndarray, world_width: int, world_height: int) -> int:
        """Find nearest prey and move towards it within attack distance - vectorized version"""
        self.last_x = self.x
        self.last_y = self.y
        self.has_moved = False
        
        if len(prey_array) == 0:
            return -1
            
        # Vectorized distance calculation for all prey at once
        dx1 = prey_array[:, 0] - self.x
        dx2 = np.where(dx1 > 0, dx1 - world_width, dx1 + world_width)
        dx = np.where(np.abs(dx1) < np.abs(dx2), dx1, dx2)
        
        dy1 = prey_array[:, 1] - self.y
        dy2 = np.where(dy1 > 0, dy1 - world_height, dy1 + world_height)
        dy = np.where(np.abs(dy1) < np.abs(dy2), dy1, dy2)
        
        distances = np.sqrt(dx**2 + dy**2)
        
        # Find prey within sight radius
        visible_prey = distances <= self.sight_radius
        if not np.any(visible_prey):
            return -1
            
        # Find nearest visible prey
        visible_distances = np.where(visible_prey, distances, np.inf)
        nearest_idx = np.argmin(visible_distances)
        min_distance = visible_distances[nearest_idx]
        
        if min_distance == np.inf:
            return -1
            
        # Move towards nearest prey
        best_dx = dx[nearest_idx]
        best_dy = dy[nearest_idx]
        
        if min_distance > 0:
            # Calculate movement direction
            move_distance = min(self.attack_distance, min_distance)
            
            # Normalize direction and apply movement
            direction_x = best_dx / min_distance
            direction_y = best_dy / min_distance
            
            new_x = self.x + direction_x * move_distance
            new_y = self.y + direction_y * move_distance
            
            # Apply toroidal wrapping
            self.x = new_x % world_width
            self.y = new_y % world_height
            self.has_moved = True
            
            # Check if close enough to "eat" prey
            if min_distance <= self.attack_distance:
                return int(nearest_idx)
            
        return -1

class Prey(Agent):
    """Prey agent that flees from predators"""
    def __init__(self, x: float, y: float, sight_radius: float = 80.0):
        super().__init__(x, y, sight_radius, color='blue', size=4.0)
        self.last_x = x
        self.last_y = y
        self.has_moved = False
        self.flee_speed = 15.0  # How far they move when fleeing
        
    def flee(self, predator_array: np.ndarray, world_width: int, world_height: int):
        """Move away from nearby predators in toroidal world - vectorized version"""
        self.last_x = self.x
        self.last_y = self.y
        self.has_moved = False
        
        if len(predator_array) == 0:
            return
            
        # Vectorized distance calculation for all predators at once
        dx1 = predator_array[:, 0] - self.x
        dx2 = np.where(dx1 > 0, dx1 - world_width, dx1 + world_width)
        dx = np.where(np.abs(dx1) < np.abs(dx2), dx1, dx2)
        
        dy1 = predator_array[:, 1] - self.y
        dy2 = np.where(dy1 > 0, dy1 - world_height, dy1 + world_height)
        dy = np.where(np.abs(dy1) < np.abs(dy2), dy1, dy2)
        
        distances = np.sqrt(dx**2 + dy**2)
        
        # Find predators within sight radius
        nearby_mask = distances <= self.sight_radius
        if not np.any(nearby_mask):
            return
            
        # Get nearby predators
        nearby_dx = dx[nearby_mask]
        nearby_dy = dy[nearby_mask]
        nearby_distances = distances[nearby_mask]
        
        # Calculate flee direction (vectorized)
        weights = 1.0 / (nearby_distances + 1)  # +1 to avoid division by zero
        flee_x = np.sum((-nearby_dx / nearby_distances) * weights)
        flee_y = np.sum((-nearby_dy / nearby_distances) * weights)
        
        # Normalize flee direction
        flee_magnitude = math.sqrt(flee_x**2 + flee_y**2)
        if flee_magnitude > 0:
            flee_x = (flee_x / flee_magnitude) * self.flee_speed
            flee_y = (flee_y / flee_magnitude) * self.flee_speed
            
            # Apply movement with toroidal wrapping
            new_x = self.x + flee_x
            new_y = self.y + flee_y
            
            # Apply toroidal wrapping
            self.x = float(new_x % world_width)
            self.y = float(new_y % world_height)
            self.has_moved = True

class PredatorPreySimulation:
    """Main simulation class"""
    
    def __init__(self, width: int = 800, height: int = 600, 
                 num_predators: int = 5, num_prey: int = 30, attack_distance: float = 25.0):
        self.width = width
        self.height = height
        self.predators: List[Predator] = []
        self.prey: List[Prey] = []
        self.frame_count = 0
        self.attack_distance = attack_distance
        
        # Store all movement lines for persistence
        self.predator_lines = []  # List of (x1, y1, x2, y2) tuples
        self.prey_lines = []      # List of (x1, y1, x2, y2) tuples
        
        # Initialize agents
        self.spawn_predators(num_predators, attack_distance)
        self.spawn_prey(num_prey)
        
    def spawn_predators(self, count: int, attack_distance: float = 25.0):
        """Spawn predators at random locations"""
        for _ in range(count):
            x = random.uniform(0, self.width)
            y = random.uniform(0, self.height)
            predator = Predator(x, y, attack_distance=attack_distance)
            self.predators.append(predator)
            
    def spawn_prey(self, count: int):
        """Spawn prey at random locations"""
        for _ in range(count):
            x = random.uniform(0, self.width)
            y = random.uniform(0, self.height)
            prey = Prey(x, y)
            self.prey.append(prey)
    
    def update(self):
        """Update simulation by one timestep - optimized version"""
        if not self.prey:
            return
            
        # Convert agents to numpy arrays for vectorized operations
        prey_positions = np.array([[prey.x, prey.y] for prey in self.prey])
        predator_positions = np.array([[pred.x, pred.y] for pred in self.predators])
        
        # Phase 1: Predators hunt
        eaten_prey_indices = []
        for i, predator in enumerate(self.predators):
            prey_caught_idx = predator.hunt(prey_positions, self.width, self.height)
            if prey_caught_idx >= 0:
                eaten_prey_indices.append(prey_caught_idx)
            # Store predator movement line (handle toroidal wrapping)
            if predator.has_moved:
                self.predator_lines.append(self._wrap_line(predator.last_x, predator.last_y, predator.x, predator.y))
        
        # Remove eaten prey (in reverse order to avoid index issues)
        for idx in sorted(eaten_prey_indices, reverse=True):
            if idx < len(self.prey):
                del self.prey[idx]
        
        # Phase 2: Remaining prey flee from predators
        if self.prey and len(self.predators) > 0:
            # Update predator positions after hunting
            predator_positions = np.array([[pred.x, pred.y] for pred in self.predators])
            
            for prey in self.prey:
                prey.flee(predator_positions, self.width, self.height)
                # Store prey movement line (handle toroidal wrapping)
                if prey.has_moved:
                    self.prey_lines.append(self._wrap_line(prey.last_x, prey.last_y, prey.x, prey.y))
            
        self.frame_count += 1
    
    def _wrap_line(self, x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float, float, float]:
        """Handle line drawing across toroidal boundaries"""
        # Check if line crosses boundaries and needs to be drawn differently
        dx = x2 - x1
        dy = y2 - y1
        
        # If the movement is more than half the world size, it likely wrapped
        if abs(dx) > self.width / 2:
            # Line wraps horizontally
            if dx > 0:
                x2 = x2 - self.width
            else:
                x2 = x2 + self.width
                
        if abs(dy) > self.height / 2:
            # Line wraps vertically  
            if dy > 0:
                y2 = y2 - self.height
            else:
                y2 = y2 + self.height
        
        return (x1, y1, x2, y2)
        
    def draw_frame(self, show_sight_radius: bool = False, 
                   show_trajectories: bool = True) -> Figure:
        """Draw current frame of the simulation - optimized version"""
        # Use specific figure size to ensure even pixel dimensions for video encoding
        fig, ax = plt.subplots(figsize=(16, 9), dpi=120)  # Set DPI here to avoid recalculation
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        ax.set_facecolor('white')
        
        # Hide axes (faster method)
        ax.axis('off')
        
        # Draw all persistent movement trajectories (vectorized)
        if show_trajectories and (self.predator_lines or self.prey_lines):
            # Batch draw predator lines
            if self.predator_lines:
                pred_lines = np.array(self.predator_lines)
                for i in range(len(pred_lines)):
                    x1, y1, x2, y2 = pred_lines[i]
                    ax.plot([x1, x2], [y1, y2], color='red', linewidth=1, alpha=0.6)
            
            # Batch draw prey lines  
            if self.prey_lines:
                prey_lines = np.array(self.prey_lines)
                for i in range(len(prey_lines)):
                    x1, y1, x2, y2 = prey_lines[i]
                    ax.plot([x1, x2], [y1, y2], color='blue', linewidth=0.5, alpha=0.4)
        
        # Draw agents using scatter plots (much faster than individual circles)
        if self.predators:
            pred_x = [p.x for p in self.predators]
            pred_y = [p.y for p in self.predators]
            ax.scatter(pred_x, pred_y, c='red', s=200, alpha=0.8, marker='o')
        
        if self.prey:
            prey_x = [p.x for p in self.prey]
            prey_y = [p.y for p in self.prey]
            ax.scatter(prey_x, prey_y, c='blue', s=50, alpha=0.7, marker='o')
        
        return fig

def generate_animation_frames(output_dir: str = "predator_prey_frames", 
                            num_frames: int = 100,
                            world_size: Tuple[int, int] = (800, 600),
                            num_predators: int = 5,
                            num_prey: int = 30,
                            show_sight_radius: bool = False,
                            show_trajectories: bool = True):
    """Generate animation frames and save as PNG files"""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"🎬 Starting Predator-Prey Simulation Animation")
    print(f"📁 Output directory: {output_path}")
    print(f"🌍 World size: {world_size[0]}x{world_size[1]}")
    print(f"🔴 Predators: {num_predators}")
    print(f"🔵 Prey: {num_prey}")
    print(f"🎞️ Frames to generate: {num_frames}")
    print()
    
    # Initialize simulation
    sim = PredatorPreySimulation(
        width=world_size[0], 
        height=world_size[1],
        num_predators=num_predators,
        num_prey=num_prey
    )
    
    # Generate initial frame
    fig = sim.draw_frame(show_sight_radius=show_sight_radius, 
                        show_trajectories=show_trajectories)
    frame_path = output_path / f"frame_{0:04d}.png"
    # Use specific DPI to ensure even pixel dimensions (1920x1080 for 16:9 at 120 DPI)
    fig.savefig(frame_path, dpi=120, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"✅ Generated frame 0000")
    
    # Generate animation frames
    for frame in range(1, num_frames + 1):
        # Update simulation
        sim.update()
        
        # Check if all prey are eaten
        if len(sim.prey) == 0:
            print(f"🦴 All prey consumed at frame {frame}!")
            break
        
        # Draw and save frame
        fig = sim.draw_frame(show_sight_radius=show_sight_radius, 
                           show_trajectories=show_trajectories)
        frame_path = output_path / f"frame_{frame:04d}.png"
        # Use specific DPI to ensure even pixel dimensions (1920x1080 for 16:9 at 120 DPI)
        fig.savefig(frame_path, dpi=120, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        # Progress indicator
        if frame % 10 == 0:
            print(f"✅ Generated frame {frame:04d} - Predators: {len(sim.predators)}, Prey: {len(sim.prey)}")
    
    print(f"\n🎉 Animation generation complete!")
    print(f"📊 Final stats:")
    print(f"   - Frames generated: {sim.frame_count}")
    print(f"   - Predators remaining: {len(sim.predators)}")
    print(f"   - Prey remaining: {len(sim.prey)}")
    print(f"\n💡 To create video from frames:")
    print(f"   Basic command:")
    print(f"   ffmpeg -r 10 -i {output_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p predator_prey_animation.mp4")
    print(f"\n   If you get 'width not divisible by 2' error, use this command to force even dimensions:")
    print(f"   ffmpeg -r 10 -i {output_dir}/frame_%04d.png -vf \"scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2\" -c:v libx264 -pix_fmt yuv420p predator_prey_animation.mp4")

def create_video_from_frames(frames_dir: str, output_video: str, fps: int = 10) -> bool:
    """Create MP4 video from PNG frames using FFmpeg"""
    try:
        # Check if FFmpeg is available
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        
        # FFmpeg command with padding to ensure even dimensions
        cmd = [
            'ffmpeg', '-y',  # -y to overwrite output file
            '-r', str(fps),
            '-i', f'{frames_dir}/frame_%04d.png',
            '-vf', 'scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            output_video
        ]
        
        # Run FFmpeg
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            return True
        else:
            print(f"FFmpeg error: {result.stderr}")
            return False
            
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("FFmpeg not found. Please install FFmpeg to create videos.")
        return False

def run_simulation_with_feedback(world_width, world_height, num_predators, num_prey, 
                                predator_sight, prey_sight, prey_speed, attack_distance,
                                num_frames, fps, progress=gr.Progress()):
    """Run simulation with real-time progress feedback - optimized version"""
    
    # Create output directory
    output_dir = "predator_prey_gradio"
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Initialize simulation with custom parameters
    sim = PredatorPreySimulation(
        width=world_width, 
        height=world_height,
        num_predators=num_predators,
        num_prey=num_prey,
        attack_distance=attack_distance
    )
    
    # Update agent parameters
    for predator in sim.predators:
        predator.sight_radius = predator_sight
        predator.attack_distance = attack_distance
    
    for prey in sim.prey:
        prey.sight_radius = prey_sight
        prey.flee_speed = prey_speed
    
    frames_generated = 0
    status_updates = []
    
    # Pre-configure matplotlib for speed
    plt.ioff()  # Turn off interactive mode
    
    # Generate frames with progress updates
    progress(0, desc="Starting optimized simulation...")
    
    for frame in range(num_frames):
        # Update simulation
        if frame > 0:
            sim.update()
        
        # Check if all prey are eaten
        if len(sim.prey) == 0:
            status_updates.append(f"🦴 All prey consumed at frame {frame}!")
            break
        
        # Draw frame (optimized)
        fig = sim.draw_frame(show_sight_radius=False, show_trajectories=True)
        frame_path = output_path / f"frame_{frame:04d}.png"
        
        # Faster save settings
        fig.savefig(frame_path, format='png', bbox_inches='tight', 
                   facecolor='white', edgecolor='none', 
                   pad_inches=0, transparent=False)
        
        plt.close(fig)  # Important: close figure to free memory
        frames_generated += 1
        
        # Update progress less frequently for speed
        if frame % 5 == 0:
            progress_val = frame / num_frames
            progress(progress_val, desc=f"Frame {frame}/{num_frames} - Predators: {len(sim.predators)}, Prey: {len(sim.prey)}")
        
        # Status update every 50 frames (less frequent for speed)
        if frame % 50 == 0:
            status_updates.append(f"✅ Frame {frame:04d} - Predators: {len(sim.predators)}, Prey: {len(sim.prey)}")
    
    # Create video
    progress(0.9, desc="Creating video...")
    video_path = f"{output_dir}/predator_prey_animation.mp4"
    video_success = create_video_from_frames(output_dir, video_path, fps)
    
    # Final status
    final_status = f"""
🎉 Simulation Complete!
📊 Final Stats:
   - Frames generated: {frames_generated}
   - Predators remaining: {len(sim.predators)}
   - Prey remaining: {len(sim.prey)}
   - Video created: {'✅ Yes' if video_success else '❌ Failed'}
   - ⚡ Performance: Optimized vectorized calculations
"""
    
    progress(1.0, desc="Complete!")
    status_updates.append(final_status)
    
    # Return video path and status
    video_output = video_path if video_success and os.path.exists(video_path) else None
    
    return video_output, "\n".join(status_updates)

def create_gradio_interface():
    """Create Gradio interface for the predator-prey simulation"""
    
    with gr.Blocks(title="🎮 Predator-Prey Simulation") as interface:
        
        gr.HTML("""
        <div style="text-align: center; margin-bottom: 30px;">
            <h1>🎮 Predator-Prey Simulation</h1>
            <p>Configure parameters and generate animated simulations with real-time feedback</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("<h3>🌍 World Parameters</h3>")
                
                world_width = gr.Slider(
                    minimum=400, maximum=1500, value=1000, step=50,
                    label="World Width (pixels)",
                    info="Width of the simulation world"
                )
                
                world_height = gr.Slider(
                    minimum=300, maximum=1000, value=700, step=50,
                    label="World Height (pixels)",
                    info="Height of the simulation world"
                )
                
                gr.HTML("<h3>🐾 Agent Parameters</h3>")
                
                num_predators = gr.Slider(
                    minimum=1, maximum=15, value=6, step=1,
                    label="Number of Predators",
                    info="How many predators to spawn"
                )
                
                num_prey = gr.Slider(
                    minimum=5, maximum=100, value=40, step=5,
                    label="Number of Prey",
                    info="How many prey to spawn"
                )
                
                predator_sight = gr.Slider(
                    minimum=50, maximum=200, value=100, step=10,
                    label="Predator Sight Radius",
                    info="How far predators can see prey"
                )
                
                prey_sight = gr.Slider(
                    minimum=30, maximum=150, value=80, step=10,
                    label="Prey Sight Radius",
                    info="How far prey can see predators"
                )
                
                prey_speed = gr.Slider(
                    minimum=5, maximum=30, value=15, step=2,
                    label="Prey Flee Speed",
                    info="How fast prey move when fleeing"
                )
                
                attack_distance = gr.Slider(
                    minimum=10, maximum=50, value=25, step=5,
                    label="Predator Attack Distance",
                    info="Maximum distance predators can move per turn"
                )
                
                gr.HTML("<h3>🎬 Animation Parameters</h3>")
                
                num_frames = gr.Slider(
                    minimum=50, maximum=500, value=150, step=25,
                    label="Number of Frames",
                    info="How many frames to generate"
                )
                
                fps = gr.Slider(
                    minimum=5, maximum=30, value=10, step=1,
                    label="Video FPS",
                    info="Frames per second for output video"
                )
                
                # Control buttons
                generate_btn = gr.Button(
                    "🎬 Generate Simulation", 
                    variant="primary", 
                    size="lg"
                )
                
                gr.HTML("""
                <div style="margin-top: 20px; padding: 15px; background-color: #f0f8ff; border-radius: 8px;">
                    <h4>💡 Tips:</h4>
                    <ul>
                        <li><strong>Red circles:</strong> Predators hunting prey</li>
                        <li><strong>Blue circles:</strong> Prey fleeing from predators</li>
                        <li><strong>Red lines:</strong> Predator movement trails</li>
                        <li><strong>Blue lines:</strong> Prey escape routes</li>
                        <li><strong>Toroidal world:</strong> Edges wrap around</li>
                        <li><strong>Limited attack:</strong> Predators move step-by-step</li>
                    </ul>
                </div>
                """)
            
            with gr.Column(scale=2):
                gr.HTML("<h3>📊 Simulation Results</h3>")
                
                # Status output
                status_output = gr.Textbox(
                    label="📋 Status Log",
                    value="Ready to run simulation...",
                    lines=12,
                    max_lines=15,
                    interactive=False
                )
                
                # Video output
                video_output = gr.Video(
                    label="🎬 Generated Animation",
                    height=500
                )
                
                # Download section
                gr.HTML("""
                <div style="margin-top: 20px; padding: 15px; background-color: #f0f0f0; border-radius: 8px;">
                    <h4>💾 Downloads:</h4>
                    <p>• <strong>Video:</strong> Right-click video and "Save as..." or use download button</p>
                    <p>• <strong>All frames:</strong> Check the 'predator_prey_gradio' folder</p>
                    <p>• <strong>Toroidal world:</strong> Agents wrap around edges seamlessly</p>
                </div>
                """)
        
        # Connect the generate button
        generate_btn.click(
            fn=run_simulation_with_feedback,
            inputs=[world_width, world_height, num_predators, num_prey, 
                   predator_sight, prey_sight, prey_speed, attack_distance, 
                   num_frames, fps],
            outputs=[video_output, status_output]
        )
        
        # Preset configurations
        with gr.Row():
            gr.HTML("<h3>🎯 Quick Presets</h3>")
        
        with gr.Row():
            preset1_btn = gr.Button("🏃 Fast Hunt", size="sm")
            preset2_btn = gr.Button("🌊 Swarm Escape", size="sm") 
            preset3_btn = gr.Button("🎯 Precision Hunt", size="sm")
            preset4_btn = gr.Button("🌪️ Chaos Mode", size="sm")
        
        # Preset button functions
        def load_fast_hunt():
            return 800, 600, 3, 25, 120, 60, 20, 30, 100, 15
        
        def load_swarm_escape():
            return 1200, 800, 8, 60, 80, 100, 25, 20, 200, 12
        
        def load_precision_hunt():
            return 1000, 700, 4, 30, 150, 70, 12, 25, 150, 10
        
        def load_chaos_mode():
            return 1400, 900, 12, 80, 90, 90, 18, 35, 250, 15
        
        preset1_btn.click(
            fn=load_fast_hunt,
            outputs=[world_width, world_height, num_predators, num_prey, 
                    predator_sight, prey_sight, prey_speed, attack_distance, 
                    num_frames, fps]
        )
        
        preset2_btn.click(
            fn=load_swarm_escape,
            outputs=[world_width, world_height, num_predators, num_prey, 
                    predator_sight, prey_sight, prey_speed, attack_distance, 
                    num_frames, fps]
        )
        
        preset3_btn.click(
            fn=load_precision_hunt,
            outputs=[world_width, world_height, num_predators, num_prey, 
                    predator_sight, prey_sight, prey_speed, attack_distance, 
                    num_frames, fps]
        )
        
        preset4_btn.click(
            fn=load_chaos_mode,
            outputs=[world_width, world_height, num_predators, num_prey, 
                    predator_sight, prey_sight, prey_speed, attack_distance, 
                    num_frames, fps]
        )
    
    return interface

def main():
    """Launch Gradio interface"""
    print("🎮 Starting Predator-Prey Simulation Interface...")
    
    interface = create_gradio_interface()
    
    interface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True
    )

if __name__ == "__main__":
    main()
