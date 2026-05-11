"""
Fourier Epicycles Animation - Star Shape
Visualize how a star curve can be drawn by rotating circles (epicycles)
"""
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Circle as MplCircle
import os

# Set font to avoid Chinese character issues
plt.rcParams['font.family'] = 'Arial'

OUTPUT_DIR = 'figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== Generate Star Shape ====================

def generate_star(n=200, peaks=5):
    """Generate a 5-pointed star"""
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    r_outer = 3.0
    r_inner = 1.2
    r = r_outer + (r_inner - r_outer) * (1 + np.cos(peaks * t)) / 2
    x = r * np.cos(t)
    y = r * np.sin(t)
    return np.column_stack([x, y])

# ==================== Compute Fourier Coefficients ====================

def compute_dft_coefficients(points):
    """Compute Discrete Fourier Transform coefficients"""
    z = points[:, 0] + 1j * points[:, 1]
    N = len(z)
    coeffs = {}
    for k in range(-(N//2), N//2 + 1):
        c_k = np.sum(z * np.exp(-2j * np.pi * k * np.arange(N) / N)) / N
        coeffs[k] = c_k
    return coeffs

def get_sorted_coeffs(coeffs, n_terms):
    """Sort coefficients by amplitude"""
    sorted_keys = sorted(coeffs.keys(), key=lambda k: abs(coeffs[k]), reverse=True)
    return [(k, coeffs[k]) for k in sorted_keys[:n_terms]]

# ==================== GUI Animation Class ====================

class FourierStarGUI:
    def __init__(self):
        self.is_playing = True
        self.current_frame = 0
        self.n_frames = 300
        self.n_terms = 10
        self.save_counter = 0
        
        # Generate star and compute coefficients
        self.points = generate_star(200)
        self.all_coeffs = compute_dft_coefficients(self.points)
        self.update_coeffs()
        
        # Trace storage
        self.trace_x = []
        self.trace_y = []
        
        self.setup_gui()
    
    def update_coeffs(self):
        """Update Fourier coefficients"""
        self.sorted_coeffs = get_sorted_coeffs(self.all_coeffs, self.n_terms)
    
    def setup_gui(self):
        """Create GUI interface"""
        self.fig = plt.figure(figsize=(12, 8))
        self.fig.patch.set_facecolor('#1a1a2e')
        
        # Main animation area
        self.ax_main = self.fig.add_axes([0.05, 0.2, 0.9, 0.75])
        self.ax_main.set_facecolor('#16213e')
        self.ax_main.set_aspect('equal')
        self.ax_main.grid(True, alpha=0.2, color='white')
        self.ax_main.tick_params(colors='white')
        for spine in self.ax_main.spines.values():
            spine.set_color('white')
        
        # Slider for number of terms
        ax_slider = self.fig.add_axes([0.15, 0.08, 0.35, 0.03])
        self.slider_terms = Slider(ax_slider, 'Terms N', 1, 30, valinit=self.n_terms, 
                                   valstep=1, color='#e94560')
        self.slider_terms.label.set_color('white')
        self.slider_terms.valtext.set_color('white')
        self.slider_terms.on_changed(self.on_slider_change)
        
        # Play/Pause button
        ax_btn_play = self.fig.add_axes([0.55, 0.075, 0.08, 0.04])
        self.btn_play = Button(ax_btn_play, 'Pause', color='#e94560', hovercolor='#ff6b6b')
        self.btn_play.label.set_color('white')
        self.btn_play.on_clicked(self.toggle_play)
        
        # Save frame button
        ax_btn_save = self.fig.add_axes([0.65, 0.075, 0.08, 0.04])
        self.btn_save = Button(ax_btn_save, 'Save', color='#4CAF50', hovercolor='#66BB6A')
        self.btn_save.label.set_color('white')
        self.btn_save.on_clicked(self.save_frame)
        
        # Title
        self.fig.suptitle('Fourier Epicycles - Star Shape', fontsize=16, 
                         color='white', fontweight='bold')
        
        # Initialize plot elements
        self.init_plots()
        
        # Start animation
        self.anim = FuncAnimation(self.fig, self.animate, frames=self.n_frames,
                                  interval=33, blit=False, repeat=True)
        plt.show()
    
    def init_plots(self):
        """Initialize plot elements"""
        # Original curve (gray dashed line)
        self.ax_main.plot(self.points[:, 0], self.points[:, 1], 
                         '--', color='gray', alpha=0.4, linewidth=1.5, label='Original')
        
        # Trace line
        self.line_trace, = self.ax_main.plot([], [], '-', color='#e94560', 
                                              linewidth=2.5, alpha=0.9, label='Fourier')
        
        # Circles and lines
        self.circles = []
        self.lines = []
        self.dot, = self.ax_main.plot([], [], 'o', color='#00ff88', 
                                       markersize=10, zorder=10)
        
        # Set limits
        pad = 1
        xmin, xmax = self.points[:, 0].min() - pad, self.points[:, 0].max() + pad
        ymin, ymax = self.points[:, 1].min() - pad, self.points[:, 1].max() + pad
        self.ax_main.set_xlim(xmin, xmax)
        self.ax_main.set_ylim(ymin, ymax)
        
        # Info text
        self.text_info = self.ax_main.text(0.02, 0.98, '', transform=self.ax_main.transAxes,
                                            fontsize=11, color='white', va='top',
                                            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        self.ax_main.legend(loc='upper right', fontsize=10, 
                           facecolor='#16213e', edgecolor='white', labelcolor='white')
    
    def animate(self, frame):
        """Animation update function"""
        if not self.is_playing:
            return
        
        self.current_frame = (self.current_frame + 1) % self.n_frames
        t = self.current_frame / self.n_frames
        
        # Clear old circles
        for c in self.circles:
            c.remove()
        for l in self.lines:
            l.remove()
        self.circles = []
        self.lines = []
        
        # Calculate epicycle chain
        x, y = 0.0, 0.0
        
        for k, c_k in self.sorted_coeffs:
            radius = abs(c_k)
            if radius < 1e-6:
                continue
            
            # Draw circle
            circle = MplCircle((x, y), radius, fill=False, 
                              color='#00d2ff', alpha=0.35, linewidth=1.2)
            self.ax_main.add_patch(circle)
            self.circles.append(circle)
            
            # Calculate next point
            angle = 2 * np.pi * k * t + np.angle(c_k)
            new_x = x + radius * np.cos(angle)
            new_y = y + radius * np.sin(angle)
            
            # Draw connecting line
            line, = self.ax_main.plot([x, new_x], [y, new_y], '-', 
                                      color='#00d2ff', alpha=0.7, linewidth=1.8)
            self.lines.append(line)
            
            x, y = new_x, new_y
        
        # Update pen tip
        self.dot.set_data([x], [y])
        
        # Update trace
        self.trace_x.append(x)
        self.trace_y.append(y)
        
        if len(self.trace_x) > self.n_frames:
            self.trace_x = self.trace_x[-self.n_frames:]
            self.trace_y = self.trace_y[-self.n_frames:]
        
        self.line_trace.set_data(self.trace_x, self.trace_y)
        
        # Update info
        self.text_info.set_text(f'Terms: {self.n_terms}\nFrame: {self.current_frame}/{self.n_frames}\nt = {t:.3f}')
    
    def on_slider_change(self, val):
        """Slider value changed callback"""
        self.n_terms = int(val)
        self.update_coeffs()
        self.trace_x = []
        self.trace_y = []
        self.current_frame = 0
    
    def toggle_play(self, event):
        """Toggle play/pause"""
        self.is_playing = not self.is_playing
        self.btn_play.label.set_text('Play' if not self.is_playing else 'Pause')
    
    def save_frame(self, event):
        """Save current frame"""
        self.save_counter += 1
        filename = f'exp5_star_frame_{self.save_counter}.png'
        filepath = os.path.join(OUTPUT_DIR, filename)
        self.fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
        print(f'Saved: {filename}')

# ==================== Main ====================

if __name__ == '__main__':
    print("Fourier Epicycles Animation - Star Shape")
    print("Controls:")
    print("  - Drag slider to adjust number of Fourier terms")
    print("  - Click 'Pause' to pause/resume animation")
    print("  - Click 'Save' to save current frame")
    print("  - Close window to exit")
    
    gui = FourierStarGUI()
