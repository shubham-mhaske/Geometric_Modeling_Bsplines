"""
B-Spline Visualization Functions
Professional visualization for B-spline curves, de Boor steps, Frenet frames, and segments.
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Set matplotlib style for professional appearance
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f9f9f9'
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'

# Professional color palette
COLOR_CURVE = '#e74c3c'
COLOR_POLYGON = '#7f7f7f'
COLOR_POINT = '#e74c3c'
COLOR_TANGENT = '#27ae60'
COLOR_NORMAL = '#3498db'
COLOR_BINORMAL = '#9b59b6'

def _get_fig_ax(ax=None, is_3d=False):
    """Create or retrieve figure and axis with proper styling."""
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        if is_3d:
            ax = fig.add_subplot(111, projection='3d')
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('lightgray')
            ax.yaxis.pane.set_edgecolor('lightgray')
            ax.zaxis.pane.set_edgecolor('lightgray')
        else:
            ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()
    return fig, ax

def _style_axis(ax, is_3d=False):
    """Apply consistent styling to axis."""
    if is_3d:
        ax.set_xlabel('X', fontsize=11, fontweight='bold', labelpad=10)
        ax.set_ylabel('Y', fontsize=11, fontweight='bold', labelpad=10)
        ax.set_zlabel('Z', fontsize=11, fontweight='bold', labelpad=10)
        ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
    else:
        ax.set_xlabel('X', fontsize=11, fontweight='bold')
        ax.set_ylabel('Y', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.25, linestyle=':', linewidth=0.7, color='gray')
        ax.set_axisbelow(True)
    
    ax.legend(loc='best', framealpha=0.9, edgecolor='gray', fontsize=9)
    ax.tick_params(labelsize=9)

def plot_bspline(control_points, curve_points, ax=None):
    """Plot B-spline curve with control polygon."""
    is_3d = control_points.shape[1] == 3
    fig, ax = _get_fig_ax(ax, is_3d=is_3d)
    
    if is_3d:
        ax.plot(control_points[:,0], control_points[:,1], control_points[:,2], 
                'o--', color=COLOR_POLYGON, markersize=7, linewidth=1.5, 
                label='Control Polygon', alpha=0.7)
        ax.plot(curve_points[:,0], curve_points[:,1], curve_points[:,2], 
                '-', color=COLOR_CURVE, linewidth=2.5, label='B-Spline Curve', alpha=0.9)
    else:
        ax.plot(control_points[:,0], control_points[:,1], 
                'o--', color=COLOR_POLYGON, markersize=8, linewidth=1.5, 
                label='Control Polygon', alpha=0.7, zorder=3)
        ax.plot(curve_points[:,0], curve_points[:,1], 
                '-', color=COLOR_CURVE, linewidth=2.5, label='B-Spline Curve', 
                alpha=0.95, zorder=4)

    # Set fixed aspect ratio and padded limits to stabilize animation
    all_points = np.vstack([control_points, curve_points])
    min_coords = all_points.min(axis=0)
    max_coords = all_points.max(axis=0)
    
    center = (min_coords + max_coords) / 2
    size = max_coords - min_coords
    max_size = max(size) if max(size) > 1e-6 else 1.0
    
    # Add padding to accommodate Frenet frame vectors
    padding = max_size * 0.2
    
    if is_3d:
        ax.set_xlim(center[0] - max_size/2 - padding, center[0] + max_size/2 + padding)
        ax.set_ylim(center[1] - max_size/2 - padding, center[1] + max_size/2 + padding)
        ax.set_zlim(center[2] - max_size/2 - padding, center[2] + max_size/2 + padding)
    else:
        ax.set_xlim(center[0] - max_size/2 - padding, center[0] + max_size/2 + padding)
        ax.set_ylim(center[1] - max_size/2 - padding, center[1] + max_size/2 + padding)
        ax.set_aspect('equal', adjustable='box')

    ax.set_title("B-Spline Curve", fontsize=14, fontweight='bold', pad=15)
    _style_axis(ax, is_3d)
    return fig, ax

def plot_spline_segments(curve_points, knot_vector, degree, ax):
    """Color different spline segments by knot intervals."""
    fig = ax.get_figure()
    is_3d = curve_points.shape[1] == 3
    n_segments = len(knot_vector) - 1 - degree
    cmap = plt.get_cmap('viridis', n_segments)
    
    for i in range(n_segments):
        start = int(i * len(curve_points) / n_segments)
        end = int((i+1) * len(curve_points) / n_segments)
        color = cmap(i)
        
        if is_3d:
            ax.plot(curve_points[start:end,0], curve_points[start:end,1], 
                   curve_points[start:end,2], color=color, linewidth=2.5, 
                   label=f'Segment {i+1}', alpha=0.9)
        else:
            ax.plot(curve_points[start:end,0], curve_points[start:end,1], 
                   color=color, linewidth=2.5, label=f'Segment {i+1}', 
                   alpha=0.9, zorder=4)
    
    ax.set_title("B-Spline Segments (Colored by Knot Intervals)", 
                fontsize=14, fontweight='bold', pad=15)
    _style_axis(ax, is_3d)
    return fig, ax

def illustrate_de_boor_steps(steps, ax):
    """Visualize all de Boor algorithm levels on the main plot."""
    fig = ax.get_figure()
    is_3d = steps[0].shape[1] == 3
    num_levels = len(steps)
    cmap = plt.get_cmap('plasma', num_levels)
    
    for r, pts in enumerate(steps):
        color = cmap(r)
        if is_3d:
            ax.plot(pts[:,0], pts[:,1], pts[:,2], 'o-', color=color, 
                   markersize=6, linewidth=1.5, label=f'Level {r}', alpha=0.7)
        else:
            ax.plot(pts[:,0], pts[:,1], 'o-', color=color, 
                   markersize=7, linewidth=1.8, label=f'Level {r}', 
                   alpha=0.75, zorder=3+r)
    
    final_point = steps[-1][-1]
    if is_3d:
        ax.plot([final_point[0]], [final_point[1]], [final_point[2]], 
               'o', color=COLOR_POINT, markersize=12, label='Final Point', zorder=10)
    else:
        ax.plot(final_point[0], final_point[1], 
               'o', color=COLOR_POINT, markersize=12, label='Final Point', zorder=10)
    
    ax.set_title("de Boor Algorithm - Recursive Blending Steps", 
                fontsize=14, fontweight='bold', pad=15)
    _style_axis(ax, is_3d)
    return fig, ax

def illustrate_de_boor_steps_separate(steps):
    """Plot each de Boor level in separate subplots."""
    num_levels = len(steps)
    is_3d = steps[0].shape[1] == 3
    
    cols = min(num_levels, 3)
    rows = (num_levels + cols - 1) // cols
    fig = plt.figure(figsize=(5 * cols, 5 * rows))
    fig.patch.set_facecolor('white')
    
    for r, pts in enumerate(steps):
        if is_3d:
            ax = fig.add_subplot(rows, cols, r + 1, projection='3d')
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.plot(pts[:,0], pts[:,1], pts[:,2], 
                   'o-', markersize=10, linewidth=2.5, color='#3498db', alpha=0.9)
            ax.set_xlabel('X', fontsize=10, fontweight='bold')
            ax.set_ylabel('Y', fontsize=10, fontweight='bold')
            ax.set_zlabel('Z', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
        else:
            ax = fig.add_subplot(rows, cols, r + 1)
            ax.plot(pts[:,0], pts[:,1], 
                   'o-', markersize=10, linewidth=2.5, color='#3498db', alpha=0.9, zorder=4)
            ax.set_xlabel('X', fontsize=10, fontweight='bold')
            ax.set_ylabel('Y', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.25, linestyle=':', linewidth=0.7, color='gray')
            ax.set_axisbelow(True)
        
        ax.set_title(f"Level {r}", fontsize=12, fontweight='bold', pad=12)
        ax.tick_params(labelsize=9)
    
    fig.suptitle("de Boor Evaluation - Individual Levels", 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

def plot_frenet_frame(point, tangent, normal, binormal, ax, scale=1.5, show_magnitudes=True):
    """Plot Frenet frame with optional magnitude display."""
    fig = ax.get_figure()
    is_3d = point.shape[0] == 3
    
    tangent_mag = np.linalg.norm(tangent)
    normal_mag = np.linalg.norm(normal)
    binormal_mag = np.linalg.norm(binormal) if binormal is not None else 0
    
    tangent_display = tangent * scale
    normal_display = normal * scale
    binormal_display = binormal * scale if binormal is not None else None
    
    if is_3d:
        ax.plot([point[0]], [point[1]], [point[2]], 'o', 
               color=COLOR_POINT, markersize=12, label='Point', zorder=10)
        
        t_label = f'Tangent |T|={tangent_mag:.2f}' if show_magnitudes else 'Tangent'
        n_label = f'Normal |N|={normal_mag:.2f}' if show_magnitudes else 'Normal'
        b_label = f'Binormal |B|={binormal_mag:.2f}' if show_magnitudes else 'Binormal'
        
        ax.quiver(point[0], point[1], point[2], 
                 tangent_display[0], tangent_display[1], tangent_display[2], 
                 color=COLOR_TANGENT, linewidth=2.5, label=t_label,
                 arrow_length_ratio=0.2, alpha=0.9)
        ax.quiver(point[0], point[1], point[2], 
                 normal_display[0], normal_display[1], normal_display[2], 
                 color=COLOR_NORMAL, linewidth=2.5, label=n_label,
                 arrow_length_ratio=0.2, alpha=0.9)
        if binormal_display is not None:
            ax.quiver(point[0], point[1], point[2], 
                     binormal_display[0], binormal_display[1], binormal_display[2], 
                     color=COLOR_BINORMAL, linewidth=2.5, label=b_label,
                     arrow_length_ratio=0.2, alpha=0.9)
    else:
        ax.plot(point[0], point[1], 'o', 
               color=COLOR_POINT, markersize=12, label='Point', zorder=10)
        
        ax.annotate('', 
                   xy=(point[0] + tangent_display[0], point[1] + tangent_display[1]),
                   xytext=(point[0], point[1]),
                   arrowprops=dict(arrowstyle='->', color=COLOR_TANGENT, lw=2.5, alpha=0.9),
                   zorder=8)
        ax.annotate('', 
                   xy=(point[0] + normal_display[0], point[1] + normal_display[1]),
                   xytext=(point[0], point[1]),
                   arrowprops=dict(arrowstyle='->', color=COLOR_NORMAL, lw=2.5, alpha=0.9),
                   zorder=8)
        
        ax.text(point[0] + tangent_display[0]*1.15, point[1] + tangent_display[1]*1.15, 
               'T', color=COLOR_TANGENT, fontsize=9, fontweight='bold')
        ax.text(point[0] + normal_display[0]*1.15, point[1] + normal_display[1]*1.15, 
               'N', color=COLOR_NORMAL, fontsize=9, fontweight='bold')

        # Create proxy artists for legend
        label_t = f'Tangent |T|={tangent_mag:.2f}' if show_magnitudes else 'Tangent'
        label_n = f'Normal |N|={normal_mag:.2f}' if show_magnitudes else 'Normal'
        ax.plot([], [], color=COLOR_TANGENT, lw=2.5, label=label_t)
        ax.plot([], [], color=COLOR_NORMAL, lw=2.5, label=label_n)
    
    ax.set_title("Frenet Frame", fontsize=14, fontweight='bold', pad=15)
    _style_axis(ax, is_3d)
    return fig, ax