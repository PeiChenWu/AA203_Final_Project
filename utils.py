import pickle
import os
    

def get_folder_name(filename):
    return '/'.join(filename.split('/')[:-1])

def maybe_makedirs(path_to_create):
    """This function will create a directory, unless it exists already,
    at which point the function will return.
    The exception handling is necessary as it prevents a race condition
    from occurring.
    Inputs:
        path_to_create - A string path to a directory you'd like created.
    """
    try:
        os.makedirs(path_to_create)
    except OSError:
        if not os.path.isdir(path_to_create):
            raise

def save_dict(di_, filename_):
    maybe_makedirs(get_folder_name(filename_))
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)
        


"""
Planar quadrotor animation using `matplotlib`.
Author: Spencer M. Richards
        Autonomous Systems Lab (ASL), Stanford
        (GitHub: spenrich)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import matplotlib.animation as animation


def animate_planar_quad_with_inverted_pendulum(t, x, y, θ, ϕ, title_string=None, display_in_notebook=False):
    """Animate the planar quadrotor system from given position data.
    All arguments are assumed to be 1-D NumPy arrays, where `x`, `y`, `θ`, and `ϕ`
    are the degrees of freedom of the planar quadrotor over time `t`.
    Example usage:
        import matplotlib.pyplot as plt
        from animations import animate_planar_quad
        fig, ani = animate_planar_quad_with_inverted_pendulum(t, x, y, θ, ϕ)
        ani.save('planar_quad.mp4', writer='ffmpeg')
        plt.show()
    """
    # Geometry
    rod_width = 2.
    rod_height = 0.15
    axle_height = 0.2
    axle_width = 0.05
    prop_width = 0.5 * rod_width
    prop_height = 1.5 * rod_height
    hub_width = 0.3 * rod_width
    hub_height = 2.5 * rod_height
    
    
    # Add pendulum 
    pole_length = rod_width
    mass_radius = 0.25
    
    

    # Figure and axis
    fig, ax = plt.subplots(dpi=100)
    x_min, x_max = np.min(x) - pole_length, np.max(x) + pole_length # increase x limit for pendulum
    x_pad = (rod_width + prop_width) / 2 + 0.1 * (x_max - x_min)
    y_min, y_max = np.min(y) - pole_length, np.max(y) + pole_length # increase y limit for pendulum
    y_pad = (rod_width + prop_width) / 2 + 0.1 * (y_max - y_min)
    ax.set_xlim([x_min - x_pad, x_max + x_pad])
    ax.set_ylim([y_min - y_pad, y_max + y_pad])
    ax.set_aspect(1.)
    if title_string is not None:
        plt.title(title_string)

    # Artists
    rod = mpatches.Rectangle((-rod_width / 2, -rod_height / 2),
                             rod_width,
                             rod_height,
                             facecolor='tab:blue',
                             edgecolor='k')
    hub = mpatches.FancyBboxPatch((-hub_width / 2, -hub_height / 2),
                                  hub_width,
                                  hub_height,
                                  facecolor='tab:blue',
                                  edgecolor='k',
                                  boxstyle='Round,pad=0.,rounding_size=0.05')
    axle_left = mpatches.Rectangle((-rod_width / 2, rod_height / 2),
                                   axle_width,
                                   axle_height,
                                   facecolor='tab:blue',
                                   edgecolor='k')
    axle_right = mpatches.Rectangle((rod_width / 2 - axle_width, rod_height / 2),
                                    axle_width,
                                    axle_height,
                                    facecolor='tab:blue',
                                    edgecolor='k')
    prop_left = mpatches.Ellipse(((axle_width - rod_width) / 2, rod_height / 2 + axle_height),
                                 prop_width,
                                 prop_height,
                                 facecolor='tab:gray',
                                 edgecolor='k',
                                 alpha=0.7)
    prop_right = mpatches.Ellipse(((rod_width - axle_width) / 2, rod_height / 2 + axle_height),
                                  prop_width,
                                  prop_height,
                                  facecolor='tab:gray',
                                  edgecolor='k',
                                  alpha=0.7)
    
    # Add pendulum parts pole & mass
    mass = mpatches.Circle((0., 0.), mass_radius, color='k')
    pole = ax.plot([], [], '-', linewidth=2, color='k')[0]
    
    patches = (rod, hub, axle_left, axle_right, prop_left, prop_right, mass) # Add pendulum mass
    for patch in patches:
        ax.add_patch(patch)
    trace = ax.plot([], [], '--', linewidth=2, color='tab:orange')[0]
    timestamp = ax.text(0.1, 0.9, '', transform=ax.transAxes)

    def animate(k, t, x, y, θ, ϕ):
        # Add Pendulum
        pole_start = np.array([x[k], y[k]])
        pole_end = pole_start + pole_length*np.array([np.sin(ϕ[k]),
                                                      -np.cos(ϕ[k])])
        
        # Pendulum
        pole.set_data([pole_start[0], pole_end[0]],
                      [pole_start[1], pole_end[1]])
        mass.set_center(pole_end)
        mass_x = x[:k+1] + pole_length*np.sin(ϕ[:k+1])
        mass_y = y[:k+1] - pole_length*np.cos(ϕ[:k+1])
        #trace.set_data(mass_x, mass_y) # if we want to trace pendulum mass
        
        
        transform = mtransforms.Affine2D().rotate_around(0., 0., θ[k])
        transform += mtransforms.Affine2D().translate(x[k], y[k])
        transform += ax.transData
    
        patches = (rod, hub, axle_left, axle_right, prop_left, prop_right) # we don't want to transform pendulum
        
        for patch in patches:
            patch.set_transform(transform)
        trace.set_data(x[:k + 1], y[:k + 1]) # if we want to trace the drone position
        timestamp.set_text('t = {:.1f} s'.format(t[k]))
        
        artists = patches + (mass, pole, trace, timestamp) # Add mass and pole (inverted pendulum)
        return artists

    dt = t[1] - t[0]
    step = max(int(np.floor((1 / 30) / dt)), 1)  # max out at 30Hz for faster rendering
    ani = animation.FuncAnimation(fig,
                                  animate,
                                  t[::step].size,
                                  fargs=(t[::step], x[::step], y[::step], θ[::step], ϕ[::step]), 
                                  interval=step * dt * 1000,
                                  blit=True)
    if display_in_notebook:
        try:
            get_ipython()
            from IPython.display import HTML
            ani = HTML(ani.to_html5_video())
        except (NameError, ImportError):
            raise RuntimeError("`display_in_notebook = True` requires this code to be run in jupyter/colab.")
    plt.close(fig)
    return ani