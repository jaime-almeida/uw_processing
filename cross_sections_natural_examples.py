import matplotlib.pyplot as plt

from uw_model_old import *

# plt.style.use('PaperDoubleFig.mplstyle')

# Initiate the model object
model = uw_model(model_dir=r'D:\\3D_AgeTest\\model_results\\3D_block_S', ts=500, scf=1e22)

# Get the variables we'll need
model.get_all()

# Create a slice along the middle of the plateau
z_profile = 2.6e6
model.set_slice('z', z_profile, find_closest=True)

# Correct the depth:
model.correct_depth('y')

# Find the trench:
trench_position = model.find_trench(horizontal_direction='x',
                                    vertical_direction='y')

# Set a window:
model.set_window(hmin=trench_position - 600e3,
                 hmax=trench_position + 600e3,
                 vmin=0,
                 vmax=3e5)

# Reinterpolate and plot it:
X, Y, mat_int = model.reinterpolate_window(hdir='x', vdir='y',
                                           variable=model.output['material'].mat)
_, _, temp_int = model.reinterpolate_window(hdir='x', vdir='y',
                                            variable=model.output['temperature'].C)
_, _, eta_int = model.reinterpolate_window(hdir='x', vdir='y',
                                           variable=model.output['viscosity'].eta)

# Plot it
plt.close('all')
fig, ax = plt.subplots(figsize=(61 / 25.4, 60 / 25.4))
# mat_fig = ax.pcolormesh(X / 1e3, Y / 1e3, mat_int, cmap='RdYlBu_r')
mat_fig = ax.pcolormesh(X / 1e3, Y / 1e3, eta_int, cmap='coolwarm')
# ax.contour(X/1e3, Y/1e3, temp_int, levels=[1000], colors='k', linewidths=.5)
ax.set_ylim(ax.get_ylim()[::-1])
ax.axis('equal')
ax.axis('off')

plt.show()

plt.savefig(r'D:\\3D_AgeTest\\PaperFigures\\Figure_ X_natural\\model_section_2_eta.tiff',
            bbox_inches='tight',
            dpi=600,
            transparent=True)
