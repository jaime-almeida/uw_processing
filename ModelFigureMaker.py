from uw_model import UwModel
import numpy as np
import matplotlib.pyplot as plt
import os as os
import pandas as pd

plt.ioff()


def add_quiver(ax, cx, cy, vx, vy, narrows=500, scale=5):
    index = np.random.choice(np.arange(0, np.ravel(cx).shape[0]), narrows)
    index = index.astype(int)

    # Get only x amount of points
    x  = np.ravel(cx)[index]
    y  = np.ravel(cy)[index]
    vx_u = np.ravel(vx)[index]
    vy_u = np.ravel(vy)[index]


    # MAke the patch
    # patch_x = x.min() + 0.78 * (x.max() - x.min())
    # patch_y = y.max() - 0.12 * (y.max() - y.min())
    qv = ax.quiver(x, y, vx_u, vy_u,
                   color='k',
                   scale=scale,
                   scale_units='inches',
                   angles='uv',
                   pivot='middle',
                   width=1.3e-3,
                   headwidth=5,
                   headlength=5, zorder=2)

    # patch = patches.Rectangle((patch_x, patch_y),
    #                           width=153, height=23,
    #                           edgecolor='k', facecolor='w',
    #                           lw=.75)

    # ax.add_patch(patch)

    qk = ax.quiverkey(qv, X=.8, Y=.16, U=2,
                      label=r'2 cm/yr', labelpos='E', zorder=10)

    return qv, qk


# # Define the age ranges for the models:
op_age = [30] #np.arange(30, 100, 10)
dp_age = [90] #np.arange(10, 100, 10)


for op in op_age:
    for dp in dp_age:
        print('Currently in model:  {}OP_{}DP'.format(str(op), str(dp)))

        # Initiate the uw object:
        try:
            model = UwModel(model_dir=r'..\model_results\{}OP_{}DP'.format(str(op), str(dp)), scf=1e23)
        except OSError:
            continue

        # Make a folder to save this on
        folder = r'..\ModelImages\{}'.format(model.model_name.split('\\')[-1])

        # Initiate the trench position array:
        trench_data = []

        try:
            os.mkdir(folder)
        except FileExistsError:
            pass

        # For a ridiculously long ts chain
        for ts in np.arange(0, 3000, 50):
            try:
                model.set_current_ts(step=ts)
            except OSError:
                break

            # Get the variables:
            model.get_velocity()
            model.get_temperature()
            model.get_viscosity()
            model.get_material()

            # Correct the depth:
            model.correct_depth()

            # Interpolate the variables:
            X, Y, mat = model.reinterpolate_window(variable=model.output.mat)
            _, _, T_C = model.reinterpolate_window(variable=model.output.temp_C)
            _, _, eta = model.reinterpolate_window(variable=model.output.eta)

            # Velocities
            _, _, vx = model.reinterpolate_window(variable=model.output.vx)
            _, _, vy = model.reinterpolate_window(variable=model.output.vy)

            # Get the trench position:
            trench_position = model.find_trench()

            # append it to the array:
            trench_data.append([model.time_Ma, trench_position])

            # ====================== MAKE THE FIGURE ==========================
            # Make the figure:
            fig, ax = plt.subplots(nrows=2, figsize=(9.26, 4.84))

            im_mat = ax[0].pcolormesh(X/1e3, Y/1e3, mat, cmap='RdYlBu_r')
            im_eta = ax[1].pcolormesh(X/1e3, Y/1e3, eta, cmap='coolwarm')

            ax[0].set_title('Current TS: {}, Time (Ma): {}'.format(str(ts), str(model.time_Ma)))

            for ax_i in ax:
                ax_i.set_xlabel('X, km')
                ax_i.set_ylabel('Y, km')
                ax_i.axis('image')
                ax_i.invert_yaxis()
                t_cont = ax_i.contour(X / 1e3,  Y / 1e3, T_C, levels=[1000], colors='k', linewidths=1.3)
                clabels = plt.clabel(t_cont, fmt='%g', fontsize=11,
                                     manual=[((X.min() + .3 * (X.max() - X.min())) / 1e3, 150)])
                # Mark the trench on the plots
                ax_i.scatter(trench_position/1e3, -10, marker='v', color='k', s=10, zorder=10, clip_on=False)

            plt.colorbar(im_mat, ax=ax[0])
            plt.colorbar(im_eta, ax=ax[1])

            fig.tight_layout()
            plt.savefig('{}/overview_{}.jpg'.format(folder, str(ts)), dpi=300)

            plt.close('all')

        # Save the output trench data:
        output_data = pd.DataFrame(data=trench_data, columns={'Time', 'TrenchPosition'})
        output_data.to_excel('{}/trench_data.xls'.format(folder))
