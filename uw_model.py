# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 13:34:51 2019

@author: jaime
#"""
# import re as re
# import os as os
import h5py as h5
import numpy as np
import pandas as pd
from multiprocessing import Pool
import gc as gc
import re as re
from breshenam_digit import *

cmy = 365 * 24 * 60 * 60. * 100


class UserChoice(Exception):
    def __init__(self, message):
        self.message = message


# Print iterations progress
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def get_model_name(model_dir):
    if model_dir[-1] == '/':
        model_dir -= '/'

    return re.split('/', model_dir)[-1]


def parallelize_dataframe(df, func, n_cores=4):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


def velocity_rescale(df, scf):
    df = df / scf * cmy
    return df


def viscosity_rescale(df, scf):
    df = np.log10(df * scf)
    return df


def dim_eval(res):
    # Not likely to be a 1D model.
    if len(res) > 2:
        return 3
    else:
        return 2


def get_res(model_dir):
    # Make  the file path
    filename = model_dir + 'Mesh.linearMesh.00000.h5'

    # Read everything
    data = h5.File(filename, 'r')
    res = data.attrs['mesh resolution']

    # Get the dimensions:
    ndims = dim_eval(res)

    if ndims == 2:
        return {'x': res[0] + 1, 'y': res[1] + 1}, ndims
    else:
        return {'x': res[0] + 1, 'y': res[1] + 1, 'z': res[2] + 1}, ndims


def ts_writer(ts_in):
    # Making the timestep text:
    return str(ts_in).zfill(5)


def get_time(mdir, ts):
    data = h5.File(mdir + 'timeInfo.' + ts + '.h5', 'r')

    time_out = data['currentTime'][0]

    return time_out


def get_nproc(mdir):
    data = h5.File(mdir + '/timeInfo.00000.h5', 'r')

    return data['nproc'][0]


# %%
class UwModel:

    def __init__(self, model_dir, ts=0, scf=1e22):
        if model_dir[-1] != '/':
            self.model_dir = model_dir + '/'
        else:
            self.model_dir = model_dir

        self.res, self.dim = get_res(self.model_dir)
        # Cores are not needed for now.

        # Initiate a boundary coordinate
        self.boundary = {}

        # Set the default scaling:
        self.scf = scf

        # Save the model name
        self.model_name = get_model_name(model_dir)

        # Save an empty list/dict for any slicing that will be done
        self.performed_slices = []

        # Get the number of processors used
        self.nproc = get_nproc(model_dir)

        # set th initial timestep:
        try:
            self.set_current_ts(step=ts)
        except KeyError:
            raise ValueError('Timestep {} not present.'.format(ts))

        # Initiate a output dataframe
        self.output = None
        self._get_mesh()

    def set_current_ts(self, step):
        # Set the timestep for further processes
        if step > 1e5:
            raise Exception('Max timestep is 10,000.')

        # If a previous iteration of the model exists:
        try:
            if self.current_step:
                #                print('A previous iteration was detected, replacing the output.')

                # Clean the output dictionary:
                self.output = {}

                # Supposedly help clean further
                gc.collect()
        except:
            pass

        # Set the current TS
        self.current_step = str(step).zfill(5)

        # Get the current time and descale it
        self.time_Ma = np.round(get_time(self.model_dir, self.current_step) * self.scf / (365 * 24 * 3600) / 1e6, 3)

    # def set_scaling_factor(self, scf):
    # # Set the timestep for further processes
    # self.scf = float(scf)

    ##################################################
    #              RETRIEVING INFORMATION            #
    ##################################################
    def get_all(self):
        """
        Function to get all existing variables from the current working directory.
        """
        print('Getting all variables...')
        self.get_velocity()
        self.get_material()
        self.get_strain()
        self.get_stress()
        self.get_viscosity()
        self.get_temperature()

    # Get mesh information:
    def _get_mesh(self):
        # Set the file path:
        filename = self.model_dir + 'Mesh.linearMesh.' + \
                   self.current_step + '.h5'

        # Read the h5 file:
        data = h5.File(filename, 'r')

        # Get the information from the file:
        mesh_info = data['vertices'][()]

        # Write the info accordingly:
        if self.dim == 2:
            self.output = pd.DataFrame(data=mesh_info, columns=['x', 'y'], dtype='float')
        else:
            # in 3D:
            self.output = pd.DataFrame(data=mesh_info, columns=['x', 'y', 'z'], dtype='float')

        # Save the model dimensions:
        axes = self.output.columns.values
        max_dim = self.output.max().values
        min_dim = self.output.min().values

        for axis, min_val, max_val in zip(axes, min_dim, max_dim):
            self.boundary[axis] = [min_val, max_val]

    def get_velocity(self):
        try:
            self.scf
        except NameError:
            raise ValueError('No Scaling Factor detected!')

        if type(self.output) == dict:
            self._get_mesh()

        # Set the file path:
        filename = self.model_dir + 'VelocityField.' + \
                   self.current_step + '.h5'

        # Read the h5 file:
        data = h5.File(filename, 'r')

        # Get the information from the file:
        vel_info = data['data'][()]

        # Write the info accordingly:
        if self.dim == 2:
            velocity = pd.DataFrame(data=vel_info, columns=['vx', 'vy'])
        else:
            # in 3D:
            velocity = pd.DataFrame(data=vel_info, columns=['vx', 'vy', 'vz'])

        # Rescale
        velocity = velocity_rescale(velocity, self.scf)

        # Merge with the current output dataframe
        self.output = self.output.merge(velocity, left_index=True, right_index=True)

    def get_viscosity(self):
        try:
            self.scf
        except:
            raise ValueError('No Scaling Factor detected!')

        if self.output is None:
            self._get_mesh()

            # Set the file path:
        filename = self.model_dir + 'ViscosityField.' + \
                   self.current_step + '.h5'

        # Read the h5 file:
        data = h5.File(filename, 'r')

        # Get the information from the file:
        mat_info = data['data'][()]

        # Write the info accordingly:

        viscosity = pd.DataFrame(data=mat_info,
                                 columns=['eta'])

        # Rescale
        viscosity = viscosity_rescale(viscosity, self.scf)

        # Merge:
        self.output = self.output.merge(viscosity, left_index=True, right_index=True)

    def get_material(self):
        # Set the file path:
        filename = self.model_dir + 'MaterialIndexField.' + \
                   self.current_step + '.h5'

        # Read the h5 file:
        data = h5.File(filename, 'r')

        # Get the information from the file:
        mat_info = data['data'][()]

        # Write the info accordingly:
        material = pd.DataFrame(data=mat_info, columns=['mat'])

        # Merge
        self.output = self.output.merge(material, left_index=True, right_index=True)

    def get_temperature(self):
        # Set the file path:
        filename = self.model_dir + 'TemperatureField.' + \
                   self.current_step + '.h5'

        # Read the h5 file:
        data = h5.File(filename, 'r')

        # Get the information from the file:
        temp_info = data['data'][()]

        # Write the info accordingly:
        temperature = pd.DataFrame(data=temp_info, columns=['temp_K'])
        temperature['temp_C'] = temperature.temp_K - 273.15

        # Merge:
        self.output = self.output.merge(temperature, left_index=True, right_index=True)

    # Get the strain information 
    def get_strain(self):
        # Set the file path:
        filename = self.model_dir + 'recoveredStrainRateField.' + \
                   self.current_step + '.h5'
        filename2 = self.model_dir + 'recoveredStrainRateInvariantField.' + \
                    self.current_step + '.h5'

        # Read the h5 file:
        data = h5.File(filename, 'r')

        invariant = True
        try:
            data2 = h5.File(filename2, 'r')
        except OSError:
            invariant = False

        # Get the information from the file:
        strain_info = data['data'][()]

        if invariant:
            invariant_info = data2['data'][()]

        # Write the info accordingly:
        if self.dim == 2:
            strain = pd.DataFrame(data=strain_info,
                                  columns=['e_xx', 'e_yy', 'e_xy'])
        else:
            # in 3D:
            strain = pd.DataFrame(data=strain_info,
                                  columns=['e_xx', 'e_yy', 'e_zz',
                                           'e_xy', 'e_xz', 'e_yz'])

        # Add the invariant
        if invariant:
            strain['e_II'] = invariant_info

        # Merge
        self.output = self.output.merge(strain, left_index=True, right_index=True)

    # Get the stress information
    def get_stress(self):
        # Set the file path:
        filename = self.model_dir + 'recoveredDeviatoricStressField.' + \
                   self.current_step + '.h5'
        filename2 = self.model_dir + 'recoveredDeviatoricStressInvariantField.' + \
                    self.current_step + '.h5'

        # Read the h5 file:
        data = h5.File(filename, 'r')
        invariant = True
        try:
            data2 = h5.File(filename2, 'r')
        except OSError:
            invariant = False
        # Get the information from the file:
        stress_info = data['data'][()]

        if invariant:
            invariant_info = data2['data'][()]

        # Write the info accordingly:
        if self.dim == 2:
            stress = pd.DataFrame(data=stress_info,
                                  columns=['s_xx', 's_yy', 's_xy'])
        else:
            # in 3D:
            stress = pd.DataFrame(data=stress_info,
                                  columns=['s_xx', 's_yy', 's_zz',
                                           's_xy', 's_xz', 's_yz'])
        # Add the invariant
        if invariant:
            stress['s_II'] = invariant_info

        # Merge:
        self.output = self.output.merge(stress, left_index=True, right_index=True)

    ##################################################
    #              ADDITIONAL FUNCTIONS              #
    ##################################################  
    #
    #    def get_single_material(self, material_range):
    #        # Background is generally MI = min(MI)
    #        mi = self.output['material'].mat.unique()
    #        mi.sort()
    #
    #        # Drop the rows where this thing is found:
    #        mat_index = self.output['material'][self.output['material'].mat <\
    #                               mi[mi > bg_phase+1][0]].index
    #
    #        return material_list_df
    def reinterpolate_window(self, variable, hdir='x', vdir='y', n_elements=int(1e3)):
        """
        Reinterpolate the dataframes into 2D numpy arrays for easier plotting
        
        returns: 
            hmesh, vmesh, var
        
        """
        # if variable == 'all':
        # variable = dict.fromkeys(self.output.keys())

        # if type(variable) == list:
        # variable = dict.fromkeys(variable)

        from scipy.interpolate import griddata

        # Get window extent:
        hmin = self.output[hdir].min()
        hmax = self.output[hdir].max()
        vmin = self.output[vdir].min()
        vmax = self.output[vdir].max()

        # Create the original coordinates array:
        coord = np.column_stack((self.output[hdir].to_numpy(), self.output[vdir].to_numpy()))

        # Create the coordinate arrays:
        h_n = np.linspace(hmin, hmax, n_elements)
        v_n = np.linspace(vmin, vmax, n_elements)

        # Create the meshes
        H, V = np.meshgrid(h_n, v_n)

        # Interpolate the variable:
        # if type(variable) == str:
        interpolated = griddata(coord, variable, (H, V), method='linear')
        # else:
        # interpolated[i_var] = griddata(coord, i_var, (H, V), method='linear')
        return H, V, interpolated

    def correct_depth(self, vertical_direction='y'):
        """
        Set the correct vertical direction with depth increasing from the surface.
        Defaults to vertical 'y'.

        Parameters:
           > vertical_direction: single character string with the correct direction

        """
        self.output[vertical_direction] = np.abs(
            self.output[vertical_direction] - self.output[vertical_direction].max())

    def set_window(self, hmin, hmax, vmin, vmax, window_dirs=('x', 'y')):
        """
        Function to limit the domain to a rectangular window.
        Arguments:
            > Window domain: [hmin, hmax, vmin, vmax]

         Returns:
            Recreated output dictionary with limited mesh size to the specified domain

         Example use:
            model = uw_model('path/to/model')
            model.set_current_ts(time)
            model.get_material()
            model.create_window([0, 100, 0, 1000])

        """
        # TODO: Add polygon support and 3D
        # if self.dim != 2:
        #     raise TypeError('Currently not working for 3D!')

        # Find the specific mesh domain for this window:
        if window_dirs is None:
            window_dirs = ['x', 'y']
        window_bool = np.all(
            [self.output[window_dirs[0]] > hmin,
             self.output[window_dirs[0]] < hmax,
             self.output[window_dirs[1]] > vmin,
             self.output[window_dirs[1]] < vmax],
            axis=0
        )
        window_index = self.output[window_bool].index

        # Recreate the output dictionary:
        for key in self.output:
            self.output[key] = self.output[key].iloc[window_index].reset_index(drop=True)

        # Remove the nans in the dataset:
        self.output = self.output.dropna()

    def limit_by_index(self, index):

        # Recreate the output dictionary:
        for key in self.output:
            self.output[key] = self.output[key].iloc[index].reset_index(drop=True)

    def remove_background(self, bg_phase=1):
        # Background is generally MI = min(MI)
        mi = self.output.mat.unique()
        mi.sort()

        # Drop the rows where this thing is found:
        bg_index = self.output[self.output.mat < mi[mi > bg_phase + 1][0]].index

        # Recreate the output dictionary:
        for key in self.output:
            self.output[key] = self.output[key].drop(bg_index).reset_index(drop=True)

    def extract_by_material(self, mat_index):

        if type(mat_index) == int or type(mat_index) == float:
            # Extract only one material index:
            mat_index = float(mat_index)

            # Limit the output:
            mesh_sorted = self.output.loc[self.output.mat == mat_index]

            # If empty:
            if mesh_sorted.shape[0] == 0:
                raise Exception('Invalid material submitted, check material DB to see if it exists.')

            # Recreate the output files, resetting the index:
            for key in self.output:
                self.output[key] = self.output[key].iloc[mesh_sorted.index]. \
                    reset_index(drop=True)

        elif type(mat_index) == list or type(mat_index) == np.ndarray:

            # Extract by subset
            temp_bool = [self.output.mat.values == float(x) for x in mat_index]

            # Find correct locations:
            temp_bool = sum(temp_bool)  # Any area with zero is not important

            # Get the index array
            index_bool = temp_bool > 0

            # Recreate the output dictionary:
            for key in self.output:
                self.output[key] = self.output[key].iloc[index_bool]. \
                    reset_index(drop=True)

    def remove_slices(self):
        '''
        Remove the slices made previously and remake the outputs. 
        
        PROBLEM: Slices limit the ID and recreating adds either zeros or NaNs. This can be problematic.
        TODO: Think about it ?
        '''

        # for some reason self isn't working
        model = self

        # =============================== ATTRIBUTES SEARCHED ==========================
        # get previously used keys:
        key_id = []

        for key in self.output.keys():
            key_id.append(key)

        # ================================= EXTRA FUNCTIONS ==========================
        # Check for any functions that were ran before:
        #         # Best way to do is to check the mesh grid for any extra columns
        #         if self.dim == 3:
        #             base_columns = ['x', 'y', 'z']
        #         else:
        #             base_columns = ['x', 'y']

        #         # Loop over all columns and check if they're not in the base set for mesh:
        #         extra_columns = [column for column in self.output['mesh'] if column not in base_columns]

        #         # Copy the information that was previously calculated:
        #         extra_df = self.output['mesh'][extra].copy()

        # ========================== RECREATE THE DATABASE WITHOUT SLICES ===============
        # Reset the timestep
        self.set_current_ts(int(self.current_step))

        # Recreate the model
        for key in key_id:
            try:
                eval('model.get_' + key + '()')
            except AttributeError as AE:
                print(f'Attribute {key} not attainable.')
                pass

    def set_slice(self, direction, value=0, n_slices=None, find_closest=False, save=False):

        ''' 
        
        TODO
        
        '''
        # This makes unlimited slices of the model. Use with care
        if np.all(direction != 'x' and direction != 'y' and direction != 'z'):
            raise Exception('The slice direction must be: ''x'', ''y'' or ''z''!')

        # ======================= SAVE THE SLICES INTO A DICTIONARY OR OBJECT =======================
        if save:
            # This would be useful to redo any slicing destroyed by a function or other
            self.performed_slices.append({'direction': direction,
                                          'value': value,
                                          'nslices': n_slices,
                                          'find_closest': find_closest},
                                         )

        if not n_slices:

            # If the rounder is disables
            if not find_closest:
                # Limit the mesh:
                mesh_sorted = self.output.loc[self.output[direction] == value]

                # If empty:
                if mesh_sorted.shape[0] == 0:
                    raise Exception('Invalid slice value, check mesh dataframe for possible slice index.')

                # Recreate the output files, resetting the index:
                for key in self.output:
                    self.output[key] = self.output[key].iloc[mesh_sorted.index].reset_index(drop=True)
            else:
                # If the rounder is on:

                # get the deltas
                mesh_delta = self.output[direction].copy() - value

                # get the index for the closest value:
                depth_id = mesh_delta.abs().sort_values().index[0]

                # create the slice IDs
                slice_id = self.output[self.output[direction] == self.output[direction].iloc[depth_id]]

                # recreate the domain
                for key in self.output:
                    self.output[key] = self.output[key].iloc[slice_id.index].reset_index(drop=True)

        if n_slices:
            if n_slices > self.res[direction]:
                raise ValueError('More slices than amount of rows along direction.')

            # Make sure this is an integer
            n_slices = int(n_slices)

            # Get the possible values for the direction:
            possible = self.output[direction].unique()
            direction_values = np.linspace(0, len(possible) - 1, n_slices, dtype=int)

            self.slice_info = pd.DataFrame(
                data={'slice_id': range(n_slices), 'slice_value': possible[direction_values]})

            # Extract by subset
            temp_bool = [self.output[direction].values == float(x) for x in possible[direction_values]]

            # Find correct locations:
            temp_bool = sum(temp_bool)  # Any area with zero is not important

            # Get the index array
            index_bool = temp_bool > 0

            #             Recreate the output dictionary:
            for key in self.output:
                self.output[key] = self.output[key].iloc[index_bool]. \
                    reset_index(drop=True)

            # Extract by subset for slicing IDS
            temp_bool = [self.output[direction].values == float(x) for x in possible[direction_values]]
            temp_ids = np.ones(np.array(temp_bool)[0].shape) * 1e3

            for n_slice, index in zip(range(n_slices), temp_bool):
                temp_ids[index] = int(n_slice)

            # Save the ID
            self.output['slice_id'] = temp_ids

        # Regardless of the process, get rid of the nans:
        self.output = self.output.dropna()

    #################################################
    #             SUBDUCTION FUNCTIONS              #
    #################################################

    def find_trench(self, horizontal_direction='x',
                    vertical_direction='y',
                    surface_value=0,
                    filter=True):  # , horizontal_plane='xz', override_dim_check=False
        """
        Function that returns the surface position of the subduction trench, following the 
        minimum divergence method.
        
        
        TODO: 3D
        
        Returns: 
            2D: Horizontal position of the trench 
            3D: Coordinate array for a line that represents the trench along the horizontal plane
        """
        # Check for dimensions
        # if self.dim == 2:
        # Get the vertical coordinates
        hdir = self.output[horizontal_direction]
        vdir = self.output[vertical_direction]

        # Get a surface slice
        surface_index = vdir[vdir == surface_value].index

        # Get velocity fields
        vx = self.output.vx.iloc[surface_index].to_numpy()
        vy = self.output.vy.iloc[surface_index].to_numpy()

        # Calculate the fields 1st derivative
        dvx = np.gradient(vx)
        dx = np.gradient(hdir[surface_index])

        # Calculate divergence (i.e. scalar change of a vector field)
        div_v = dvx / dx

        if filter:
            div_v = div_v[30:-30]

            # Store the trench id:
            trench_id = div_v == min(div_v)

            trench_id = np.array(trench_id)

            trench_id = np.pad(trench_id, 30, constant_values=0)
            # trench_id = np.argmax(trench_id == 1) + 30

        else:

            # Store the trench id:
            trench_id = div_v == min(div_v)

        return float(hdir[surface_index][trench_id])

        # return trench_id
        # elif self.ndims == 3:

    def get_polarity(self, op_material=4, plate_thickness=100., horizontal_plane='xz', trench_direction='z'):
        """
         Function for finding the overriding plate at a critical depth. This depth is 25% deeper than the expected thickness.

         Parameters
            > uw_object: an object created with the uw_model script, loaded with timestep, mesh and material.
            > op_material: the ID or range of IDS for the overriding plate crust.
            > plate_thickness: self-explanatory, maximum expected thickness for the lithosphere in km
            > horizontal_plane: indicate the horizontal plane directions, by default 'xy'.
                                Options: 'xy', 'yz', 'xz'
            > trench_direction: indicate the along trench direction, by default 'z'.
                                  Options: 'x', 'y', 'z'

         Returns:
            New dataframe under model.polarity.
            model.polarity with two columns: along trench axis positions and polarity state.
            Zero (0) represents normal (i.e. initial polarity) while one (1) represents a reversed state.


         Example use:
            model = uw_model('path/to/model')
            model.set_current_ts(time)
            model.get_material()
            model.get_polarity()

        """
        # Catch a few errors:
        if type(horizontal_plane) != str:
            raise TypeError('Plane must be a string!')

        if len(horizontal_plane) != 2:
            raise ValueError('Plane can only contain two letters!')

        if len(trench_direction) != 1:
            raise ValueError('Trench direction is a single letter!')

        # ====================================== CHECK VALIDITY ======================================

        # Ensure the strings are correctly formatted.
        horizontal_plane = "".join(sorted(horizontal_plane.lower()))  # Correctly sorted and in lower case.
        trench_direction = trench_direction.lower()

        # Check if the plane is valid:
        valid_planes = ['xy', 'yz', 'xz']
        check = np.sum([sorted(horizontal_plane) == sorted(valid) for valid in valid_planes])

        if check == 0:
            raise ValueError('Plane is invalid. Please try a combination of ''x'', ''y'' and ''z''.')

        # Check the plane direction:
        slice_direction = 'xyz'

        for char in horizontal_plane:
            slice_direction = slice_direction.replace(char, '')

        # Check if the direction of the trench is valid:
        valid_direction = ['x', 'y', 'z']
        check = np.sum([trench_direction == valid for valid in valid_direction])

        if check == 0:
            raise ValueError('Trench is invalid. Please try ''x'', ''y'' or ''z''.')

        # ================================ DETECT THE POLARITY ========================================

        # Remove any slices:
        self.remove_slices()

        # Set the critical depth:
        critical_depth = 1.25 * plate_thickness * 1e3

        # Create a slice at that depth:
        self.set_slice(slice_direction, value=self.output.y.max() - critical_depth, find_closest=True)

        # Create a database just for the next operations, saves on memory and code:
        reversed_index = self.output[self.output.mat == op_material].index.to_numpy()

        # Detect along trench direction where it is reversed:
        trench_dir_reverse = self.output[trench_direction].loc[reversed_index].unique()

        # Remove any slices:
        self.remove_slices()

        # Create a zeros array, each zero will represent the normal polarity
        polarity = pd.DataFrame(data=np.array([self.output[trench_direction].to_numpy(),
                                               np.zeros(self.output.x.shape)]).T,
                                columns=(trench_direction, 'state'))

        # Check all locations where trench direction reversed is found:
        _, _, reversed_index = np.intersect1d(trench_dir_reverse,
                                              self.output[trench_direction].to_numpy(),
                                              return_indices=True)

        # This only adds a zero to a single value of that trench_direction value:
        polarity.loc[reversed_index, 'state'] = 1

        # Copy those values for all trench_direction values:
        for td in trench_dir_reverse:
            polarity.state[polarity[trench_direction] == td] = 1

        # Add polarity to the main frame
        self.output = self.output.merge(polarity, left_index=True, right_index=True)

        # Check slices that were made before:
        needed_slices = self.performed_slices.copy()

        # Remake the ones deleted:
        for slices in needed_slices:
            print(f'Making slice: {slices}')
            self.set_slice(**slices)

        # Broadcast the polarity into the output?

    def get_swarm(self, n_particles=5e3, assume_yes=False, correct_depth=False):

        """TODO: WRITE THE DOCUMENTATION"""
        # CHECK if the user is sure of what they're doing

        if not assume_yes:
            while True:
                user_input = input('Reading swarms could potentially take a VERY long time. Do you wish to continue? '
                                   '(Y/N)   ')

                if user_input.lower() == 'y':
                    break
                elif user_input.lower() == 'n':
                    raise UserChoice('User terminated the operation.')

        # Start the output lists:
        density, position, material = [], [], []

        # for each of the cores
        print('Amount of particles per core: {}'.format(int(n_particles)))
        for core in range(1, self.nproc + 1):
            # Load their respective file
            data = h5.File(self.model_dir + "/materialSwarm.{}.{}of{}.h5".format(self.current_step, core, self.nproc),
                           mode='r')

            # Get a "low" amount of random points (around 10k):
            index = np.random.choice(len(data['Position']), int(n_particles))

            # Append to the list:
            density.append(data['DensityLabel'][()][index])
            position.append(data['Position'][()][index])
            material.append(data['MaterialIndex'][()][index])

            # Add a progress bar to this VERY lengthy progress
            printProgressBar(core, self.nproc, prefix='Reading swarm data at timestep {}:'.format(self.current_step),
                             suffix='complete', length=50)

        # Concatenate all the information
        position = np.concatenate(position)
        density = np.concatenate(density)
        material = np.concatenate(density)

        # add these properties to the object
        self.particle_data = pd.DataFrame(position, columns=['x', 'y', 'z'])
        self.particle_data['density'] = density
        self.particle_data['material'] = material

        if correct_depth:
            self.particle_data.y = np.abs(self.particle_data.y - self.particle_data.y.max())

    def swarms_to_nodes(self):
        """TODO: combine all the output DFS into a single one.
           For now this will just merge nodal positions with the swarm data"""
        import scipy.spatial as spatial

        # Get nodal positions:
        if self.dim == 3:
            mesh = self.output[['x', 'y', 'z']].to_numpy()
        else:
            mesh = self.output[['x', 'y']].to_numpy()

        # Initiate the tree:
        self._particle_tree = spatial.cKDTree(self.particle_data[['x', 'y', 'z']])

        # Get the grid spacing (this assumes regular grids) TODO: allow irregular grids
        dx = np.diff(self.output.x.unique())[0]

        # Create a final density list:
        density = np.zeros(self.output.x.shape)

        for point, k in zip(mesh, range(mesh.shape[0])):

            # add a progress bar:
            printProgressBar(k, mesh.shape[0]-1,
                             prefix='Interpolating density data at timestep {}:'.format(self.current_step),
                             suffix='complete', length=50)

            # At each nodal point get the 50 (?) closest particles:
            swarm_index = self._get_neighbour_swarms(point, k=10)

            # At each point, integrate the density of the swarms into the node point
            density[k] = self.particle_data.iloc[swarm_index].density.mean()

            if np.isnan(density[k]):
                break

        # add the density array to the dataframe:
        self.output['density'] = density

    def _get_neighbour_swarms(self, point, k=5):
        """
        Get the closest neighbours to a specified node
        :param radius:
        :param number_of_particles:
        :param point:
        :return:
        """
        # Check the closest
        distance, index = self._particle_tree.query(point, k=k)

        # Clear the "INF" issues on Nearest particles:
        # output = self.particle_data.to_numpy()[index[index != np.inf]]

        return index[distance != np.inf]
