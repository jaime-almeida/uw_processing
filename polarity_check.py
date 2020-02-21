

def polarity_check(uw_obj, op_material=4, plate_thickness=100., horizontal_plane='xz', trench_direction='z'):
    '''
     Function for finding the overriding plate at a critical depth. This depth is 15% larger than the expected thickness.
     
     Parameters:
        >>> uw_object: an object created with the uw_model script, loaded with timestep, mesh and material.
        >>> op_material: the ID or range of IDS for the overriding plate crust. 
        >>> plate_thickness: self-explanatory, maximum expected thickness for the lithosphere in km
        >>> horizontal_plane: indicate the horizontal plane directions, by default 'xy'.
                              Options: 'xy', 'yz', 'xz'
        >>> trench_direction: indicate the along trench direction, by default 'z'.
                              Options: 'x', 'y', 'z'                      
    
     Returns: 
        Coordinates (purely along the trench direction) of the reversed polarities and their indexes
    '''
    # Catch a few errors:
    if type(horizontal_plane) != str:
        raise TypeError('Plane must be a string!')
        
    if len(horizontal_plane) != 2:
        raise ValueError('Plane can only contain two letters!')
    
    if len(trench_direction) != 1:
        raise ValueError('Trench direction is a single letter!')
        
    # ====================================== CHECK VALIDITY ======================================
    
    # Ensure the strings are correctly formatted.
    horizontal_plane = "".join(sorted(horizontal_plane.lower())) # Correctly sorted and in lower case.
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
    # Set the critical depth:
    critical_depth = 2*plate_thickness*1e3
    
    # Create a slice at that depth:
    uw_obj.set_slice(slice_direction, value=uw_obj.output['mesh'].y.max() - critical_depth, find_closest = True)
    
    # Create a database just for the next operations, saves on memory and code:
    temp = uw_obj.output['mesh'].copy()
    temp['m'] = uw_obj.output['material'].mat.copy()
    
    # Return the ids and z coordinates of the crust subduction:
    return temp[temp.m == op_material].z.to_numpy(), temp[temp.m == op_material].index.to_numpy()
    
