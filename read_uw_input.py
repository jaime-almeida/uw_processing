import ast
import re as re

import matplotlib.pyplot as plt
import numpy as np
import shapely
import shapely.ops as so
from bs4 import BeautifulSoup
from descartes import PolygonPatch
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from shapely.geometry import Point
from shapely.geometry import Polygon as shpPolygon

plt.ioff()


def cascading_intersection(polygon_list):
    """
    Performs cascading intersection, applying the comutative property of intersection:
    AnBnC = (AnB)nC
    this extends towards infinity

    :param polygon_list:
    :return:
    """
    intersect_results = polygon_list[0]

    # Make the intersections:
    for i in np.arange(1, len(polygon_list)):
        # Make the first intersections
        intersect_results = intersect_results.intersection(polygon_list[i])

    return intersect_results


def identify_shape_type(struct):
    # Get the struct name:
    try:
        name = struct.attrs['name']
    # if it doesn't register
    except KeyError:
        return
    # Find the boolean structs:
    if 'Intersection' in struct.text.split('\n')[1] or 'Union' in struct.text.split('\n')[1]:
        # Ignore the particle layouts for ppc control
        if 'Particle' not in struct.text.split('\n')[1]:
            return 'bool'

    # Recognize if it is a possible shape:
    if 'Box' in struct.text.split('\n')[1] or 'Polygon' in struct.text.split('\n')[1]:
        # Check if it is a thermal shape:
        if re.findall('T', name) and 'mask' not in name:
            return 'thermal'
        else:
            return 'material'

    if 'circle' in name:
        return 'circle'


def read_box_struct(struct):
    # Get the information:
    information = struct.text.split('\n')

    # Coordinates:
    box_xmin = float(information[2])
    box_xmax = float(information[3])
    box_ymin = float(information[4])
    box_ymax = float(information[5])

    return box_xmin, box_xmax, box_ymin, box_ymax


def read_polygon_shape(struct):
    # Get the vertices data:
    vertices_data = struct.contents[3].contents[1].contents[-1]

    # Split into lines!!
    vertices_data = vertices_data.split('\n')[1:-1]

    # In each entry, separate the two values in a list:
    for i in range(len(vertices_data)):
        # Convert to str, strip the formatting and add commas:
        vertices_data[i] = str(vertices_data[i]).strip()

        if '\t' in vertices_data[i]:
            # Check for the stupid \t that sometimes show up
            vertices_data[i] = vertices_data[i].split('\t')
            vertices_data[i] = '[{}, {}]'.format(vertices_data[i][0], vertices_data[i][-1])

        else:
            vertices_data[i] = vertices_data[i].replace(' ', ',')

        if vertices_data[i] == '':
            continue

        # Convert to list:
        vertices_data[i] = ast.literal_eval(vertices_data[i])

    # Deal with the extra '' that sometimes shows up
    while '' in vertices_data:
        try:
            vertices_data.remove('')
        except ValueError:
            pass

    vertices_data = np.array(vertices_data, dtype=float)

    return vertices_data


def mpl_to_shpl(iter_shape):
    if type(iter_shape) == plt.Rectangle:
        shape_path = iter_shape.get_verts()
        shape_path = shpPolygon(shape_path)

    elif type(iter_shape) == plt.Polygon:
        shape_path = iter_shape.get_xy()
        shape_path = shpPolygon(shape_path)

    elif type(iter_shape) == plt.Circle:
        center = iter_shape.get_center()
        radius = iter_shape.get_radius()

        # Convert to a shapely path:
        shape_path = Point(center).buffer(radius)
    else:
        # If it's an already processed shape
        shape_path = iter_shape

    return shape_path


class UwShapePlotter:

    def __init__(self, filename, model_dimensions, ax=None, figsize=None):
        # XML file to be parsed:
        self.filename = filename

        # Amazing parser to unwrap all this shit:
        with open(self.filename) as f:
            soup = BeautifulSoup(f, 'xml')

            # Find all Structs! this avoids plenty of unneeded shapes
            self.results = soup.find_all('struct')

        # patch list
        self.thermal_patch_list = []
        self.thermal_patch_name = []
        self.material_patch_list = []
        self.material_patch_name = []
        self._bool_list = []
        self._bool_shape_name = []

        # Create the possible everywhere shape:
        model_xmin, model_xmax, model_ymin, model_ymax = model_dimensions
        everywhere_shape = plt.Rectangle((model_xmin, model_ymin),
                                         width=model_xmin + (model_xmax - model_xmin),
                                         height=model_ymin + (model_ymax - model_ymin),
                                         fill=False,
                                         edgecolor=None
                                         )
        self.material_patch_list.append(everywhere_shape)
        self.material_patch_name.append('everywhere')
        self.everywhere_shpl = mpl_to_shpl(everywhere_shape)

        # Create a figure or import from the user
        if ax:
            self.fig, self.ax = plt.gcf(), plt.gca()
        else:
            self.fig, self.ax = plt.subplots(figsize=figsize, nrows=2)

        # GEt the material shapes:
        self._get_material_shapes()
        self._get_boolean_configuration()

        # Get the thermal shapes
        self._allocated_thermal_shapes = self._check_thermal_allocation()
        self.get_thermal_shapes()

        # Apply the boolean conditions
        self._apply_boolean_configurations()

        # Add the stupid everywhere properly
        for counter, iter_shape in enumerate(self.material_patch_list):
            self.material_patch_list[counter] = mpl_to_shpl(iter_shape)

        # clean the extra shapes
        valid_shapes = self._evaluate_shape_validity()

        # Plot material shapes
        self._proxy_artists = []
        self.plot_material_shapes(valid_shapes)
        self.ax[1].legend(self._proxy_artists, [item.get_label() for item in self._proxy_artists], loc='best')

        # Plot the thermal shapes
        for p in self.thermal_patch_list:
            self.ax[0].add_artist(p)
        self.ax[0].axis(model_dimensions)
        self.ax[1].axis(model_dimensions)
        self.ax[0].set_title('Model')
        self.fig.show()
        # self.ax.axis(model_dimensions)
        # self.ax.set_title('MaterialShapes')
        # self.fig.show()

    def _evaluate_shape_validity(self):
        """
        Check if the shape is allocated to a material rheology structure
        """
        valid_shapes = []
        for struct in self.results:
            if struct.text.split('\n')[1] == 'RheologyMaterial':
                # get the shape that is allocated to a rheology material
                valid_shapes.append(struct.text.split('\n')[2].strip())

        # Remember the order:
        valid_shape_order = list(range(len(valid_shapes)))

        # Create a material dict with only the valid shapes, keeping the order:
        material_list = []
        for shape_name in valid_shapes:
            material_list.append([shape_name, self.material_patch_list[self.material_patch_name.index(shape_name)]])

        return material_list

    def _get_boolean_configuration(self):
        # Read the struct type
        for struct in self.results:

            struct_type = identify_shape_type(struct)

            if struct_type == 'bool':
                # Get the name:
                name = struct.attrs['name']

                # Text contained in the structure
                struct_text = struct.text.split('\n')

                # Type of boolean applied
                bool_type = struct_text[1]
                needed_shapes = struct_text[3:-2]

                # SAve the data into the lists
                self._bool_list.append([bool_type, needed_shapes])
                self._bool_shape_name.append(struct.attrs['name'])

    def _apply_boolean_configurations(self):
        for boolean_struct, boolean_name in zip(self._bool_list, self._bool_shape_name):
            # Get the list of possible shapes (both thermal and non thermal), which changes at every iteration:
            list_of_shapes = self.material_patch_list + self.thermal_patch_list
            names_of_shapes = self.material_patch_name + self.thermal_patch_name

            # For each found boolean structure, identify the type of boolean and the shapes:
            boolean_type = boolean_struct[0]
            boolean_shape_names = boolean_struct[1]
            boolean_shapes = []
            negation_needed = []

            # Get the needed shapes:
            for shape_name in boolean_shape_names:
                # if it is a negated shape:
                if shape_name[0] == '!':
                    boolean_shapes.append(list_of_shapes[names_of_shapes.index(shape_name[1:])])
                    negation_needed.append(True)
                else:
                    boolean_shapes.append(list_of_shapes[names_of_shapes.index(shape_name)])
                    negation_needed.append(False)

            # Apply the needed boolean:
            if boolean_type == 'Intersection':
                shpl_polygons = []
                for iter_shape, negation, name in zip(boolean_shapes, negation_needed, boolean_shape_names):
                    # Convert to shapely
                    converted_shape = mpl_to_shpl(iter_shape)

                    # Apply the negation if needed, negation being the difference operation between everywhere and
                    # the shape
                    if negation:
                        negated_shape = self.everywhere_shpl.difference(converted_shape)
                        shpl_polygons.append(negated_shape)
                    else:
                        # Append to the list
                        shpl_polygons.append(converted_shape)

                # Perform the intersection
                intersected_shape = cascading_intersection(shpl_polygons)
                if intersected_shape.is_empty:
                    raise Exception('The {} shape produces an empty output.'.format(boolean_name))

                self.material_patch_name.append(boolean_name)
                self.material_patch_list.append(intersected_shape)

            if boolean_type == 'Union':
                shpl_polygons = []
                for iter_shape in boolean_shapes:
                    # Convert to shapely
                    converted_shape = mpl_to_shpl(iter_shape)

                    # Append to the list
                    shpl_polygons.append(converted_shape)

                # Create a new shape with the correct name:
                united_shape = so.cascaded_union(shpl_polygons)

                # Append this new shape to the already existing shapes:
                self.material_patch_list.append(united_shape)
                self.material_patch_name.append(boolean_name)

    def _get_material_shapes(self, c='k'):
        """
        Need to get all material shapes in a dictionary that contains their definition.
        Definitions can be:
        a) Box
        b) Polygon
        c) Circle
        """
        for struct in self.results:
            # Get the struct name and type:
            struct_type = identify_shape_type(struct)

            if struct_type == 'material' or struct_type == 'mask':
                # if the shape is not a thermal shape
                # get the name
                name = struct.attrs['name']
                # Line containing the shape
                shape_type = struct.contents[1]

                # verify the type:
                if 'Box' in shape_type:
                    # Get the box information:
                    box_xmin, box_xmax, box_ymin, box_ymax = read_box_struct(struct)

                    # Create the rectangle artist:
                    shape_box = plt.Rectangle((box_xmin, box_ymin), width=box_xmax - box_xmin,
                                              height=box_ymax - box_ymin,
                                              fill='w',
                                              edgecolor=c)

                    # append to the patch list:
                    if struct_type == 'material':
                        self.material_patch_list.append(shape_box)
                        self.material_patch_name.append(name)

                # If it is a polygon:
                if 'PolygonShape' in shape_type:
                    # Get the vertices data:
                    vertices_data = read_polygon_shape(struct)

                    # Create polygon shape:
                    shape_polygon = Polygon(vertices_data, fill='w', closed=True, edgecolor=c)

                    # Save in the object
                    self.material_patch_list.append(shape_polygon)
                    self.material_patch_name.append(name)

                # Circles are different shapes, which are plain named circlex:
            if struct_type == 'circle':
                # Get the name
                name = struct.attrs['name']

                # iterate of the structs:
                information = [data.strip() for data in struct.text.split('\n')]

                # Circle info:
                radius = float(information[4])
                centre_x = float(information[2])
                centre_y = float(information[3])

                # Create the circle shape
                circle_shape = plt.Circle(xy=(centre_x, centre_y),
                                          radius=radius,
                                          fill='w',
                                          edgecolor=c)

                # Save in the object
                self.material_patch_list.append(circle_shape)
                self.material_patch_name.append(name)

    def plot_material_shapes(self, material_list):
        # Generate the colormap
        color = iter(plt.cm.RdYlBu_r(np.linspace(0, 1, len(material_list))))

        for key, shape in material_list:

            # Get the next color
            c = next(color)

            # for each material shape, draw it in:
            if type(shape) == shapely.geometry.collection.GeometryCollection:
                continue

            current_shape = PolygonPatch(shape, fc=c, ec='k')

            # add the patch
            self.ax[1].add_patch(current_shape)

            # Add a proxy artist:
            temp_artist = Line2D([-1e5], [-1e5], linestyle='none', marker='s', markersize=10, markerfacecolor=c,
                                 markeredgecolor='k', label=key)
            self.fig.show()
            self._proxy_artists.append(temp_artist)

    def get_thermal_shapes(self):

        color = iter(plt.cm.hsv(np.linspace(0, 1, 20)))

        for struct in self.results:
            # Get the struct name:
            struct_type = identify_shape_type(struct)

            if struct_type == 'thermal':
                # If we're in a new shape, change the colour
                c = next(color)

                # Get the name:
                name = struct.attrs['name']

                # Line containing the shape
                shape_type = struct.contents[1]

                # If the output is a polygon:
                if 'poly' in str(shape_type).lower():
                    # Get the shape vertices data:
                    vertices_data = read_polygon_shape(struct)

                    # Convert to a Polygon
                    shape_polygon = Polygon(vertices_data, fill=False, closed=True, edgecolor=c, lw=2, linestyle='--')

                    # Save in the object
                    self.thermal_patch_list.append(shape_polygon)
                    self.thermal_patch_name.append(name)

                    # Identify the structure in the plot:
                    ann_point = np.array(
                        [vertices_data[:, 0].min() + (vertices_data[:, 0].max() - vertices_data[:, 0].min()) / 2,
                         vertices_data[:, 1].min() + (vertices_data[:, 1].max() - vertices_data[:, 1].min()) / 2]
                    )

                    self.ax[0].text(x=ann_point[0],
                                    y=ann_point[1],
                                    s='(T{})'.format(len(self.thermal_patch_list)),
                                    c=c
                                    )

                # If the output is a box:
                if 'box' in str(shape_type).lower():
                    # Get the box information:
                    box_xmin, box_xmax, box_ymin, box_ymax = read_box_struct(struct)

                    # Create the rectangle artist:
                    shape_box = plt.Rectangle((box_xmin, box_ymin), width=box_xmax - box_xmin,
                                              height=box_ymax - box_ymin,
                                              fill=False,
                                              edgecolor=c,
                                              lw=2,
                                              linestyle='--')

                    # append to the patch list:
                    self.thermal_patch_list.append(shape_box)
                    self.thermal_patch_name.append(name)

                    # Annotate it:
                    ann_point = [box_xmin + (box_xmax - box_xmin) / 2,
                                 box_ymin + (box_ymax - box_ymin) / 2]

                    self.ax[0].text(x=ann_point[0],
                                    y=ann_point[1],
                                    s='(T{})'.format(len(self.thermal_patch_list)),
                                    c=c)

                if name in self._allocated_thermal_shapes:
                    print("T{}: {} <--- Allocated".format(len(self.thermal_patch_list), name))
                else:
                    print("T{}: {}".format(len(self.thermal_patch_list), name))

    def _check_thermal_allocation(self):
        thermal_file = self.filename.split('.')[0] + '_T.' + self.filename.split('.')[1]

        with open(thermal_file, 'r') as f:
            soup = BeautifulSoup(f, 'xml')

        # Find all linearThermalShapes
        linear_ic = soup.find_all('list', {'name': 'linearShapeIC'})

        # Find the shapes, first word after the param name Shape definition (ignoring the first entry,
        # which is list):
        allocated_shapes = str(linear_ic).split('<param name="Shape">')[1:]

        # Clean the formatting
        allocated_shapes = [shape.strip().split(' ')[0] for shape in allocated_shapes][1:]

        # GEt the boundary shapes
        thermal_bcs = soup.find_all('list', {'name': 'vcList'})

        # Find the shapes, first word after the param name Shape definition (ignoring the first entry,
        # which is list):
        temp = str(thermal_bcs).split('<param name="Shape">')

        # Clean the formatting
        temp = [shape.strip().split(' ')[0] for shape in temp][1:]

        allocated_shapes += temp

        return allocated_shapes


if __name__ == '__main__':
    model_dims = [-700, 4300, 340, 1020]

    shape = UwShapePlotter(filename=r'F:/NoPlateauTest/InputFiles/FSurface/main.xml', model_dimensions=model_dims,
                           figsize=[17.99, 8.60])

    # plt.close('all')x
