from bs4 import BeautifulSoup
import ast
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

import re as re


class ThermalShapePlotter:

    def __init__(self, filename, model_dimensions, ax=None, figsize=None):
        # XML file to be parsed:
        self.filename = filename

        # Amazing parser to unwrap all this shit:
        with open(self.filename) as f:
            soup = BeautifulSoup(f, 'xml')

            # Find all Structs! this avoids plenty of unneeded shapes
            self.results = soup.find_all('struct')

            # patch list
            self._patch_list = []

            # Create a figure or import from the user
            if ax:
                self.fig, self.ax = plt.gcf(), plt.gca()
            else:
                self.fig, self.ax = plt.subplots(figsize=figsize)

            self.plot_thermal_boxes()
            self.plot_thermal_polygons()

            for p in self._patch_list:
                self.ax.add_artist(p)
            self.ax.axis(model_dimensions)
            self.fig.show()

    def plot_thermal_boxes(self):
        for struct in self.results:
            # Get the struct name:
            try:
                name = struct.attrs['name']
            # if it doesn't register
            except KeyError:
                continue

            # Check if it is a shape:
            if 'shape' not in name:
                continue

            # Check if it is a thermal shape:
            if not re.findall('T', name):
                continue

            # Line containing the shape
            shape_type = struct.contents[1]

            # If the output is a box:
            if 'box' in str(shape_type).lower():
                # Get the information:
                information = struct.text.split('\n')

                # Coordinates:
                box_xmin = float(information[2])
                box_xmax = float(information[3])
                box_ymin = float(information[4])
                box_ymax = float(information[5])

                # Create the rectangle artist:
                shape_box = plt.Rectangle((box_xmin, box_ymin), width=box_xmax - box_xmin, height=box_ymax - box_ymin,
                                          fill=False)

                # append to the patch list:
                self._patch_list.append(shape_box)

                # Annotate it:
                ann_point = [box_xmin + (box_xmax - box_xmin) / 2,
                             box_ymin + (box_ymax - box_ymin) / 2]

                # generate a text position, to avoid the overlapping
                mover = np.array([1, np.random.choice(np.arange(1, 6, 1), 1)])

                self.ax.annotate(name, xy=ann_point,
                                 xycoords='data',
                                 xytext=ann_point + np.array([50, -100]) * mover,
                                 textcoords='data',
                                 arrowprops=dict(arrowstyle="->",
                                                 connectionstyle="arc3")
                                 )

    def plot_thermal_polygons(self):
        for struct in self.results:
            # Get the struct name:
            try:
                name = struct.attrs['name']
            # if it doesn't register
            except KeyError:
                continue

            # Check if it is a shape:
            if 'shape' not in name:
                continue

            # Check if it is a thermal shape:
            if not re.findall('T', name):
                continue

            # Line containing the shape
            shape_type = struct.contents[1]

            # If the output is a polygon:
            if 'poly' in str(shape_type).lower():
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

                    # Convert to list:
                    vertices_data[i] = ast.literal_eval(vertices_data[i])

                # Convert to array:
                vertices_data = np.array(vertices_data, dtype=float)

                # Convert to a Polygon
                shape_polygon = Polygon(vertices_data, fill=False, closed=True, edgecolor='red')
                self._patch_list.append(shape_polygon)

                # Annotate it:
                ann_point = vertices_data[3]

                # generate a text position, to avoid the overlapping
                mover = np.zeros((2, 1))
                while mover.sum() == 0 and np.sign(mover.prod()) >= 0:
                    mover = np.random.choice(np.arange(-3, 3, 1), 2)

                self.ax.annotate(name, xy=ann_point,
                                 xycoords='data',
                                 xytext=ann_point + np.array([100, 100]) * mover,
                                 textcoords='data',
                                 arrowprops=dict(arrowstyle="->",
                                                 connectionstyle="arc3")
                                 )


if __name__ == '__main__':
    model_dims = [-700, 4300, 0, 1020]

    shape = ThermalShapePlotter(filename='main.xml',
                                model_dimensions=[-700, 4300, 0, 1020],
                                figsize=[17.99, 5.15])
