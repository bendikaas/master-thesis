import math

# spatial spatial grid for storing motion vectors (EPSG:32630). the default values for teh bounding box is teh area this work took place
class Grid:
    def __init__(self, cellsize=10, east_min=437200, east_max=440200, north_min=9026200, north_max=9029000):
        self.cellsize = cellsize  # size of each grid cell [m]

        self.north_min, self.north_max = north_min, north_max  # bounding box (northing)
        self.east_min, self.east_max = east_min, east_max  # bounding box (easting)

        self.ni = int((self.east_max - self.east_min) / self.cellsize)  # number of columns
        self.nj = int((self.north_max - self.north_min) / self.cellsize)  # number of rows

        self.raw_cells = [[[] for _ in range(self.ni)] for _ in range(self.nj)]  # motion vectors before averaging
        self.known_cells = []  # (i, j) of all cells with input data

        self.cells = [[None for _ in range(self.ni)] for _ in range(self.nj)]  # final averaged velocity vectors
        self.distances = [[None for _ in range(self.ni)] for _ in range(self.nj)]  # distance to nearest known cell

    def get_grid_index(self, x, y):
        # in:  x, y [m] (world coordinates)
        # out: i, j [grid indices]
        if not (self.east_min <= x <= self.east_max and self.north_min <= y <= self.north_max):
            raise ValueError("Coordinates are out of bounds")

        i = int((x - self.east_min) / self.cellsize)  # column index
        j = int((y - self.north_min) / self.cellsize)  # row index
        return i, j

    def get_velocity_and_distance(self, i, j):
        # in:  i, j [grid indices]
        # out: velocity [tuple or None], distance [float or None]
        if not (0 <= i < self.ni and 0 <= j < self.nj):
            raise ValueError("Grid index is out of bounds")

        velocity = self.cells[j][i]  # final velocity vector
        distance = self.distances[j][i]  # distance to nearest filled cell
        return velocity, distance

    def add_velocity_vector(self, pos, velocity_vector):
        # in:  pos (x, y) [m], velocity_vector (vx, vy) [m/s]
        x, y = pos
        if not (self.east_min <= x < self.east_max and self.north_min <= y < self.north_max):
            raise ValueError("Grid index is out of bounds")

        i, j = self.get_grid_index(x, y)
        self.raw_cells[j][i].append(velocity_vector)  # add vector to raw cell
        self.known_cells.append((i, j))  # mark cell as known

    def average_raw_cells(self):
        for j in range(self.nj):
            for i in range(self.ni):
                if self.raw_cells[j][i]:
                    vectors = self.raw_cells[j][i]
                    # average each component (vx, vy)
                    avg_vector = tuple(sum(v) / len(v) for v in zip(*vectors))
                    self.raw_cells[j][i] = avg_vector
                else:
                    self.raw_cells[j][i] = None  # mark empty cells

    def fill_cells(self):
        for j in range(self.nj):
            for i in range(self.ni):
                if self.raw_cells[j][i] is not None:
                    # for already known cells, just copy the value directly
                    self.cells[j][i] = self.raw_cells[j][i]
                    self.distances[j][i] = 0
                else:
                    # fill missing cell by nearest known neighbor(s)
                    min_distance = float('inf')
                    nearest_vectors = []

                    for filled_i, filled_j in self.known_cells:
                        # distance to this known cell
                        di = (i - filled_i) * self.cellsize
                        dj = (j - filled_j) * self.cellsize
                        distance = math.sqrt(di ** 2 + dj ** 2)

                        if distance < min_distance:
                            min_distance = distance
                            nearest_vectors = [self.raw_cells[filled_j][filled_i]]  # reset
                        elif distance == min_distance:
                            nearest_vectors.append(self.raw_cells[filled_j][filled_i])  # add equal distance

                    if len(nearest_vectors) == 1:
                        self.cells[j][i] = nearest_vectors[0]
                        self.distances[j][i] = min_distance
                    else:
                        # average the vectors if multiple equidistant neighbors
                        vx_avg = sum(v[0] for v in nearest_vectors) / len(nearest_vectors)
                        vy_avg = sum(v[1] for v in nearest_vectors) / len(nearest_vectors)
                        self.cells[j][i] = (vx_avg, vy_avg)
                        self.distances[j][i] = min_distance

    def finalise(self):
        # computes the final vector field and distance map
        self.average_raw_cells()
        self.fill_cells()
