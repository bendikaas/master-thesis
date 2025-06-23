import math

# coordinates are in "EPSG:32630"
class Grid:
    def __init__(self, cellsize = 10, east_min = 438200, east_max = 439200, north_min = 9026200, north_max = 9027600):
        # Grid cell size in meters
        self.cellsize = cellsize # meters

        # Bounding box (EPSG:6052)
        self.north_min, self.north_max = north_min, north_max
        self.east_min, self.east_max = east_min, east_max

        # Grid dimensions
        self.ni = int((self.east_max - self.east_min) / self.cellsize)
        self.nj = int((self.north_max - self.north_min) / self.cellsize)

        # raw cells are lists that store all the motion vectors that falls in that cell
        self.raw_cells = [[[] for _ in range(self.ni)] for _ in range(self.nj)]

        # list to keep track of the filled cells, where each element is a tuple of a known cell (i, j)
        self.known_cells = []

        # Finalized cells with averaged motion vectors
        self.cells = [[None for _ in range(self.ni)] for _ in range(self.nj)]
        self.distances = [[None for _ in range(self.ni)] for _ in range(self.nj)]

    # given a world coord (x, y) in EPSG:6052, return the corresponding grid index (i, j)
    def get_grid_index(self, x, y):
        if not (self.east_min <= x <= self.east_max and self.north_min <= y <= self.north_max):
            raise ValueError("Coordinates are out of bounds")

        i = int((x - self.east_min) / self.cellsize)
        j = int((y - self.north_min) / self.cellsize)

        return i, j
    
    # returns the motion and distance for a given cell (i, j)
    def get_velocity_and_distance(self, i, j):
        if not (0 <= i < self.ni and 0 <= j < self.nj):
            raise ValueError("Grid index is out of bounds")

        velocity = self.cells[j][i]
        distance = self.distances[j][i]

        return velocity, distance
    
    # adds a motion vector (dx, dy) in meters to the raw cell at pos (x, y) in world coord. and updates the known cells list
    def add_velocity_vector(self, pos, velocity_vector):
        x, y = pos

        if not (self.east_min <= x < self.east_max and self.north_min <= y < self.north_max):
            raise ValueError("Grid index is out of bounds")
        
        # find the raw cell index (i, j) that corresponds to (x, y)
        i, j = self.get_grid_index(x, y)
        self.raw_cells[j][i].append(velocity_vector)
        self.known_cells.append((i, j))

    # if there are more than one motion vector in a cell, average them and set the cell to the average motion vector
    def average_raw_cells(self):
        for j in range(self.nj):
            for i in range(self.ni):
                if self.raw_cells[j][i]:
                    # Average the motion vectors in the raw cells
                    avg_vector = tuple(sum(v) / len(v) for v in zip(*self.raw_cells[j][i]))
                    self.raw_cells[j][i] = avg_vector
                else:
                    self.raw_cells[j][i] = None # If the cell is empty, set it to None

    # fill the finalized cells with the average motion vectors and get the distance to the nearest filled cell
    def fill_cells(self):
        for j in range(self.nj):
            for i in range(self.ni):
                if self.raw_cells[j][i] is not None: # its a known cell
                    self.cells[j][i] = self.raw_cells[j][i]
                    self.distances[j][i] = 0
                
                else:
                    # If the cell is None, fill it with the nearest filled cell's motion vector
                    min_distance = float('inf')
                    nearest_vectors = []

                    for filled_i, filled_j in self.known_cells:
                        di = (i - filled_i) * self.cellsize
                        dj = (j - filled_j) * self.cellsize
                        distance = math.sqrt(di ** 2 + dj ** 2)

                        if distance < min_distance:
                            # Found a closer one → reset list
                            min_distance = distance
                            nearest_vectors = [self.raw_cells[filled_j][filled_i]]
                        elif distance == min_distance:
                            # Same distance → add it
                            nearest_vectors.append(self.raw_cells[filled_j][filled_i])

                    if len(nearest_vectors) == 1:
                        self.cells[j][i] = nearest_vectors[0]
                        self.distances[j][i] = min_distance
                    else:
                        vx_avg = sum(v[0] for v in nearest_vectors) / len(nearest_vectors)
                        vy_avg = sum(v[1] for v in nearest_vectors) / len(nearest_vectors)
                        self.cells[j][i] = (vx_avg, vy_avg)
                        self.distances[j][i] = min_distance
    
    def finalise(self):
        self.average_raw_cells()
        self.fill_cells()
