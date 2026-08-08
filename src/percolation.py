import numpy as np


def generate_grid(L: int, p: float) -> np.ndarray:
    grid = (np.random.rand(L, L) < p).astype(int)
    return grid

class FindUnion:
    def __init__(self):
        self.parent = [0, 1]  
        self.rank = [0, 0]    

    def make_set(self) -> int:
        new_label = len(self.parent)
        self.parent.append(new_label)
        self.rank.append(0)
        return new_label

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x]) #сжатие пути
        return self.parent[x]

    def union(self, x: int, y: int) -> int:
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return root_x
        #объединение по рангу
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
            return root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
            return root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
            return root_x                         #класс готов, скорость ~О(1) на операцию

class HoshenKopelman:
    def __init__(self, grid: np.ndarray):
        self.grid = grid.copy()
        self.dsu = FindUnion()

    def first_pass(self) -> None:
        rows, cols = self.grid.shape
        for r in range(rows):
            for c in range(cols):
                if self.grid[r, c] == 1:
                    left = self.grid[r, c - 1] if c > 0 else 0
                    top = self.grid[r - 1, c] if r > 0 else 0

                    if left == 0 and top == 0:
                        self.grid[r, c] = self.dsu.make_set()

                    elif left > 0 and top == 0:
                        self.grid[r, c] = left
                    elif left == 0 and top > 0: 
                        self.grid[r, c] = top
                    else: 
                        self.grid[r, c] = self.dsu.union(left, top)

    def second_pass(self) -> None:
        rows, cols = self.grid.shape
        for r in range(rows):
            for c in range(cols):
                if self.grid[r,c] > 0:
                    self.grid[r,c] = self.dsu.find(self.grid[r,c])

    def run(self) -> np.ndarray:
        self.first_pass()
        self.second_pass()
        return self.grid

    def check_percolation(self) -> tuple[bool, int | None]:
        top_row = set(self.grid[0, :][self.grid[0, :] > 0])
        bottom_row = set(self.grid[-1, :][self.grid[-1, :] > 0])
        common_clusters = top_row.intersection(bottom_row)

        if common_clusters:        
            percolating_id = list(common_clusters)[0]
            return True, percolating_id

        return False, None


    