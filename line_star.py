from astar import AStar
import math


class LineStar(AStar):
    def __init__(self, img):
        super().__init__()
        self.img = img
        self.img_height, self.img_width = img.shape

    def get_path(self, start, end):
        result = self.astar(start, end)
        if result is None:
            # Poke holes
            print("Couldn't find path")
            return None
        return list(result)

    def check_pos(self, node):
        x, y = node
        if x < 0 or y < 0 or x >= self.img_width or y >= self.img_height:
            return False
        value = self.img[y, x]
        return value > 128

    def neighbors(self, node):
        x, y = node
        neighbours = [
            (x + 1, y + 1),  # top right
            (x + 1, y + 0),  # right
            (x + 1, y - 1),  # bottom right
            (x - 1, y + 1),  # top left
            (x - 1, y + 0),  # left
            (x - 1, y - 1),  # bottom left
            (x + 0, y + 1),  # top
            (x + 0, y - 1),  # bottom
        ]

        neighbours = filter(self.check_pos, neighbours)
        return neighbours

    def distance_between(self, n1, n2):
        x1, y1 = n1
        x2, y2 = n2
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def heuristic_cost_estimate(self, current, goal):
        return abs(current[0] - goal[0]) + abs(current[1] - goal[1])
