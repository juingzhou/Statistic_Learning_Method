class kd_node:
    def __init__(self, data=None, depth=0, left=None, right=None):
        self.data = data
        self.depth = depth
        self.left = left
        self.right = right

    def travel(self):
        if not self.data:
            return
        print(self.data, self.depth)
        self.left.travel()
        self.right.travel()
    
    def build_tree(self, points, depth):
        if not points:
            return
        dims = len(points[0])
        axis = depth % dims
        points.sort(key = lambda x:x[axis])
        median_index = len(points) // 2
        self.data = points[median_index]
        self.left = kd_node(None, depth + 1)
        self.right = kd_node(None, depth + 1)
        self.left.build_tree(points[: median_index], depth+1)
        self.right.build_tree(points[median_index+1: ], depth+1)


class kd_tree:
    def __init__(self, data = None, depth = 0, left = None, right = None):
        self.root = kd_node(data,depth,left,right) 
    #build the tree by input points
    def build_tree(self, points, depth):
        self.root.build_tree(points, depth)
    #preorder travelkd_tree.travel_tree()
    def travel_tree(self):
        self.root.travel()

kd_tree = kd_tree()
points = [[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]]
kd_tree.build_tree(points,0)
kd_tree.travel_tree()