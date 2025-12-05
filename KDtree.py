from typing import List, Optional, Tuple, Any


class KDNode:
    def __init__(self, point: Tuple[float, ...], axis: int,
                 left: Optional["KDNode"] = None, right: Optional["KDNode"] = None):
        self.point = point
        self.axis = axis
        self.left = left
        self.right = right

    def __repr__(self):
        return f"KDNode(point={self.point}, axis={self.axis})"


class KDTree:
    """
    Simple KD-tree implementation for k-dimensional points.
    - Build: median split by axis = depth % k (stable sort, O(n log n) total).
    - Search: range search using coordinate comparisons to prune branches.
    """

    def __init__(self, points: List[Tuple[float, ...]]):
        self.k = 0 if not points else len(points[0])
        self.root = self._build(points, depth=0) if points else None

    def _build(self, points: List[Tuple[float, ...]], depth: int) -> Optional[KDNode]:
        if not points:
            return None
        if len(points) == 1:
            return KDNode(points[0], axis=depth % self.k)
        axis = depth % self.k
        # sort by current axis and pick median
        points_sorted = sorted(points, key=lambda p: p[axis])
        mid = len(points_sorted) // 2
        # median point becomes the node; left = points before median, right = after
        node = KDNode(points_sorted[mid], axis=axis)
        node.left = self._build(points_sorted[:mid], depth + 1)
        node.right = self._build(points_sorted[mid + 1 :], depth + 1)
        return node

    def range_search(self, region: List[Tuple[float, float]]) -> List[Tuple[float, ...]]:
        """
        region: list of (low, high) pairs, length == k
        returns list of points inside the region (inclusive)
        """
        if self.root is None:
            return []
        assert len(region) == self.k, "Region dimensionality must match points"

        result: List[Tuple[float, ...]] = []
        self._range_search_rec(self.root, region, result)
        return result

    def _point_in_region(self, point: Tuple[float, ...], region: List[Tuple[float, float]]) -> bool:
        for coord, (lo, hi) in zip(point, region):
            if coord < lo or coord > hi:
                return False
        return True

    def _range_search_rec(self, node: Optional[KDNode], region: List[Tuple[float, float]], out: List[Tuple[float, ...]]):
        if node is None:
            return
        # If node.point is inside region, report it
        if self._point_in_region(node.point, region):
            out.append(node.point)
        axis = node.axis
        lo, hi = region[axis]
        coord = node.point[axis]
        # If region's low <= coord, left subtree might intersect
        if lo <= coord:
            self._range_search_rec(node.left, region, out)
        # If region's high >= coord, right subtree might intersect
        if hi >= coord:
            self._range_search_rec(node.right, region, out)

    # Added: nearest neighbor search (returns nearest point and Euclidean distance)
    def nearest(self, target: Tuple[float, ...]) -> Tuple[Optional[Tuple[float, ...]], float]:
        """
        Find nearest neighbor to target using the KD-tree (Euclidean on coordinate values).
        Returns (nearest_point, euclidean_distance). If tree empty returns (None, inf).
        """
        if self.root is None:
            return None, float("inf")
        best_point: Optional[Tuple[float, ...]] = None
        best_dist2 = float("inf")

        def rec(node: Optional[KDNode]):
            nonlocal best_point, best_dist2
            if node is None:
                return
            pt = node.point
            # squared Euclidean distance
            d2 = sum((c - t) ** 2 for c, t in zip(pt, target))
            if d2 < best_dist2:
                best_dist2 = d2
                best_point = pt
            axis = node.axis
            diff = target[axis] - pt[axis]
            # search nearer side first
            first = node.left if diff < 0 else node.right
            second = node.right if diff < 0 else node.left
            rec(first)
            # if hypersphere crosses splitting plane, search other side
            if diff * diff < best_dist2:
                rec(second)

        rec(self.root)
        return best_point, best_dist2 ** 0.5


if __name__ == "__main__":
    import random
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import os

    random.seed(0)

    # generate 2000 random 2D points uniformly in [-10, 10] x [-10, 10]
    pts = [(random.uniform(-10, 10), random.uniform(-10, 10)) for _ in range(2000)]

    tree = KDTree(pts)

    regions = [
        [(-1.0, 1.0), (-2.0, 2.0)],
        [(-2.0, 1.0), (3.0, 5.0)],
        [(-7.0, 0.0), (-6.0, 4.0)],
        [(-2.0, 2.0), (-3.0, 3.0)],
        [(-7.0, 5.0), (-3.0, 1.0)],
    ]

    # prepare plotting: 2 rows x 3 cols (first plot = full point cloud, then one per region)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Plot 0: all points
    ax0 = axes[0]
    xs_all = [p[0] for p in pts]
    ys_all = [p[1] for p in pts]
    ax0.scatter(xs_all, ys_all, s=8, color="#bbbbbb", label="all points")
    ax0.set_title("All points (2000)")
    ax0.set_xlim(-11, 11)
    ax0.set_ylim(-11, 11)
    ax0.set_aspect("equal", adjustable="box")
    ax0.grid(True)

    # For each region, search and plot
    for i, region in enumerate(regions, start=1):
        ax = axes[i]
        # plot all points faintly
        ax.scatter(xs_all, ys_all, s=8, color="#eeeeee")
        # do range search using KD-tree
        found = tree.range_search(region)
        fx = [p[0] for p in found]
        fy = [p[1] for p in found]
        ax.scatter(fx, fy, s=20, color="red", label=f"found: {len(found)}")
        # draw rectangle for region (x,y)
        xlo, xhi = region[0]
        ylo, yhi = region[1]
        rect = Rectangle((xlo, ylo), xhi - xlo, yhi - ylo, linewidth=2, edgecolor="blue", facecolor="none")
        ax.add_patch(rect)
        ax.set_title(f"Region {i}: x∈[{xlo},{xhi}], y∈[{ylo},{yhi}] → {len(found)} pts")
        ax.set_xlim(-11, 11)
        ax.set_ylim(-11, 11)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True)

    axes[-1].axis("off")

    plt.tight_layout()

    out_path = os.path.join(os.path.dirname(__file__), "kdtree_regions.png")
    fig.savefig(out_path, dpi=300)
    print(f"Saved figure to {out_path}")

    plt.show()

    # print counts and example points
    for j, region in enumerate(regions, start=1):
        res = tree.range_search(region)
        print(f"Region {j} bounds x={region[0]}, y={region[1]} -> {len(res)} points")
        if len(res) <= 20:
            print(res)
        else:
            print("First 20:", res[:20])