import cv2
import networkx
from libpysal import weights, examples
from libpysal.cg import voronoi_frames
import matplotlib as plt

def extractCoords():
    im = cv2.imread('floor1.jpg')
    ## convert to hsv
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    ## mask of green (36,0,0) ~ (70, 255,255)
    mask1 = cv2.inRange(hsv, (36, 0, 0), (70, 255,255))

    ## mask o yellow (15,0,0) ~ (36, 255, 255)
    mask2 = cv2.inRange(hsv, (15,0,0), (36, 255, 255))

    ## final mask and masked
    mask = cv2.bitwise_or(mask1, mask2)
    target = cv2.bitwise_and(im,im, mask=mask)

def makeGraph():
    ### credit: https://networkx.org/documentation/stable/auto_examples/geospatial/plot_delaunay.html#sphx-glr-auto-examples-geospatial-plot-delaunay-py
    coordinates = np.column_stack((cases.geometry.x, cases.geometry.y))
    cells, generators = voronoi_frames(coordinates, clip="convex hull")
    delaunay = weights.Rook.from_dataframe(cells)
    delaunay_graph = delaunay.to_networkx()

    positions = dict(zip(delaunay_graph.nodes, coordinates))

    # Now, we can plot with a nice basemap.
    ax = cells.plot(facecolor="lightblue", alpha=0.50, edgecolor="cornsilk", linewidth=2)
    add_basemap(ax)
    ax.axis("off")
    nx.draw(
        delaunay_graph,
        positions,
        ax=ax,
        node_size=2,
        node_color="k",
        edge_color="k",
        alpha=0.8,
    )
    plt.show()