import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import shutil


# ----------------------------
# Function + gradient
# f(x, y) = (x^2 + y^2)/2
# ∇f(x, y) = (x, y)
# ----------------------------
def g(y):
    """
    Broadcastable version of:
      (y-1)^2 if y>1
      (y+1)^2 if y<-1
      0       otherwise
    Works for scalars, lists, numpy arrays.
    """
    y = np.asarray(y)
    return np.where(y > 1, (y - 1) ** 2, np.where(y < -1, (y + 1) ** 2, 0.0))


def dg(y):
    """
    Broadcastable derivative of g.
    """
    y = np.asarray(y)
    return np.where(y > 1, 2 * (y - 1), np.where(y < -1, 2 * (y + 1), 0.0))


def f(x, y):
    return 0.5 * (10 * x**2 + g(y + 1))


def grad_f(x, y):
    return np.array([10 * x, dg(y + 1)])


# ----------------------------
# Trajectory simulators
# ----------------------------
def run_gd(x0, y0, lr=0.25, steps=40):
    path = np.zeros((steps + 1, 2))
    path[0] = [x0, y0]
    x, y = x0, y0
    for k in range(steps):
        gx, gy = grad_f(x, y)
        x, y = x - lr * gx, y - lr * gy
        path[k + 1] = [x, y]
    return path


def run_nesterov(x0, y0, lr=0.25, beta=0.85, steps=40):
    """
    Nesterov's Accelerated Gradient (common formulation):
      lookahead = x_k + beta * v_k
      v_{k+1} = beta * v_k - lr * ∇f(lookahead)
      x_{k+1} = x_k + v_{k+1}
    """
    x = np.array([x0, y0], dtype=float)
    v = np.zeros(2, dtype=float)
    path = np.zeros((steps + 1, 2))
    path[0] = x
    for k in range(steps):
        lookahead = x + beta * v
        g = grad_f(lookahead[0], lookahead[1])
        v = beta * v - lr * g
        x = x + v
        path[k + 1] = x
    return path


# ----------------------------
# Settings
# ----------------------------
x0, y0 = -1.5, 1.8
lr = 1 / 100
steps = 100
beta = 0.95


gd_path = run_gd(x0, y0, lr=lr, steps=steps)
nag_path = run_nesterov(x0, y0, lr=lr, beta=beta, steps=steps)

gd_x, gd_y = gd_path[:, 0], gd_path[:, 1]
nag_x, nag_y = nag_path[:, 0], nag_path[:, 1]
gd_z = f(gd_x, gd_y)
nag_z = f(nag_x, nag_y)

# ----------------------------
# Surface grid (keep moderate for speed)
# ----------------------------
xmin, xmax = -3, 1
ymin, ymax = -3, 3
n = 95

X = np.linspace(xmin, xmax, n)
Y = np.linspace(ymin, ymax, n)
XX, YY = np.meshgrid(X, Y)
ZZ = f(XX, YY)

z_offset = ZZ.min() - 0.6
z_range = ZZ.max() - ZZ.min()
eps = 0.02 * z_range  # lift so paths render above surface/wireframe

# ----------------------------
# Figure setup
# ----------------------------
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

surf = ax.plot_surface(
    XX, YY, ZZ, cmap="turbo", linewidth=0, antialiased=True, alpha=0.92, zorder=4
)

# Wireframe overlay (push behind everything)
wf = ax.plot_wireframe(
    XX, YY, ZZ, rstride=6, cstride=6, color="k", linewidth=0.35, alpha=0.6
)
wf.set_sort_zpos(-1e9)

# Contours on the floor
ax.contour(
    XX,
    YY,
    ZZ,
    zdir="z",
    offset=z_offset,
    levels=18,
    cmap="turbo",
    linewidths=0.8,
    alpha=0.95,
)

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_zlim(z_offset, ZZ.max())
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("f(x, y)")

# Labels
ax.text2D(0.02, 0.96, "Solid: GD", transform=ax.transAxes, fontsize=12)
ax.text2D(0.02, 0.92, "Dashed: NAG", transform=ax.transAxes, fontsize=12)
step_text = ax.text2D(0.78, 0.96, "", transform=ax.transAxes, fontsize=12)

# Path collections (updated each frame)
gd_coll = Line3DCollection([], colors="c", linewidths=2, linestyles="--", zorder=5)
nag_coll = Line3DCollection([], colors="r", linewidths=2, linestyles="-", zorder=5)
# gd_coll.set_sort_zpos(1e9)
# nag_coll.set_sort_zpos(1e9)
ax.add_collection3d(gd_coll, autolim=False)
ax.add_collection3d(nag_coll, autolim=False)


# Moving points
gd_pt = ax.scatter(
    [gd_x[0]], [gd_y[0]], [gd_z[0] + eps], color="k", s=55, depthshade=False, zorder=5
)
nag_pt = ax.scatter(
    [nag_x[0]],
    [nag_y[0]],
    [nag_z[0] + eps],
    color="k",
    s=55,
    depthshade=False,
    zorder=5,
)
# gd_pt.set_sort_zpos(1e9)
# nag_pt.set_sort_zpos(1e9)


def segs_up_to(px, py, pz, k):
    if k < 1:
        return []
    pts = np.column_stack([px[: k + 1], py[: k + 1], pz[: k + 1] + eps])
    return np.stack([pts[:-1], pts[1:]], axis=1)


# Initial view
ax.view_init(elev=45, azim=-65)


def update(k):
    gd_coll.set_segments(segs_up_to(gd_x, gd_y, gd_z, k))
    nag_coll.set_segments(segs_up_to(nag_x, nag_y, nag_z, k))

    gd_pt._offsets3d = ([gd_x[k]], [gd_y[k]], [gd_z[k] + eps])
    nag_pt._offsets3d = ([nag_x[k]], [nag_y[k]], [nag_z[k] + eps])

    # gentle camera rotation
    # ax.view_init(elev=25, azim=-55 + 0.35 * k)

    step_text.set_text(f"step = {k}/{steps}")
    return gd_coll, nag_coll, gd_pt, nag_pt, step_text


anim = FuncAnimation(fig, update, frames=steps + 1, interval=100, blit=False)

# Save (mp4 if ffmpeg available, else gif)
out_mp4 = "gd_vs_nag_3d.mp4"
out_gif = "gd_vs_nag_3d.gif"

if shutil.which("ffmpeg"):
    anim.save(out_mp4, writer="ffmpeg", fps=10, dpi=160)
else:
    anim.save(out_gif, writer="pillow", fps=10, dpi=130)

plt.show()
