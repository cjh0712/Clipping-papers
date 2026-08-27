import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def fold_chain(M, panel_w, angles):
    """零厚度线铰链折叠：返回每块面板的朝向 theta 和铰链位置 (Hx, Hz)。"""
    theta = np.zeros(M)
    for p in range(1, M):
        theta[p] = theta[p - 1] + angles[p - 1]
    Hx = np.zeros(M + 1)
    Hz = np.zeros(M + 1)
    for p in range(M):
        Hx[p + 1] = Hx[p] + panel_w * np.cos(theta[p])
        Hz[p + 1] = Hz[p] + panel_w * np.sin(theta[p])
    return theta, Hx, Hz


def darken(hex_color, factor=0.5):
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f'#{int(r * factor):02x}{int(g * factor):02x}{int(b * factor):02x}'


def build_slabs(M, panel_w, panel_h, thickness, angles, progress):
    """把每块面板建成带厚度的薄板（8 顶点 + 6 面），折痕处无缝贴合。

    progress : 折叠进度 0..1，控制层叠抬高量，折满时正好是一摞。
    """
    theta, Hx, Hz = fold_chain(M, panel_w, angles)
    verts = []
    faces = []
    colors = []
    for p in range(M):
        d = np.array([np.cos(theta[p]), 0.0, np.sin(theta[p])])   # 沿面板方向
        n = np.array([-np.sin(theta[p]), 0.0, np.cos(theta[p])])  # 面板法向（平放时朝上）
        s = (p + 0.5) * thickness * progress                       # 层叠抬高
        H = np.array([Hx[p], 0.0, Hz[p] + s])                     # 面板左下参考点

        base = len(verts)
        for iy, y in enumerate((-panel_h / 2, panel_h / 2)):
            for iu, u in enumerate((0.0, panel_w)):
                for io, off in enumerate((-thickness / 2, thickness / 2)):
                    pt = H + u * d + off * n
                    pt[1] = y
                    verts.append(pt)

        def idx(iy, iu, io):
            return base + iy * 4 + iu * 2 + io

        c_top = '#1565c0' if p % 2 == 0 else '#0d47a1'
        c_side = darken(c_top)
        # 顶面、底面、四个侧面
        faces.append([idx(0, 0, 1), idx(0, 1, 1), idx(1, 1, 1), idx(1, 0, 1)])
        faces.append([idx(0, 0, 0), idx(0, 1, 0), idx(1, 1, 0), idx(1, 0, 0)])
        faces.append([idx(0, 0, 0), idx(0, 1, 0), idx(0, 1, 1), idx(0, 0, 1)])
        faces.append([idx(1, 0, 0), idx(1, 1, 0), idx(1, 1, 1), idx(1, 0, 1)])
        faces.append([idx(0, 0, 0), idx(1, 0, 0), idx(1, 0, 1), idx(0, 0, 1)])
        faces.append([idx(0, 1, 0), idx(1, 1, 0), idx(1, 1, 1), idx(0, 1, 1)])
        colors += [c_top, c_top, c_side, c_side, c_side, c_side]

    return np.array(verts), np.array(faces), colors


# ---------- 参数 ----------
M = 6
panel_w, panel_h = 1.8, 1.5      # 放大面板
thickness = 0.25                 # 单层面板厚度

max_angle = np.deg2rad(178)
half = np.linspace(0, max_angle, 90)
frames = np.concatenate([half, half[::-1]])

fig = plt.figure(figsize=(11, 6))
ax = fig.add_subplot(111, projection='3d')


def update(k):
    ax.clear()
    th = frames[k]
    angles = th * np.array([(-1) ** p for p in range(M - 1)])   # 山/谷交替
    verts, faces, colors = build_slabs(M, panel_w, panel_h, thickness, angles, th / np.pi)

    mesh = Poly3DCollection(verts[faces], facecolors=colors,
                            edgecolor='black', linewidths=0.3)
    ax.add_collection3d(mesh)

    ax.set_xlim(-1.2, 12.0)
    ax.set_ylim(-1.4, 1.4)
    ax.set_zlim(-0.8, 6.4)
    ax.set_box_aspect((13.2, 2.8, 7.2))
    ax.view_init(elev=25, azim=-65)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_title(f'Solar Array Fold: {np.degrees(th):.0f} deg  ({M} panels)')


ani = FuncAnimation(fig, update, frames=len(frames), interval=40)
ani.save('origami_solar.gif', writer='pillow', fps=25)
print('done -> origami_solar.gif')
