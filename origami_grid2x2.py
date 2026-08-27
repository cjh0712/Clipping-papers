import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import least_squares


# ---------- 基础工具 ----------
def R(axis, angle):
    """绕任意单位轴 axis 旋转 angle（右手定则）的旋转矩阵。"""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    x, y, z = a
    c, s = np.cos(angle), np.sin(angle)
    C = 1 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def norm(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def darken(hex_color, factor=0.5):
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f'#{int(r * factor):02x}{int(g * factor):02x}{int(b * factor):02x}'


# ---------- 4 度顶点（degree-4 vertex）刚性折叠 ----------
def build_vertex(alphas, rhos):
    """给定扇形角 alphas=[a0,a1,a2,a3]（4 条折痕绕中心依次的夹角）
    与折角 rhos=[r0,r1,r2,r3]，返回 4 块面板的法向量 ns 与 4 条折痕方向 cs（3D 单位向量）。

    面板 0 固定：法向 n0=(0,0,1)，折痕 c0=(1,0,0)。
    依次绕折痕 c_i 旋转 rhos[i] 得到下一块面板。
    """
    a0, a1, a2, a3 = alphas
    r0, r1, r2, r3 = rhos
    n0 = np.array([0.0, 0.0, 1.0])
    c0 = np.array([1.0, 0.0, 0.0])
    n1 = R(c0, r0) @ n0
    c1 = np.cos(a1) * c0 + np.sin(a1) * norm(np.cross(n1, c0))
    n2 = R(c1, r1) @ n1
    c2 = np.cos(a2) * c1 + np.sin(a2) * norm(np.cross(n2, c1))
    n3 = R(c2, r2) @ n2
    c3 = np.cos(a3) * c2 + np.sin(a3) * norm(np.cross(n3, c2))
    return [n0, n1, n2, n3], [c0, c1, c2, c3]


def closure(alphas, rhos):
    """闭链残差：折痕 3 必须落回面板 0 的另一条折痕上，且折角 r3 一致。"""
    a0, *_ = alphas
    ns, cs = build_vertex(alphas, rhos)
    c3_target = np.array([np.cos(a0), -np.sin(a0), 0.0])
    r1 = cs[3] - c3_target
    r2 = R(cs[3], rhos[3]) @ ns[3] - ns[0]
    return np.concatenate([r1, r2])


def fold_path(alphas, n_half=60):
    """以折角 r0 为驱动（0 -> pi），沿光滑分支连续求解其余折角。

    返回形状 (n_half+1, 4) 的数组，每行是 [r0, r1, r2, r3]。
    """
    rho0s = np.linspace(0.0, np.pi, n_half + 1)
    path = [np.zeros(3)]                  # 依次存 [rho1, rho2, rho3]
    for rho0 in rho0s[1:]:
        # 用上一步 + 线性外推作为初值，保证沿同一分支
        prev = np.asarray(path[-1], float)
        guess = prev if len(path) == 1 else prev + (prev - np.asarray(path[-2], float))
        best = None
        for g in (guess, guess * 0.5, prev, np.zeros(3)):
            try:
                sol = least_squares(
                    lambda x: closure(alphas, [rho0, *x]), g, max_nfev=2000)
                f = np.linalg.norm(sol.fun)
                if best is None or f < best[0]:
                    best = (f, sol.x)
            except Exception:
                pass
        path.append(np.asarray(best[1], float))
    return np.hstack([rho0s[:, None], np.array(path)])


# ---------- 面板（带厚度的菱形薄板） ----------
def quad_slab(corners, normal, thickness, top_color):
    """把四边形 corners（4 个 3D 顶点，逆时针）建成带厚度的 8 顶点薄板。"""
    n = norm(normal)
    t = thickness
    top = [np.asarray(c) + (t / 2) * n for c in corners]
    bot = [np.asarray(c) - (t / 2) * n for c in corners]
    verts = top + bot                     # 0..3 顶面，4..7 底面
    side = darken(top_color)
    faces = [[0, 1, 2, 3], [4, 5, 6, 7]]  # 顶面、底面
    for i in range(4):
        j = (i + 1) % 4
        faces.append([i, j, j + 4, i + 4])
    colors = [top_color, top_color, side, side, side, side]
    return np.array(verts), np.array(faces), colors


def cyclic_corner(u_prev, u_next, r_OA, r_OB, r_AC, r_CB):
    """平铺状态下求面板外角 C = p*u_prev + q*u_next。

    面板 k 是圆内接四边形 O-A-C-B（A、B 是两条折痕的端点），
    四条边长为 |OA|=r_OA、|OB|=r_OB、|AC|=r_AC、|CB|=r_CB，
    全部取自同一个圆内接四边形 Q 的边长——这样四块面板全等。
    C 由以 A、B 为圆心的两圆交点确定，取与 O 异侧的交点。
    """
    A = r_OA * u_prev
    B = r_OB * u_next
    dvec = B - A
    d = np.linalg.norm(dvec)
    x = (d * d + r_AC * r_AC - r_CB * r_CB) / (2 * d)
    h = np.sqrt(max(r_AC * r_AC - x * x, 0.0))
    base = A + x * dvec / d
    perp = np.array([-dvec[1], dvec[0], 0.0]) / d
    C1, C2 = base + h * perp, base - h * perp
    side = lambda P: np.dot(perp, P - A)
    C = C1 if side(C1) * side(np.zeros(3)) < 0 else C2   # 取与 O 异侧的交点
    M = np.array([[u_prev[0], u_next[0]], [u_prev[1], u_next[1]]])
    p, q = np.linalg.solve(M, C[:2])
    return p, q


def build_grid(alphas, rhos, radius, thickness):
    """由扇形角、折角构造 4 块全等的圆内接四边形面板的顶点/面/颜色。

    四块面板全等于同一个圆内接四边形 Q：Q 的四个内角依次为
    alphas[0..3]（对角之和 180 度，满足圆内接条件）。圆的弦长由
    圆心角决定，因此 Q 的边长比固定；面板 k 把 Q 的顶点 v_k 放在
    中心 O、两条边落在折痕 c_{k-1}、c_k 上，于是四条折痕必须取
    不同长度 L[k]（各由 Q 的对应边长决定）。
    """
    ns, cs = build_vertex(alphas, rhos)
    c = [cs[3], cs[0], cs[1], cs[2]]     # c[k] = 面板 k 的「上一条」折痕
    n = [ns[0], ns[1], ns[2], ns[3]]

    # 四条折痕的长度：Q 的边长依次为 2R sin(a2/2)、2R sin(a0-a1+a2/2)、
    # 2R sin(a1-a2/2)、2R sin(a2/2)；取第一条为基准长 radius，其余按比例缩放。
    a0, a1, a2 = alphas[0], alphas[1], alphas[2]
    L = radius * np.array([1.0,
                           np.sin(a0 - a1 + a2 / 2) / np.sin(a2 / 2),
                           np.sin(a1 - a2 / 2) / np.sin(a2 / 2),
                           1.0])

    # 平铺状态下的折痕方向（绕中心逆时针），用于求每块面板的外角
    theta = np.cumsum([0.0, alphas[1], alphas[2], alphas[3]])
    u = [np.array([np.cos(t), np.sin(t), 0.0]) for t in theta]

    # 棋盘配色：对角两块同色
    palettes = {0: '#1565c0', 1: '#0d47a1', 2: '#1565c0', 3: '#0d47a1'}

    verts, faces, colors = [], [], []
    for k in range(4):
        O = np.zeros(3)
        A = L[(k - 1) % 4] * c[k]                # 折痕 k-1 端点
        B = L[k] * c[(k + 1) % 4]                # 折痕 k 端点
        p, q = cyclic_corner(u[(k - 1) % 4], u[k],
                             L[(k - 1) % 4], L[k],
                             L[(k + 2) % 4], L[(k + 1) % 4])
        C = p * c[k] + q * c[(k + 1) % 4]   # 平铺外角随面板刚性旋转到 3D
        corners = [O, A, C, B]           # 逆时针
        v, f, col = quad_slab(corners, n[k], thickness, palettes[k])
        base = len(verts)
        verts.extend(v)
        faces.extend([[i + base for i in face] for face in f])
        colors.extend(col)
    return np.array(verts), np.array(faces), colors


# ---------- 参数 ----------
# 4 条折痕从中心发出，扇形角依次为 110/60/70/120 度（和为 360）。
# 刻意不取 90 度、且两两不共线——这样顶点才是可折叠的（标准十字网格在刚性折纸下是锁死的）。
# 四块面板全等于同一个圆内接四边形 Q：Q 的四个内角依次就是这 4 个角
# （对角之和 180 度，恰好是圆内接条件）；圆的弦长由圆心角决定，
# 因此 Q 的边长比固定，四条折痕取不同长度（见 build_grid 中的 L）。
alphas = np.radians([110.0, 60.0, 70.0, 120.0])
radius = 1.0           # 基准边长（Q 的第一条边），其余边长按弦长公式等比缩放
thickness = 0.06       # 面板厚度

if __name__ == '__main__':
    print('求解折叠路径 ...')
    path = fold_path(alphas, n_half=60)
    frames = np.concatenate([path, path[::-1]])   # 折满再展开

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection='3d')


    def update(k):
        ax.clear()
        rhos = frames[k]
        verts, faces, colors = build_grid(alphas, rhos, radius, thickness)

        mesh = Poly3DCollection(verts[faces], facecolors=colors,
                                edgecolor='black', linewidths=0.3)
        ax.add_collection3d(mesh)

        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-2.0, 2.0)
        ax.set_zlim(-2.0, 2.0)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=28, azim=-58)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_title(f'2×2 Congruent Cyclic Quads: $\\rho_0$={np.degrees(rhos[0]):.0f} deg')


    ani = FuncAnimation(fig, update, frames=len(frames), interval=40)
    ani.save('origami_grid2x2_congruent.gif', writer='pillow', fps=25)
    print('done -> origami_grid2x2_congruent.gif')
