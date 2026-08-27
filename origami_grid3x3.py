"""3×3 等角四边形网格（9 面全等，论文 Example 3）刚性折叠动画 v2。

v2 改动（相对 v1）：
- 完整 3×3：中心面 + 4 边中面 + 4 角面，共 9 个全等面
- 边中面拼法：中心面绕其质心旋转 180°（整个面反过来），再平移拼接——
  共享边位置不动，共享边两端的面顶角互换（交叉配对）。
  例如左侧面的上边 = 中心面的下边（同为 60°-105° 那条边）
- 角面：两条相邻边中面的外边端点 + 圆交确定的第 4 角闭合，
  平铺状态 4 个中心顶点处 4 面角度和恰为 360°
- 折叠：二面角大小走 Bricard 分支 k1·K2·k3·K4（α1 驱动 0→178°），
  方向：上侧 T 朝下叠、下侧 B 朝上叠，L/R 朝下叠；
  角面每帧用三点刚性配准跟随两个相邻边中面，保证所有共边全程贴合不断开
- v5 修复：角面为保逆时针可能交换 A/B 角序（TL/TR 两个角面被翻转），三点配准
  与角点归属一律按平铺坐标匹配 A/B 的真实索引，否则这两个角面在折叠初期被
  近 180° 翻转渲染（视觉上"先变大"）
- v4 共边修复：每个角点的厚度偏移按其归属面法向取——中心 4 角点属中心面
  （偏移恒为 t/2·ẑ，折叠中不动），外边共享角点属所在边中面；相邻面板在
  共边上的顶边、底边、厚度侧面完全相同，12 条共边及全部角点全程精确贴合
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


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


def circle_circle(A, B, rA, rB):
    """以 A、B 为圆心、半径 rA、rB 的两圆交点，返回两个候选点。"""
    dvec = B - A
    d = np.linalg.norm(dvec)
    x = (d * d + rA * rA - rB * rB) / (2 * d)
    h = np.sqrt(max(rA * rA - x * x, 0.0))
    base = A + x * dvec / d
    perp = np.array([-dvec[1], dvec[0], 0.0]) / d
    return base + h * perp, base - h * perp


def rigid_from_points(x1, x2, y1, y2):
    """旋转矩阵把 (x1, x2, x1×x2) 映射到 (y1, y2, y1×y2)。"""
    X = np.column_stack([x1, x2, np.cross(x1, x2)])
    Y = np.column_stack([y1, y2, np.cross(y1, y2)])
    return Y @ np.linalg.inv(X)


# ---------- 圆内接四边形 ----------
def cyclic_quad(lam_deg, circ_R=1.0):
    """内角序列 lam_deg=[λ1..λ4]（逆时针，对角互补）→ 顶点 4×2（逆时针）。

    相邻顶点间半圆心角 t_i 满足 λ_i = t_{i+1} + t_{i+2}（下标循环），
    故 t1 = (λ3+λ4−λ2)/2，其余依次解出。v1 放在 135°（左上），
    顶点按弧长 2t_i 前进。返回质心平移到原点的顶点。
    """
    lam = np.radians(lam_deg)
    t1 = (lam[2] + lam[3] - lam[1]) / 2
    t4 = lam[2] - t1
    t3 = lam[1] - t4
    t2 = lam[0] - t3
    t = np.array([t1, t2, t3, t4])
    theta = np.radians(135.0) + np.concatenate([[0.0], np.cumsum(2 * t)[:-1]])
    q = circ_R * np.column_stack([np.cos(theta), np.sin(theta)])
    return q - q.mean(axis=0)


# ---------- 参数 ----------
lam_deg = [75.0, 60.0, 105.0, 120.0]     # λ1..λ4（逆时针）
lam = np.radians(lam_deg)
gam = lam[[3, 0, 1, 2]]                  # γ_i = λ_{i-1}

V = cyclic_quad(lam_deg)                 # V[i] 处内角 λ_{i+1}
V3 = np.column_stack([V, np.zeros(4)])   # 3D（z=0 平面）
lens = [np.linalg.norm(V3[(i + 1) % 4] - V3[i]) for i in range(4)]  # l1..l4

# ---------- 边中面（R180 + 平移，交叉配对） ----------
# 面 i（1..4）：铰链 = 中心边 (v_i, v_{i-1})，平移 t = v_i + v_{i-1}，
# 角点 CCW = [−v3, −v4, −v1, −v2] + t（180° 旋转副本的标准角序）。
edge = {}
hinges = {}                              # i -> (铰链上一点 a, 铰链轴 u)
for i in range(1, 5):
    vi, vj = V3[i - 1], V3[(i - 2) % 4]
    t = vi + vj
    corners = np.array([-V3[2] + t, -V3[3] + t, -V3[0] + t, -V3[1] + t])
    if np.cross(corners[1] - corners[0], corners[3] - corners[0])[2] < 0:
        corners = corners[[0, 3, 2, 1]]  # 保证逆时针
    edge[i] = corners
    o = corners.mean(axis=0) - (vi + vj) / 2
    u = norm(vj - vi)                                    # 铰链轴 = 共边方向
    if np.dot(np.cross(u, o), [0.0, 0.0, 1.0]) < 0:
        u = -u                                           # 正角抬向 +z
    hinges[i] = (vi, u)

# ---------- 角面（圆交闭合） ----------
def corner_face(v, A, rA, B, rB):
    """角面 = [v, A, C, B]（CCW），C 取两圆交点中离原点（中心面质心）较远者。"""
    C1, C2 = circle_circle(A, B, rA, rB)
    C = C1 if np.linalg.norm(C1) >= np.linalg.norm(C2) else C2
    corners = np.array([v, A, C, B])
    if np.cross(corners[1] - corners[0], corners[3] - corners[0])[2] < 0:
        corners = corners[[0, 3, 2, 1]]
    return corners


# 角面定义：中心顶点 v，A/B 分别是两个边中面的外边角点（(面号, 角点号)），
# C 到 A/B 的距离 rA/rB（圆交半径）。
corner_defs = [
    (V3[0], 2, 0, lens[0], 1, 0, lens[3]),  # TL：A=f2[0](r=l1), B=f1[0](r=l4)
    (V3[1], 2, 1, lens[0], 3, 1, lens[1]),  # BL：A=f2[1](r=l1), B=f3[1](r=l2)
    (V3[2], 3, 2, lens[1], 4, 2, lens[2]),  # BR：A=f3[2](r=l2), B=f4[2](r=l3)
    (V3[3], 1, 3, lens[3], 4, 3, lens[2]),  # TR：A=f1[3](r=l4), B=f4[3](r=l3)
]
corner = []
corner_anchor_idx = []          # 每个角面中 A、B 的真实索引（角序可能被 CCW 翻转交换）
for (v, iA, jA, rA, iB, jB, rB) in corner_defs:
    face = corner_face(v, edge[iA][jA], rA, edge[iB][jB], rB)
    corner.append(face)
    idxA = int(np.argmin([np.linalg.norm(p - edge[iA][jA]) for p in face]))
    idxB = int(np.argmin([np.linalg.norm(p - edge[iB][jB]) for p in face]))
    corner_anchor_idx.append((idxA, idxB))

faces_flat = [V3, edge[1], edge[2], edge[3], edge[4],
              corner[0], corner[1], corner[2], corner[3]]
# 索引：0=中心面，1..4=边中面（T/L/B/R），5..8=角面（TL/BL/BR/TR）

# ---------- 折叠路径（Bricard 分支 k1·K2·k3·K4） ----------
def fold_path(n_half=60, amax_deg=178.0):
    """驱动 x1 = tan(α1/2)，α1: 0 → amax。返回 (n_half+1, 4) 的 α（度）。"""
    k = (np.sin(gam) - np.sin(lam)) / np.sin(gam - lam)
    K = (np.sin(gam) + np.sin(lam)) / np.sin(gam - lam)
    a1 = np.linspace(0.0, amax_deg, n_half + 1)
    x1 = np.tan(np.radians(a1) / 2)
    x2 = k[0] * x1
    x3 = K[1] * x2
    x4 = k[2] * x3
    close_err = np.max(np.abs(K[3] * x4 - x1))     # 闭链残差，应 ≈ 0
    alphas = np.degrees(2 * np.arctan(np.stack([x1, x2, x3, x4], axis=1)))
    return alphas, close_err


# ---------- 面板（带厚度的薄板） ----------
def quad_slab(corners, offsets, top_color, crease=(), hinge_color='#37474f'):
    """把四边形 corners（4 个 3D 顶点，逆时针）建成带厚度的 8 顶点薄板。

    offsets: 每个角点顶面偏移（顶面 = 中面角点 + off，底面 = 中面角点 − off）。
    折痕边（crease）的厚度侧面用铰链色——相邻两块板在共边上生成完全相同的
    顶边、底边与侧面，因此共边及其端点角在折叠全程精确贴合、永不分离。
    """
    top = [np.asarray(c) + np.asarray(o) for c, o in zip(corners, offsets)]
    bot = [np.asarray(c) - np.asarray(o) for c, o in zip(corners, offsets)]
    verts = top + bot
    faces = [[0, 1, 2, 3], [4, 5, 6, 7]]
    colors = [top_color, top_color]
    for i in range(4):
        j = (i + 1) % 4
        faces.append([i, j, j + 4, i + 4])
        colors.append(hinge_color if i in crease else darken(top_color))
    return np.array(verts), faces, colors


thickness = 0.06                          # 面板厚度
# 棋盘配色：中心面与 4 个角面深蓝，4 个边中面浅蓝
palettes = {0: '#0d47a1', 1: '#1565c0', 2: '#1565c0', 3: '#1565c0',
            4: '#1565c0', 5: '#0d47a1', 6: '#0d47a1', 7: '#0d47a1',
            8: '#0d47a1'}


def build_mesh(alphas):
    """由 4 个二面角（度）构造 9 块面板，返回 (顶点, 面, 颜色, 配准残差)。

    共边贴合方案：每个角点的厚度偏移按「归属面」的法向取——中心 4 个角点
    属于中心面，偏移恒为 t/2·ẑ，折叠中纹丝不动（例如左面的右上角与中心面
    的左上角始终是同一个点）；外边共享角点属于所在边中面，相邻两块板用
    同一个偏移，顶角、底角逐点重合；每条折痕的厚度侧面（铰链色）两块板
    完全相同。因此 12 条共边及全部角点全程精确贴合、永不分离。
    """
    # 中心面固定，边中面绕各自铰链旋转
    cur = [faces_flat[0]]
    normals = [np.array([0.0, 0.0, 1.0])]
    for i in range(1, 5):
        a, u = hinges[i]
        mag = abs(alphas[i - 1])                    # 二面角大小（Bricard 分支）
        ang = np.radians(mag if i == 3 else -mag)   # 上侧 T 下叠、下侧 B 上叠，L/R 下叠
        cur.append(np.array([a + R(u, ang) @ (p - a) for p in faces_flat[i]]))
        normals.append(R(u, ang) @ np.array([0.0, 0.0, 1.0]))
    # 角面：三点刚性配准，跟随相邻两个边中面
    res = 0.0
    for c, (v, iA, jA, rA, iB, jB, rB) in enumerate(corner_defs):
        idxA, idxB = corner_anchor_idx[c]                     # A/B 在角面中的真实索引
        A0, B0 = faces_flat[5 + c][idxA], faces_flat[5 + c][idxB]
        Ak = cur[iA][jA]                                       # 边中面当前角点
        Bk = cur[iB][jB]
        Rk = rigid_from_points(A0 - v, B0 - v, Ak - v, Bk - v)
        cur.append(np.array([v + Rk @ (p - v) for p in faces_flat[5 + c]]))
        res = max(res, np.linalg.norm(Rk @ (A0 - v) - (Ak - v)),
                  np.linalg.norm(Rk @ (B0 - v) - (Bk - v)))
        normals.append(norm(np.cross(cur[5 + c][1] - cur[5 + c][0],
                                     cur[5 + c][3] - cur[5 + c][0])))
    # 角点归属：谁的面法向决定该角点的厚度偏移
    owners = {0: [0, 0, 0, 0]}                     # 中心面：全部角点属于自己
    for i in range(1, 5):
        hinge_idx = {i % 4, (i + 1) % 4}           # 铰链边两端角点 → 属于中心面
        owners[i] = [0 if j in hinge_idx else i for j in range(4)]
    for c in range(4):
        idxA, idxB = corner_anchor_idx[c]
        own = [0, 0, 5 + c, 0]
        own[idxA] = corner_defs[c][1]         # 角面 [v, A, C, B]：v 属中心面，
        own[idxB] = corner_defs[c][4]         # A/B 属所在边中面，C 属自己
        owners[5 + c] = own
    crease = {0: {0, 1, 2, 3}}                     # 折痕边（厚度侧面用铰链色）
    crease.update({i: {i % 4} for i in range(1, 5)})
    crease.update({i: {0, 3} for i in range(5, 9)})
    # 生成薄板
    verts, faces, colors = [], [], []
    for i, corners in enumerate(cur):
        offs = [(thickness / 2) * normals[owners[i][j]] for j in range(4)]
        v, f, col = quad_slab(corners, offs, palettes[i], crease=crease[i])
        base = len(verts)
        verts.extend(v)
        faces.extend([[j + base for j in face] for face in f])
        colors.extend(col)
    return np.array(verts), faces, colors, res


def face_stats(pts):
    """四边形的边长序列与内角序列（度）。"""
    pts = np.asarray(pts)
    sides = [np.linalg.norm(pts[(i + 1) % 4] - pts[i]) for i in range(4)]
    ang = []
    for i in range(4):
        va = pts[(i - 1) % 4] - pts[i]
        vb = pts[(i + 1) % 4] - pts[i]
        ang.append(np.degrees(np.arccos(np.clip(
            np.dot(va, vb) / np.linalg.norm(va) / np.linalg.norm(vb), -1, 1))))
    return sides, ang


def cyclic_eq(seq, ref, tol=1e-6):
    """序列在循环轮换意义下与 ref 相等，返回轮换步数（不相等返回 None）。"""
    n = len(ref)
    for s in range(n):
        if all(abs(seq[(i + s) % n] - ref[i]) < tol for i in range(n)):
            return s
    return None


if __name__ == '__main__':
    # 平铺状态检查：9 面严格全等——边长序与内角序都在循环轮换意义下相等
    print('== flat state: strict congruence (cyclic order) ==')
    ref = face_stats(faces_flat[0])
    print(f'central: sides={np.round(ref[0], 3)}  angles={np.round(ref[1], 1)}')
    for i in range(9):
        s, a = face_stats(faces_flat[i])
        ss, sa = cyclic_eq(s, ref[0]), cyclic_eq(a, ref[1])
        print(f'face {i}: sides={np.round(s, 3)} angles={np.round(a, 1)}'
              f'  shift_sides={ss}  shift_angles={sa}')
        assert ss is not None and sa is not None, f'face {i} NOT congruent!'
    assert np.allclose(sorted(ref[1]), [60.0, 75.0, 105.0, 120.0], atol=1e-9)
    print('=> 9 faces strictly congruent (same cyclic side/angle sequences)')

    print('求解折叠路径 ...')
    path, err = fold_path()
    print(f'Bricard closure residual = {err:.2e}')

    # 折叠全程角面配准残差（中面贴合检查）
    worst = 0.0
    for al in path[::5]:
        _, _, _, res = build_mesh(al)
        worst = max(worst, res)
    print(f'max corner-face registration residual = {worst:.2e}')
    assert worst < 1e-6, 'corner faces do not close along the fold path!'

    # 折叠全程面板角点贴合检查：每条共边两端，相邻面板的顶/底角点应逐点重合
    # （左面的右上角 == 中心面的左上角，等等）
    worst_gap = 0.0
    for al in path[::5]:
        verts, _, _, _ = build_mesh(al)   # 面板 i 的 8 个顶点位于 verts[8i:8i+8]
        top = lambda i, j: verts[8 * i + j]
        bot = lambda i, j: verts[8 * i + 4 + j]
        for i in range(1, 5):             # 中心折痕两端：中心角点 vs 边中面角点
            worst_gap = max(
                worst_gap,
                np.linalg.norm(top(0, i - 1) - top(i, i % 4)),
                np.linalg.norm(top(0, (i - 2) % 4) - top(i, (i + 1) % 4)),
                np.linalg.norm(bot(0, i - 1) - bot(i, i % 4)),
                np.linalg.norm(bot(0, (i - 2) % 4) - bot(i, (i + 1) % 4)))
        for c, (v, iA, jA, rA, iB, jB, rB) in enumerate(corner_defs):
            idxA, idxB = corner_anchor_idx[c]
            worst_gap = max(
                worst_gap,
                np.linalg.norm(top(5 + c, 0) - top(0, c)),
                np.linalg.norm(top(5 + c, idxA) - top(iA, jA)),
                np.linalg.norm(top(5 + c, idxB) - top(iB, jB)),
                np.linalg.norm(bot(5 + c, 0) - bot(0, c)),
                np.linalg.norm(bot(5 + c, idxA) - bot(iA, jA)),
                np.linalg.norm(bot(5 + c, idxB) - bot(iB, jB)))
    print(f'max slab corner gap along path = {worst_gap:.2e}')
    assert worst_gap < 1e-6, 'slab corners separate along the fold path!'
    applied = np.abs(path[-1]) * [-1, -1, 1, -1]
    print(f'end of fold: applied angles (T,L,B,R) = {np.round(applied, 1)} deg'
          f'  (+ up / - down)')

    frames = np.concatenate([path, path[::-1]])   # 折满再展开

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection='3d')

    def update(k):
        ax.clear()
        verts, faces, colors, _ = build_mesh(frames[k])
        mesh = Poly3DCollection([verts[f] for f in faces], facecolors=colors,
                                edgecolor='black', linewidths=0.3)
        ax.add_collection3d(mesh)
        ax.set_xlim(-3.1, 3.1)
        ax.set_ylim(-3.1, 3.1)
        ax.set_zlim(-3.1, 3.1)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=28, azim=-58)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_title(f'3×3 Congruent Mesh (Ex.3): '
                     f'$\\alpha_1$={frames[k, 0]:.0f} deg')

    ani = FuncAnimation(fig, update, frames=len(frames), interval=40)
    ani.save('origami_grid3x3.gif', writer='pillow', fps=25)
    print('done -> origami_grid3x3.gif')
