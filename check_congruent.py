"""验证 4 块面板全等：平铺状态边长/内角逐一核对，折叠全程刚性保持。"""
import numpy as np
import origami_grid2x2 as og

alphas, radius, thickness = og.alphas, og.radius, og.thickness

a0, a1, a2 = alphas[0], alphas[1], alphas[2]
L = radius * np.array([1.0,
                       np.sin(a0 - a1 + a2 / 2) / np.sin(a2 / 2),
                       np.sin(a1 - a2 / 2) / np.sin(a2 / 2),
                       1.0])
print('crease lengths L =', np.round(L, 4))

theta = np.cumsum([0.0, alphas[1], alphas[2], alphas[3]])
u = [np.array([np.cos(t), np.sin(t), 0.0]) for t in theta]


def flat_corners(k):
    A = L[(k - 1) % 4] * u[(k - 1) % 4]
    B = L[k] * u[k]
    p, q = og.cyclic_corner(u[(k - 1) % 4], u[k],
                            L[(k - 1) % 4], L[k],
                            L[(k + 2) % 4], L[(k + 1) % 4])
    return [np.zeros(3), A, p * u[(k - 1) % 4] + q * u[k], B]


def stats(pts):
    sides = [np.linalg.norm(pts[(i + 1) % 4] - pts[i]) for i in range(4)]
    angles = []
    for i in range(4):
        va, vb = pts[(i - 1) % 4] - pts[i], pts[(i + 1) % 4] - pts[i]
        angles.append(np.degrees(np.arccos(np.clip(
            np.dot(va, vb) / np.linalg.norm(va) / np.linalg.norm(vb), -1.0, 1.0))))
    return sides, angles


def cyclic_eq(seq, ref, tol=1e-6):
    """序列在循环轮换意义下相等。"""
    n = len(ref)
    return any(all(abs(seq[(i + s) % n] - ref[i]) < tol for i in range(n))
               for s in range(n))


print('\n== flat state ==')
flat = {k: stats(flat_corners(k)) for k in range(4)}
for k in range(4):
    s, a = flat[k]
    print(f'panel {k}: sides={np.round(s, 4)}  angles={np.round(a, 1)}')

ref_s, ref_a = flat[0]
assert all(cyclic_eq(flat[k][0], ref_s) and cyclic_eq(flat[k][1], ref_a)
           for k in range(4)), 'panels are NOT congruent!'
assert np.allclose(sorted(ref_a), sorted(np.degrees(alphas)), atol=1e-9), \
    'interior angles are not [110,60,70,120]!'
print('all 4 panels have identical side/angle sequences -> congruent')
print('angle set =', np.round(sorted(ref_a), 1), 'deg')

# 折叠全程：每块面板边长保持平铺值（刚性），共享折痕端点（中面）一致。
# 注意 verts 前 4 个是顶面（中面 + t/2*n），相邻面板法向不同，
# 顶面角点本身不重合；真正的共享端点在每条折痕线的中面上。
print('\n== rigidity check along fold path ==')
path = og.fold_path(alphas, n_half=60)
worst_len = worst_end = 0.0


def mid_plane(P):
    """由顶面 4 点恢复中面坐标（沿法向回退 t/2）。"""
    n = np.cross(P[1] - P[0], P[3] - P[0])
    n = n / np.linalg.norm(n)
    return [p - (thickness / 2) * n for p in P]


for rhos in path[::10]:
    verts, _, _ = og.build_grid(alphas, rhos, radius, thickness)
    for k in range(4):
        P = [np.asarray(verts[k * 8 + i]) for i in range(4)]
        for i in range(4):
            d = np.linalg.norm(P[(i + 1) % 4] - P[i]) - flat[k][0][i]
            worst_len = max(worst_len, abs(d))
        Bk = mid_plane(P)[3]                        # 面板 k 的 B = 折痕 k 端点
        P1 = [np.asarray(verts[((k + 1) % 4) * 8 + i]) for i in range(4)]
        Ak1 = mid_plane(P1)[1]                      # 面板 k+1 的 A = 同一端点
        worst_end = max(worst_end, np.linalg.norm(Bk - Ak1))
print(f'max panel side deviation = {worst_len:.2e}')
print(f'max shared-crease-endpoint gap = {worst_end:.2e}')
assert worst_len < 1e-6 and worst_end < 1e-6
print('all checks passed')
