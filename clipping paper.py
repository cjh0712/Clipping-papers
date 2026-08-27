import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')

# 假设 data 是形状为 (总帧数, 四边形数量, 4个顶点, 3坐标) 的numpy数组
# data = your_calculation_function() 

def update(frame):
    ax.clear()
    ax.set_xlim(-1,1); ax.set_ylim(-1,1); ax.set_zlim(-1,1)
    # 关键：绘制面片集合
    mesh = Poly3DCollection(data[frame], alpha=0.4, facecolor='cyan', edgecolor='black', linewidth=1.2)
    ax.add_collection3d(mesh)
    ax.view_init(elev=25, azim=frame*2) # 缓慢自转，提升空间感
    ax.set_title(f'Fold Angle: {frame}°')

ani = FuncAnimation(fig, update, frames=len(data), interval=50)
ani.save('origami_folding.mp4', writer='ffmpeg', fps=20) # 直接存MP4投稿