# Clipping-papers

折纸模型与动画。`planar` 表示平面面片，`nonplanar` 表示空间四边形面片。

## GeoGebra

- [2x2 planar folding](geogebra/2x2%20planar%20folding.ggb)：原始平面面片折叠。
- [2x2 nonplanar folding](geogebra/2x2%20nonplanar%20folding.ggb)：原始空间四边形折叠。
- [5x5 planar folding](geogebra/5x5%20planar%20folding.ggb)：平面面片的 5×5 扩展。
- [5x5 nonplanar folding](geogebra/5x5%20nonplanar%20folding.ggb)：25 面共点共边的几何联动，新增面片不保证刚性。
- [5x5 nonplanar static](geogebra/5x5%20nonplanar%20static.ggb)：已确认的静态拼接参照。

用 GeoGebra 3D 打开 `.ggb`，拖动 F。`static` 文件只用于查看初态。

## 动画与预览

- [3×3 动画](animations/origami_grid3x3_rhombi_squares.gif) · [5×5 动画](animations/origami_grid5x5_rhombi_squares.gif) · [全部动画](animations/)
- [5×5 平面面片静态图](previews/5x5%20planar%20folding.png) · [5×5 空间面片静态图](previews/5x5%20nonplanar%20static.png)
- 网页联动预览：在本目录运行 `python -m http.server 8767 --bind 127.0.0.1`，打开 [本地预览](http://127.0.0.1:8767/previews/5x5%20nonplanar%20folding.html)。

原有 Python 代码与 `ToG.pdf` 保留；仅更新近期 GeoGebra 脚本的默认文件路径。弃用候选、试接示例及临时检查输出不再作为成果保留。
