# Clipping-papers

折纸模型与动画。`planar` 表示平面面片，`nonplanar` 表示空间四边形面片。

## GeoGebra

- [2x2 planar folding](geogebra/2x2%20planar%20folding.ggb)：原始平面面片折叠。
- [2x2 nonplanar folding](geogebra/2x2%20nonplanar%20folding.ggb)：原始空间四边形折叠。
- [5x5 planar folding](geogebra/5x5%20planar%20folding.ggb)：平面面片的 5×5 扩展。
- [5x5 nonplanar folding](geogebra/5x5%20nonplanar%20folding.ggb)：25 面共点共边的几何联动；已核验新增面片会变形，不是刚性折叠。
- [5x5 nonplanar static](geogebra/5x5%20nonplanar%20static.ggb)：已确认的静态拼接参照。

用 GeoGebra 3D 打开 `.ggb`，拖动 F。`static` 文件只用于查看初态。
两个 5×5 折叠文件的代数区均按构造顺序显示点、线的定义与数值，包括平面版的 P 点及非平面版的 W 点、辅助中点；三维图中仍隐藏这些点的标记，避免遮挡面片。

## 非平面模型核验

- [5×5 手风琴式非平面刚性折叠](geogebra/5x5%20nonplanar%20accordion%20folding.ggb)：25 面、3 类形状；20 段共边可动，20 段夹角固定，横向收拢约 70.4%。拖动 `fold`。[动图](animations/5x5%20nonplanar%20accordion%20folding.gif) · [初态](previews/5x5%20nonplanar%20accordion%20folding.png) · [构造说明](proofs/5x5%20nonplanar%20accordion%20folding.txt)。
- [新的 5×5 全等非平面刚性折叠](geogebra/5x5%20nonplanar%20rigid%20screw%20folding.ggb)：改变面片几何，保留整边方格连接；拖动或播放 `fold`，25 个面均不变形。[动图](animations/5x5%20nonplanar%20rigid%20screw%20folding.gif) · [静态图](previews/5x5%20nonplanar%20rigid%20screw%20folding.png) · [构造及证明](proofs/5x5%20nonplanar%20rigid%20screw%20folding.txt)。
- [完整构造 GGB](geogebra/5x5%20nonplanar%20folding%20full%20construction.ggb)：全部 385 个点、线、面、圆等对象在代数区保留；原联动未改变，仍会形变。
- [严格证明与变形说明](output/pdf/5x5%20nonplanar%20rigidity%20proof.pdf) · [可编辑 TeX](output/pdf/5x5%20nonplanar%20rigidity%20proof.tex)：仅证明当前指定初态及整边连接不存在非平凡连续刚性折叠，不泛指所有非平面 5×5。
- 精确证书可运行 `python proofs/verify_nonplanar_rigidity.py` 复核（需要 SymPy）。

## 动画与预览

- [3×3 动画](animations/origami_grid3x3_rhombi_squares.gif) · [5×5 动画](animations/origami_grid5x5_rhombi_squares.gif) · [全部动画](animations/)
- [5×5 平面面片静态图](previews/5x5%20planar%20folding.png) · [5×5 空间面片静态图](previews/5x5%20nonplanar%20static.png)
- 网页联动预览：在本目录运行 `python -m http.server 8767 --bind 127.0.0.1`，打开 [本地预览](http://127.0.0.1:8767/previews/5x5%20nonplanar%20folding.html)。

原有 Python 代码与 `ToG.pdf` 保留；仅更新近期 GeoGebra 脚本的默认文件路径。弃用候选、试接示例及临时检查输出不再作为成果保留。
