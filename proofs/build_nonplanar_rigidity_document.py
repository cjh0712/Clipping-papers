"""Build the Chinese proof as PDF and editable XeLaTeX from shared content blocks."""

import argparse
from html import escape
import json
from pathlib import Path
import re


PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / 'output/pdf'
TEMPORARY = PROJECT.parent / 'tmp/pdfs/nonplanar-rigidity'
CERTIFICATE = Path(__file__).with_name('5x5 nonplanar rigidity certificate.json')
TITLE = '当前 5×5 非平面拼接的刚性证明'
STEM = '5x5 nonplanar rigidity proof'
REFERENCE_URL = 'https://pi.math.cornell.edu/~connelly/pdf/10.1137_S0895480192229236.pdf'


def content(report):
    blocks = []

    def heading(text):
        blocks.append(('heading', text))

    def paragraph(text):
        blocks.append(('paragraph', text))

    def formula(text):
        blocks.append(('formula', text))

    def table(headers, rows, widths=None):
        blocks.append(('table', {'headers': headers, 'rows': rows, 'widths': widths}))

    def page():
        blocks.append(('page', None))

    blocks.append(('title', TITLE))
    blocks.append(('subtitle', '完整构造文件说明 · 非刚性反例 · 可精确复核的数学证明'))
    blocks.append(('subtitle', '2026 年 10 月 16 日'))
    heading('1  结论与适用范围')
    paragraph('本文只讨论项目中已确认的那一个 5×5 初始空间拼接：顶点坐标见第 4 节，25 个空间四边形的连接关系固定，每个四边形整体保持刚性，相邻面沿整条公共边连接。')
    paragraph('定理：在上述条件下，从指定初态出发不存在非平凡的连续刚性折叠。整体的平移、转动不算折叠。结论不依赖 F 的原圆，也不依赖具体驱动点、转动速度或动画算法。')
    paragraph('这不是“任意非平面 5×5 都不可折叠”的定理。改变初始坐标、面片朝向、连接关系，或者允许一个空间四边形内部的两个三角形相对转动，均属于另一个问题。本文也不排除与初态不能连续相连的其他离散拼装。')
    heading('2  交付的完整构造文件')
    paragraph('5x5 nonplanar folding full construction.ggb 是现有联动文件的完整代数区版本，不是新实现的刚性折叠。文件保留全部 203 条构造定义，其中原始 2×2 的 48 条定义逐条保留；本次没有修改点的位置关系、圆、交点或联动公式。')
    table(['对象', '数量', '代数区处理'], [
        ['点（含中间构造点）', '66', '全部显示定义和数值'],
        ['线段（含隐藏的面内边）', '247', '全部显示端点及定义'],
        ['多边形 / 三角面', '59', '保留原有及显示用面片'],
        ['圆 / 圆锥曲线', '8', '保留原圆及辅助圆'],
        ['平面；向量', '3；2', '全部保留'],
        ['合计', '385', '均不再作为隐藏的辅助对象'],
    ], [160, 65, 270])
    paragraph('左侧按构造顺序显示“定义与数值”。三维画面仍隐藏多数辅助点线圆，避免遮挡；可以在代数区逐个开启其可见性。原有 q1 = Polygon(A,B,C,D) 因四点不共面可能显示“?”：其定义仍被原样保留，实际空间面由两片同色三角形显示。不要把这个旧多边形的无定义数值误认为构造丢失。')
    paragraph('文件中的驱动点是 F = Point(e)。P = Intersect(r₁,t) 是依赖交点，不是独立驱动点。现有联动的变形问题仍保留，便于检查和复现。')

    page()
    heading('3  当前联动为什么不刚性')
    paragraph('外围 27 个网格点不是把整个刚性面统一作同一个等距变换而得到的，而是分别对不同的中点作点对称。单次点对称是等距变换，但不同顶点采用不同中心，不能推出拼成的新四边形保持全等。')
    paragraph('例如第一行第二列黄色面按顺序取顶点 W₀₁、W₀₂、D、O。文件中的中点与对称命令给出以下精确关系：')
    formula(r'W_{01}=O+F-K,\qquad W_{02}=D+A-G=D-G.')
    paragraph('初态 F = (−1,1,√2) 时，这个面的四边均长 2，短对角线 OW₀₂ 长 2。将 F 沿原圆移到以下位置，并取与初态连续对应的原交点分支，原始构造给出：')
    formula(r'F=(0,1,\sqrt{3}),\qquad G=(1,-3/2,\sqrt{3}/2),')
    formula(r'O=(-2\sqrt{6}/3,\,2,\,2\sqrt{3}/3),\qquad W_{02}=(-1,7/2,-\sqrt{3}/2).')
    paragraph('这些坐标直接代入原圆及平面交点条件即可核验。于是短对角线满足：')
    formula(r'|W_{02}-O|^2=(-1+2\sqrt{6}/3)^2+(3/2)^2+(-7\sqrt{3}/6)^2')
    formula(r'=10-\frac{4\sqrt{6}}{3}\ne 4,\qquad |W_{02}-O|\approx 2.594998.')
    paragraph('这已经是一个精确的非刚性反例，无须依赖图像观感。隐藏面内对角线不会消除这个点间距离的变化。')
    heading('原生 GeoGebra 的另一组复核')
    paragraph('以圆参数 θ 定义 F(θ) = (√3 cos θ,1,√3 sin θ)，初态 θ₀ = arccos(−1/√3)。正向 10° 指 θ₀ + 10°，不是任意投影视角下的顺时针。')
    table(['同一黄色面的长度', '初态', '圆参数增加 10°'], [
        ['边 W₀₁W₀₂', '2.000000000', '2.029394026'],
        ['长对角线 W₀₁D', '2.828427125', '3.026204354'],
        ['短对角线 W₀₂O', '2.000000000', '1.720618297'],
    ], [200, 140, 155])
    paragraph('对初态两侧 ±25° 的 101 个位置逐一读取 GeoGebra 坐标：原始四面的六个距离保持不变（浮点误差约 1.8×10⁻¹⁵）；新增 21 面均出现距离变化。该采样仅作为现有文件的诊断，不作为下面不可折叠定理的证明。')

    page()
    heading('4  精确模型与刚性约束')
    paragraph('全局顶点写作 p(r,c)，其中 r,c = 0,…,5；单元面为四点 p(r,c)、p(r,c+1)、p(r+1,c+1)、p(r+1,c)，其中 r,c = 0,…,4。相邻面共用同一顶点，因而不允许公共边的两端被复制成不同位置。')
    paragraph('为避免根式运算，使用缩放坐标 x(r,c)。物理坐标、内积和精确初态如下；所有矩阵运算都在有理数域进行。')
    formula(r'p_{rc}=Sx_{rc},\qquad S=\mathrm{diag}(1,1,\sqrt{2}),\qquad M=S^TS=\mathrm{diag}(1,1,2).')
    formula(r'\|x\|_M^2=x^TMx,\qquad a=(1,-3,1),\quad b=(3,-1,-1).')
    formula(r'x^0_{rc}=b_{\epsilon_r\epsilon_c}+\lfloor(r-1)/2\rfloor a+\lfloor(c-1)/2\rfloor b,')
    formula(r'\epsilon_r=(r-1)\;\mathrm{mod}\;2,\qquad \epsilon_c=(c-1)\;\mathrm{mod}\;2,')
    formula(r'b_{00}=(-2,2,0),\ b_{01}=(0,2,0),\ b_{10}=(-1,1,1),\ b_{11}=(0,0,0).')
    paragraph('这里的整数坐标与 5x5 nonplanar static.ggb 的根式定义逐点一致；不是从屏幕图像估计坐标，也不是把浮点测量误差当作精确几何。完整构造版在上述 F 初态计算得到同一拼接。')
    paragraph('每一个空间四边形整体刚性，要求四个顶点的全部六个点间距离恒定，即四条边和两条对角线均不变。对当前每个单元，六个距离平方的集合为 {4,4,4,4,4,8}，且四点不共面。')
    formula(r'(x_i(t)-x_j(t))^TM(x_i(t)-x_j(t))=(x_i^0-x_j^0)^TM(x_i^0-x_j^0).')
    paragraph('上式对每个单元内的每一对顶点成立。把这些约束记作“杆”只是数学表示，不是在 GGB 中添加可见线段，也不允许面内出现新折痕。25 个单元合计形成 110 个不同的点对约束。')
    paragraph('只要证明这些必要距离约束已禁止非平凡连续运动，就足以禁止真实刚性面片折叠；是否还要求无自交，并不影响这个否定结论。')

    page()
    heading('5  九面子结构与参考面的固定')
    paragraph('先选取全局 r,c = 1,…,4 的 16 个顶点，即第 2 至第 4 行、列的九个单元。将这 16 个顶点按行优先编号为 1,…,16；下表给出缩放坐标。实际高度为表中第三坐标乘 √2。')
    rows = []
    for index, (position, name) in enumerate(zip(report['patch_coordinates'], report['patch_original_names']), 1):
        rows.append([str(index), name, '(' + ', '.join(map(str, position)) + ')'])
    table(['编号', '完整构造版名称', '缩放坐标'], rows, [60, 165, 270])
    paragraph('固定参考面 D,C,A,B，即编号 2、3、6、7 的四个顶点。该面自身刚性且四点不共面，所以任意连续刚性运动都可通过一个随时间变化的整体刚体变换，把这四点固定在初态位置。此操作只去掉整体平移、转动，不额外限制折叠。')
    paragraph('固定后余下 12 点、36 个坐标自由量。该九面子结构共有 42 个不同的点对约束，详列于附录。接下来证明固定参考面后，其初态是距离方程解集中的孤立点。')

    page()
    heading('6  秩与唯一无穷小方向的精确证书')
    paragraph('记固定后的位移列向量为 h∈R³⁶；其分量按顶点编号递增、每点 x,y,z 顺序排列，跳过固定点 2、3、6、7。对点对 i<j，定义矩阵 R 的对应行为以下线性型，固定点的位移视为零。')
    formula(r'(Rh)_{ij}=(x_i^0-x_j^0)^TM(h_i-h_j),\qquad R\in\mathbb{Q}^{42\times36}.')
    paragraph('行按点对 (i,j) 的字典序排列。取下表 v 为完整的 16 点缩放速度；固定点的零向量删除后得到 v̄∈R³⁶。直接代入给出 Rv̄ = 0。')
    velocity_rows = []
    for first in range(8):
        second = first + 8
        velocity_rows.append([str(first + 1), str(tuple(report['velocity'][first])),
                              str(second + 1), str(tuple(report['velocity'][second]))])
    table(['点号', 'vᵢ', '点号', 'vᵢ'], velocity_rows, [55, 192, 55, 193])
    paragraph('为使秩的结论能够完全复核，给出一个显式非零子式：从 R 删除第 4、6、7、9、10、17、41 行，再删除第 36 列，得到 35×35 矩阵 R*。整数消元的精确结果是：')
    formula(r'\det R_*=-68719476736=-2^{36}\ne0.')
    paragraph('故 rank R≥35；另一方面 v̄≠0 且 Rv̄=0，故 rank R≤35。因此：')
    formula(r'\mathrm{rank}\,R=35,\qquad\ker R=\mathrm{span}\{\bar v\}.')
    paragraph('这一步不能单独推出存在可折叠运动。“存在一个无穷小方向”只说明距离约束的一阶项可以抵消；接下来检查这个方向能否对应真正相邻构型。')

    page()
    heading('7  二次项证书')
    paragraph('按下表给点对赋权 ωᵢⱼ，其余点对权重取零。这些正负整数是代数检验权重，不表示必须在真实折纸上施加外力。')
    stress_rows = []
    for index in range(9):
        first = report['stress'][index]
        second = report['stress'][index + 9]
        stress_rows.append([f'({first[0]},{first[1]})', str(first[2]),
                            f'({second[0]},{second[1]})', str(second[2])])
    table(['点对', '权重', '点对', '权重'], stress_rows, [150, 97, 150, 98])
    paragraph('逐列代入第 6 节矩阵，得到以下精确恒等式。这里 ω 对固定后的矩阵成立；固定点上的反作用无需另作约束。')
    formula(r'\omega^T R=0.')
    paragraph('对任意位移 h 定义二次型 Q，仍令四个固定点的位移为零。把第 6 节的整数速度逐项代入，得到：')
    formula(r'Q(h)=\sum_{(i,j)}\omega_{ij}(h_i-h_j)^TM(h_i-h_j),')
    formula(r'Q(\bar v)=32,\qquad Q(t\bar v)=32t^2.')
    paragraph('Q 未必在整个 36 维空间半正定；本证明只需它在 ker R 的所有非零向量上严格为正。以上一维核与整数值 32 已经充分。不能把 Q 的这一性质错误推广成全空间正定。')
    paragraph('附带的核验程序不使用浮点 SVD 判秩：R、v̄、ω、非零子式和数值 32 全部采用 SymPy 有理数 / 整数运算复核。')

    page()
    heading('8  严格证明九面子结构局部锁死')
    paragraph('完整距离方程在初态附近具有精确的二次展开，而非截断近似。对每个点对，令 qᵢⱼ(h) 为位移差的 M-范数平方，则：')
    formula(r'2Rh+q(h)=0,\qquad q_{ij}(h)=(h_i-h_j)^TM(h_i-h_j).')
    paragraph('左乘 ωᵀ，由 ωᵀR=0 可知，任意满足全部距离方程的位移都必须满足 Q(h)=0。')
    paragraph('反证：若固定参考面后，初态不是解集中的孤立点，则存在非零精确解 h⁽ⁿ⁾→0。取 εₙ=‖h⁽ⁿ⁾‖₂，并令 u⁽ⁿ⁾=h⁽ⁿ⁾/εₙ。单位球紧致，故可取子列使 u⁽ⁿ⁾→u 且 ‖u‖₂=1。')
    formula(r'2Ru^{(n)}+\varepsilon_n q(u^{(n)})=0,\qquad Q(u^{(n)})=0.')
    paragraph('令 n→∞，连续性给出 Ru=0、Q(u)=0。由 ker R=span{v̄}，有 u=t v̄；由于 ‖u‖₂=1，必有 t≠0。然而：')
    formula(r'0=Q(u)=Q(t\bar v)=32t^2>0,')
    paragraph('矛盾。因此存在一个初态邻域，在该邻域内固定参考面的距离方程只有初态这一解。九面子结构局部刚性，结论得证。')
    heading('为什么这已排除“换速度、换参数再折”')
    paragraph('上述证明直接讨论相邻构型的存在性，没有假设运动可微、初速度非零或二阶导数存在。因此，初速度取零、极慢启动、换参数、改变驱动点，都不能绕过结论。固定参考面后不存在任何可供连续运动离开的相邻解。')
    paragraph('这也解释了一阶线性化为何会误导：Rv̄=0 的确成立，但二次距离项在该方向上无法同时抵消。本文采用的是精确解集的孤立性证明，不是把一阶“看起来能动”当作可折叠性。')
    paragraph('该论证是应力 / 二阶刚性思想在本实例中的直接实现。Connelly-Whiteley [1] 的 Corollary 5.2.2 给出杆框架二阶应力检验，Theorem 4.3.1 给出二阶刚性蕴含刚性。这里已给出独立的邻域证明，不需仅凭文献定理或数值优化失败作结论。')

    page()
    heading('9  从九个面推广到完整 5×5')
    paragraph('九面子结构固定后，全局 r,c=1,…,4 的 16 个顶点固定。下面按表格顺序固定其余 20 点。每个目标点到所列三个中心的距离，均是某个刚性面中的边长或对角线长度；三中心在初态不共线，并在本步骤开始前已经固定。')
    extension_rows = []
    for index, (target, centers) in enumerate(report['extension_steps'], 1):
        extension_rows.append([str(index), str(tuple(target)), '；'.join(str(tuple(center)) for center in centers)])
    table(['步骤', '目标 (r,c)', '三个已固定中心的 (r,c)'], extension_rows, [45, 110, 340])
    paragraph('引理：到三个固定且不共线中心的距离均固定时，点只能处于至多两个位置。证明：三条球面方程两两相减，得到法向量线性独立的两张平面；其交集是一条直线，该直线与第一个球面至多相交两点。故原位置在解集中孤立，连续运动不能跳到另一交点。')
    paragraph('逐项应用该引理，36 个顶点全部局部固定。每一步的三中心非共线性、点对确属某个单元的六距离约束，均在核验程序中精确检查。因此整个 5×5 在去掉整体刚体运动后局部孤立。由连续性，不存在从该初态出发的非平凡连续刚性折叠。证毕。')

    page()
    heading('附录 A  矩阵的完整行顺序')
    paragraph('以下为第 6 节 42 个点对的字典序。每行对应 (xᵢ⁰−xⱼ⁰)ᵀM(hᵢ−hⱼ)。结合坐标表、固定点集合和列顺序，R 的每一个整数元素均已唯一确定；无需另行下载一个大矩阵。')
    pair_rows = []
    pairs = report['patch_pairs_one_based']
    for index in range(14):
        row = []
        for group in range(3):
            position = index + 14 * group
            row.extend([str(position + 1), str(tuple(pairs[position]))])
        pair_rows.append(row)
    table(['行号', '点对', '行号', '点对', '行号', '点对'], pair_rows, [42, 123, 42, 123, 42, 123])
    heading('附录 B  核验与文件边界')
    paragraph('在项目根目录运行 python proofs/verify_nonplanar_rigidity.py，可从静态 GGB 的精确根式定义重建证书。加上 --export-full-construction 可重新导出完整代数区版本；不会覆盖原始 2×2、静态参照或原联动文件。核验结果保存在 proofs/5x5 nonplanar rigidity certificate.json。')
    paragraph('核验项目：36 个精确初始坐标、25 面的六距离、42×36 的矩阵、−2³⁶ 子式、Rv̄=0、ωᵀR=0、Q(v̄)=32，以及 20 步三球交点传播。数学证明本身依赖这些公开的有限代数恒等式，不依赖动画播放结果。')
    paragraph('证书针对静态参照文件的 SHA-256 为：')
    blocks.append(('small', report['static_sha256']))
    heading('参考文献')
    paragraph('[1] Robert Connelly and Walter Whiteley. Second-Order Rigidity and Prestress Stability for Tensegrity Frameworks. SIAM Journal on Discrete Mathematics, 9(3), 453-491, 1996. DOI: 10.1137/S0895480192229236. 本文相关条目：Theorem 4.3.1（印刷页 474）及 Corollary 5.2.2（印刷页 479）。')
    blocks.append(('link', {'text': '作者提供的论文 PDF（Cornell University）', 'url': REFERENCE_URL}))
    paragraph('说明：文献提供一般刚性理论；本文的特定坐标、矩阵、整数权重、子式和 5×5 推广由本项目计算并逐项验证。文献本身没有直接讨论这个折纸模型，也不能作为“所有非平面 5×5 都不可折叠”的依据。')
    return blocks


def latex_escape(text):
    replacements = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
                    '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}'}
    superscripts = dict(zip('⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿᵀ', '0123456789+-=()nT'))
    subscripts = dict(zip('₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ᵢⱼₙ', '0123456789+-=()ijn'))
    symbols = {'×': r'\times', '±': r'\pm', 'ε': r'\varepsilon', 'θ': r'\theta',
               'ω': r'\omega', '‖': r'\Vert', '→': r'\to', '∈': r'\in',
               '∞': r'\infty', '≠': r'\ne', '≤': r'\le', '≥': r'\ge', '°': r'{}^\circ'}
    tokens = re.findall(r'v̄|√\d+|[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿᵀ]+|[₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ᵢⱼₙ]+|.', str(text))
    result = []
    for token in tokens:
        if token == 'v̄':
            result.append(r'\(\bar v\)')
        elif token.startswith('√'):
            result.append(r'\(\sqrt{' + token[1:] + r'}\)')
        elif token[0] in superscripts:
            result.append(r'\textsuperscript{' + ''.join(superscripts[character] for character in token) + '}')
        elif token[0] in subscripts:
            result.append(r'\textsubscript{' + ''.join(subscripts[character] for character in token) + '}')
        elif token in symbols:
            result.append(r'\(' + symbols[token] + r'\)')
        else:
            result.append(replacements.get(token, '-' if token == '−' else token))
    return ''.join(result)


def write_latex(blocks):
    document = [r'\documentclass[UTF8,a4paper,11pt]{ctexart}',
                r'\usepackage[margin=22mm]{geometry}', r'\usepackage{amsmath,amssymb,booktabs,longtable,hyperref}',
                r'\setlength{\parindent}{0pt}', r'\setlength{\parskip}{6pt}', r'\begin{document}']
    for kind, value in blocks:
        if kind == 'title':
            document.append(r'\begin{center}\LARGE\bfseries ' + latex_escape(value) + r'\end{center}')
        elif kind == 'subtitle':
            document.append(r'\begin{center}' + latex_escape(value) + r'\end{center}')
        elif kind == 'heading':
            document.append(r'\section*{' + latex_escape(value) + '}')
        elif kind == 'formula':
            document.append('\n\\[\n' + value + '\n\\]\n')
        elif kind == 'table':
            document.append(r'\begin{longtable}{' + 'l' * len(value['headers']) + '}\toprule')
            document.append(' & '.join(map(latex_escape, value['headers'])) + r' \\ \midrule')
            document.extend(' & '.join(map(latex_escape, row)) + r' \\' for row in value['rows'])
            document.append(r'\bottomrule\end{longtable}')
        elif kind == 'link':
            document.append(r'\href{' + value['url'] + '}{' + latex_escape(value['text']) + '}')
        elif kind == 'page':
            document.append(r'\newpage')
        else:
            document.append(latex_escape(value) + '\n')
    document.append(r'\end{document}')
    (OUTPUT / (STEM + '.tex')).write_text('\n'.join(document) + '\n', encoding='utf-8')


def prepare(blocks):
    from matplotlib.mathtext import math_to_image
    from matplotlib.font_manager import FontProperties, findfont
    OUTPUT.mkdir(parents=True, exist_ok=True)
    TEMPORARY.mkdir(parents=True, exist_ok=True)
    for index, (kind, value) in enumerate(blocks):
        if kind == 'formula':
            math_to_image('$' + value + '$', TEMPORARY / f'formula-{index}.png',
                          prop=FontProperties(size=12), dpi=250, format='png', color='#152d43')
    write_latex(blocks)
    (TEMPORARY / 'content.json').write_text(json.dumps(blocks, ensure_ascii=False, indent=2), encoding='utf-8')
    (TEMPORARY / 'fonts.json').write_text(json.dumps({'math': findfont('DejaVu Serif')}), encoding='utf-8')
    print('Prepared formulas and editable XeLaTeX:', OUTPUT / (STEM + '.tex'))


def render(blocks):
    from PIL import Image as PillowImage
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether

    pdfmetrics.registerFont(TTFont('Song', 'C:/Windows/Fonts/simsun.ttc', subfontIndex=0))
    pdfmetrics.registerFont(TTFont('Hei', 'C:/Windows/Fonts/simhei.ttf'))
    font_paths = json.loads((TEMPORARY / 'fonts.json').read_text(encoding='utf-8'))
    pdfmetrics.registerFont(TTFont('MathText', font_paths['math']))

    def rich_text(value):
        return ''.join('<font name="MathText">' + escape(part) + '</font>'
                       if ord(part[0]) < 0x3000 else escape(part)
                       for part in re.findall(r'[\x00-\u2fff]+|[^\x00-\u2fff]+', str(value)))

    body = ParagraphStyle('Body', fontName='Song', fontSize=10, leading=16,
                          spaceAfter=7, wordWrap='CJK', alignment=TA_JUSTIFY, textColor=colors.HexColor('#182f43'))
    heading = ParagraphStyle('Heading', parent=body, fontName='Hei', fontSize=14, leading=19, spaceBefore=6, spaceAfter=10)
    title = ParagraphStyle('Title', parent=heading, fontSize=21, leading=29, alignment=TA_CENTER, spaceBefore=8, spaceAfter=10)
    subtitle = ParagraphStyle('Subtitle', parent=body, fontSize=10, alignment=TA_CENTER, textColor=colors.HexColor('#5b6f82'))
    small = ParagraphStyle('Small', parent=body, fontSize=8, leading=11)
    cell = ParagraphStyle('Cell', parent=body, fontSize=9, leading=12, spaceAfter=0, alignment=0)
    story = []
    page_width, page_height = A4
    available_width = page_width - 92
    for index, (kind, value) in enumerate(blocks):
        if kind == 'page':
            story.append(PageBreak())
        elif kind == 'formula':
            path = TEMPORARY / f'formula-{index}.png'
            with PillowImage.open(path) as formula_image:
                width, height = formula_image.size
            width, height = width * 72 / 250, height * 72 / 250
            factor = min(1, available_width / width)
            story.append(KeepTogether([Spacer(1, 3), Image(str(path), width=width * factor, height=height * factor), Spacer(1, 9)]))
        elif kind == 'table':
            rows = [value['headers'], *value['rows']]
            table_rows = [[Paragraph(rich_text(item), cell) for item in row] for row in rows]
            widths = value['widths']
            if widths:
                widths = [width * available_width / sum(widths) for width in widths]
            result = Table(table_rows, colWidths=widths, repeatRows=1, hAlign='LEFT')
            result.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e7eff6')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f6f8fb')]),
                ('LINEBELOW', (0, 0), (-1, 0), 0.6, colors.HexColor('#859aae')),
                ('LINEBELOW', (0, -1), (-1, -1), 0.5, colors.HexColor('#b7c6d3')),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('LEFTPADDING', (0, 0), (-1, -1), 7), ('RIGHTPADDING', (0, 0), (-1, -1), 7),
                ('TOPPADDING', (0, 0), (-1, -1), 4), ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            story.extend([result, Spacer(1, 9)])
        elif kind == 'link':
            story.append(Paragraph('<link href="' + escape(value['url'], quote=True) + '" color="#195b8e">' + rich_text(value['text']) + '</link>', body))
        else:
            style = {'title': title, 'subtitle': subtitle, 'heading': heading, 'small': small}.get(kind, body)
            story.append(Paragraph(rich_text(value), style))

    def decorate(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor('#d3dee8'))
        canvas.line(46, page_height - 35, page_width - 46, page_height - 35)
        canvas.setFillColor(colors.HexColor('#637b90'))
        canvas.setFont('Song', 8)
        canvas.drawString(46, page_height - 27, '5×5 非平面拼接 / 严格刚性核查')
        canvas.drawRightString(page_width - 46, 26, str(document.page))
        canvas.drawString(46, 26, '仅针对本文指定初态与整边连接；不推广到所有 5×5 模型')
        canvas.restoreState()

    path = OUTPUT / (STEM + '.pdf')
    document = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=46, rightMargin=46,
                                 topMargin=48, bottomMargin=44, title=TITLE,
                                 author='Clipping-papers', subject='Exact local rigidity certificate for the specified nonplanar 5x5 assembly')
    document.build(story, onFirstPage=decorate, onLaterPages=decorate)
    print(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render', action='store_true')
    arguments = parser.parse_args()
    if arguments.render:
        render(json.loads((TEMPORARY / 'content.json').read_text(encoding='utf-8')))
    else:
        prepare(content(json.loads(CERTIFICATE.read_text(encoding='utf-8'))))


if __name__ == '__main__':
    main()
