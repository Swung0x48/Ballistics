# Ballistics

Ballance 地图的静态分析器。

输入一张地图的 `.nmo` 文件，输出值得尝试的球轨迹（路线）。它不启动游戏：路线是在一套手写的物理规则里算出来的候选，拿到游戏里跑过才算数。

目标是每条轨迹都附带一份证明，说明按已知的物理规则它确实可行。证明的格式和检查器还没有做。

## 设计

- **手写的是物理，不是路线。** 球怎么受力、怎么滚、怎么碰，写成规则；路线由搜索在这套规则里找出来，不靠逐关补规则。
- **几何从粗到细。** 粗层只用来看全局结构和提候选；细层是引擎实际碰撞用的三角面本身。只有细层的计算能写进证明。
- **每条规则都有依据。** 规则和常数分三种：在引擎数据上量过的、从游戏自己的表里读的、假定的。只靠假定的东西不能进证明。
- **过不去就如实留着。** 搜索没找到走法的地方是断点，原样列出来，不拿放宽条件的结果充数。
- **不依赖引擎。** 这个项目不启动、不驱动游戏引擎，代码里也不引用它。它只读文件：地图文件，和别处录下来的引擎轨迹（用来量物理规则的误差）。它的输出也是文件；拿路线去引擎里实验、复核，是引擎那一侧工具的事。

## 用法

需要 Python 3.11 以上，以及 numpy、numba。读 `.nmo` 另外需要 libcmo21 的 `pybmap`，怎么找到它见 [docs/files.md](docs/files.md)。

导出地图的碰撞网格和锚点（出生点、盘点、飞船、摆放的东西）：

```bash
python -m ballistics export <地图.nmo> --out data --templates <游戏的 3D Entities/PH 目录>
```

求解，每个小节给出轨迹和留下的断点，`--png` 另外画出俯视图：

```bash
python -m ballistics solve --mesh data/Level_01-mesh.npz --map data/Level_01-map.json --sector all --png out
```

把解出的路线写成路线文件（`L01.npz`，一条参考线）：

```bash
python -m ballistics route --map data/Level_01-map.json --results data --level 1 --out data/routes
```

可选：拿一条录下来的引擎轨迹，量物理核和引擎差多少：

```bash
python -m ballistics calibrate --mesh data/Level_01-mesh.npz --run data/runs/<轨迹>.npz
```

## 现状

第一个验收目标是第 1 关：只靠球和静态网格，从出生点到飞船。**还没有达到**：第 1、2、4 小节能走通，第 3 小节有一处断点（一排道具球挡路，推道具球还没有建模）。详见 [docs/level-1.md](docs/level-1.md)。

已有：

- 读 `.nmo`，导出碰撞网格、锚点和摆放的东西（机关、箱子、道具球按游戏模板的真实形状）
- 精确几何（球对三角面的距离和接触）和钢轨提取
- 球对静态几何的物理核，以及它和一条引擎轨迹对比的误差
- 求解器：粗层规划通路，沿通路推演，只在过不去的地方细搜
- 路线文件和俯视图

没有：

- 会动的东西（推箱子、推道具球、机关）
- 轨迹的优化（现在的轨迹远慢于高手录像）
- 证明格式和独立检查器
- 石球、纸球；第 1 关以外的关卡

## 文档

| 文件 | 内容 |
|---|---|
| [docs/physics.md](docs/physics.md) | 物理核的规则、每条规则的依据、和引擎对比的误差 |
| [docs/solver.md](docs/solver.md) | 求解器怎么工作，有哪些界限和不足 |
| [docs/level-1.md](docs/level-1.md) | 第 1 关的结果 |
| [docs/files.md](docs/files.md) | 输入的地图文件、导出的文件和路线文件的格式 |

## 本地数据

地图文件、导出的网格和引擎轨迹都来自游戏本身，不放进仓库。本地放在 `data/` 下（也可以用环境变量 `BALLISTICS_DATA` 指到别的目录）：

```
data/maps/Level_01.NMO                地图文件
data/Level_01-mesh.npz                导出的碰撞网格
data/Level_01-map.json                导出的锚点
data/runs/SR01_1_14_457_264.npz       录下来的引擎轨迹，用来量物理核的误差
data/traces/SR01_1_14_457_264.txt     同一条轨迹的逐帧文本，用来核对几何
data/engine/L01-mesh.npz              从运行中的游戏读出的三角面，只用来和导出结果比对
```

## 运行测试

另外需要 pytest。没有本地数据时，依赖真实数据的测试会跳过。

```bash
python -m pytest
```
