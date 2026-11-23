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

求解，每个小节给出一条标准路线、若干备选路线（各带预计时间和容错率）和留下的断点，写成一份路线清单；`--png` 另外画出俯视图：

```bash
python -m ballistics solve --mesh data/Level_01-mesh.npz --map data/Level_01-map.json --sector all --png out
```

只看粗层的规划线（不推演、不细搜，很快，适合先看一张地图的全貌）：

```bash
python -m ballistics coarse --mesh data/Level_01-mesh.npz --map data/Level_01-map.json --png out
```

只看连通：把粗层分成只靠滚动就互相到得了的区域，列出区域之间靠什么相连，找跨断点最少的走法（不推演，见 [docs/solver.md](docs/solver.md)）：

```bash
python -m ballistics regions --mesh data/Level_02-mesh.npz --map data/Level_02-map.json --out data --png out
```

把解出的路线写成路线文件（`L01.npz`，一条参考线；加 `--mesh` 还写出分段的版本 2）：

```bash
python -m ballistics route --map data/Level_01-map.json --results data --level 1 --out data/routes
```

查球、机关和关卡规则的目录（只是数据，求解器还不读它）：

```bash
python -m ballistics things              # 一览
python -m ballistics things P_Modul_37   # 一条
```

可选：拿一条录下来的引擎轨迹，量物理核和引擎差多少：

```bash
python -m ballistics calibrate --mesh data/Level_01-mesh.npz --run data/runs/<轨迹>.npz
```

## 现状

第 1 关**在物理核里走通了**：四个小节都有路线，共 150.5 游戏秒，没有优化。第 2 关第 1、2、5 小节走通，第 3、4 小节还留着断点（第 3 小节要软木桥、跷跷板和几步机关，见 [docs/mechanisms.md](docs/mechanisms.md)）。这些都是手写物理核里的轨迹，没有在游戏里跑过。详见 [docs/level-1.md](docs/level-1.md) 和 [docs/mechanisms.md](docs/mechanisms.md)。

已有：

- 读 `.nmo`，导出碰撞网格、锚点和摆放的东西（机关、箱子、道具球按游戏模板的真实形状）
- 精确几何（球对三角面的距离和接触）和钢轨提取
- 物理核：三种球对静态几何，会动的刚体（道具球、箱子、栅栏的滑块、绕铰链转的推板和跷跷板），风扇的推力，变球器；和引擎轨迹对比的误差
- 求解器：粗层按球种和机关的简单状态规划通路（为了推栅栏先去变石球、先推倒推板再过去），沿通路推演，只在过不去的地方细搜
- 已经接进求解的：分数球和生命球、风扇、变球器、Dome、道具球、箱子、落石、活式木栅栏、推板、纸球、跷跷板（状态表在 [docs/mechanisms.md](docs/mechanisms.md)）
- 备选路线：找出粗层没有边通上去的落点（墙顶、护栏），从标准路线上搜过去；每条路线给出预计时间和容错率
- 输出：路线清单、俯视图、参考线（`L<nn>.npz`，版本 1）和分段的路线文件（`L<nn>.segments.json`，版本 2：每段的球种、变球点、粗略的出口和走廊、机关标注，格式见 [docs/files.md](docs/files.md)）

没有：

- 其余 9 种机关（升降机、秋千、单摆、软木桥等）
- 纸球的真实形状（物理核里是球体替身）
- 轨迹的优化（现在的轨迹远慢于高手录像）；备选路线只有“经过孤岛”一种，收益只算时间
- 证明格式和独立检查器；路线文件里的出口和走廊只是扰动重放的包络
- 第 3 关以后的关卡没有认真解过

## 文档

| 文件 | 内容 |
|---|---|
| [docs/physics.md](docs/physics.md) | 物理核的规则、每条规则的依据、和引擎对比的误差 |
| [docs/solver.md](docs/solver.md) | 求解器怎么工作，有哪些界限和不足 |
| [docs/level-1.md](docs/level-1.md) | 第 1 关的结果 |
| [docs/files.md](docs/files.md) | 输入的地图文件、导出的文件和路线文件的格式 |
| [docs/things.md](docs/things.md) | 目录：三种球、每种机关和关卡规则各一条（刚体和数值、状态机、三种球的结论、对路线的影响），每个事实标了依据 |
| [docs/mechanisms.md](docs/mechanisms.md) | 目录里已经接进求解的机关和物体：模拟了什么、依据、和原版记录的对照、还没做的 |
| [docs/design.md](docs/design.md) | 思路：每个决定的理由、放弃过的做法、还没解决的问题 |
| [docs/route-generator-plan.md](docs/route-generator-plan.md) | 计划：作为路线生成器还缺什么（分段的路线文件、覆盖、为机关规划、多样性、反馈通道） |

关于游戏本身（和这个工具无关的事实，每条标了依据）：

| 文件 | 内容 |
|---|---|
| [docs/game/reading-game-data.md](docs/game/reading-game-data.md) | 游戏数据在哪、怎么读：地图文件、脚本、数据表、引擎轨迹 |
| [docs/game/scripts.md](docs/game/scripts.md) | 有哪些脚本和数据表，各管什么 |
| [docs/game/physics.md](docs/game/physics.md) | 引擎的物理：时间、按键、三种球、极速、坡和台阶、接触 |
| [docs/game/rules.md](docs/game/rules.md) | 关卡规则：小节、盘点、死亡复位、变球器、成绩 |
| [docs/game/mechanisms.md](docs/game/mechanisms.md) | 各种机关的参数和行为 |
| [docs/game/levels.md](docs/game/levels.md) | 每关的机关，第 1 关的细节和已知走法 |
| [docs/game/terminology.md](docs/game/terminology.md) | 术语 |

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
