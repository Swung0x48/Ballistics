# 脚本和数据表

游戏的逻辑全在 Virtools 行为脚本和数据表里。这一页列出有哪些脚本、哪些表、和分析有关的各自管什么。怎么把它们读出来见 [reading-game-data.md](reading-game-data.md)；标注的含义也在那一页。

## 有哪些脚本

第 10 关进关后（第 691 tick）一共 163 个根脚本【读】。每关 120 到 172 个，1 到 13 关合计 1,932 个【读】。按所有者分：

| 所有者 | 根脚本 |
|---|---|
| 关卡本身（`Level`） | `Default Level`、`Event_handler`、`Level_Init`、`Loading_Manager`、`Loading_Screen`、`Delete_All_Loaded_NMO`、`Delete_Loaded_NMO`、`Delete_LoadID`、`Debug_Info` |
| 游戏逻辑（`All_Gameplay`） | `Gameplay_Init`、`Gameplay_Ingame`、`Gameplay_Events`、`Gameplay_SectorManager`、`Gameplay_Energy`、`Gameplay_Sky`、`Gameplay_Tutorial`、`Gameplay_Blitz`、`Gameplay_Refresh` |
| 关卡构建（`All_Levelinit`） | `Levelinit_Init`、`Levelinit_build` |
| 球（`All_Balls`） | `Balls_Init`、`Ball_Explosion_Wood/Stone/Paper`、`Ball_ResetPieces_Wood/Stone/Paper`、`Ball_CollSound_Pieces`、`Ball_PaperWind`、`Ball_LightningSphere`、`Ball_Shadow`，另有 `BallParticle_Frame script` |
| 变球动画 | `AnimTrafo_Init`、`AnimTrafo_MF Script` |
| 菜单（`All_Menu`） | `Menu_Init`、`Menu_Main`、`Menu_Start`、`Menu_Pause`、`Menu_Dead`、`Menu_End`、`Menu_Score`、`Menu_Highscore`、`Menu_HighscoreEntry`、`Menu_Options`、`Menu_Opt_Graphics/Keys/Sound`、`Menu_Credits`、`Menu_YesNo`、`Menu_Zone` |
| 声音（`All_Sound`） | `Sound_Init`、`Sound_Manager`、`Music_Init`、`Music_Manager` |
| 起始盘点 | `PS_FourFlames_Init`、`PS_FourFlames_MF Script`、四团火各一个脚本 |
| 盘点 | `PC_TwoFlames_Init`、`PC_TwoFlames_MF Script`（两份）、大火焰和两团小火焰的脚本 |
| 飞船 | `PE_Balloon_Init`、`PE_Balloon Script` |
| 分数球 | `P_Extra_Point_Init`、`P_Extra_Point_MF Script`（每个实例一份），以及小球和击中判定的脚本 |
| 生命球 | `P_Extra_Life_Init`、`P_Extra_Life_MF Script`，以及动画和粒子脚本 |
| 机关 | 每种一个 `P_Modul_XX_Init`，每个实例一份 `P_Modul_XX_MF Script`。第 10 关有 01、03、08、18、19、25、26、29、30、34、37、41 |

命名上的例外：25 的实例脚本叫 `P_Modul_25 script`，03 的叫 `P_Modul03_MF Script`。风扇另有 `P_Modul_18_Particle Script`（转子和粒子，只是外观）。

箱子、道具球、变球器、Dome 没有自己的脚本，由小节管理器直接处理（见下面的 `PH_Groups`）。

## 和分析有关的脚本管什么

### `Levelinit_build`：构建关卡【读】

- **物理化路面**（子图 Phys. Floors）：按表 `Physicalize_Floors` 把 `Phys_Floors`、`Phys_FloorRails`、`Phys_FloorStopper` 三个组的物体变成固定的凹面碰撞体。
- **建锚体 FixCube**：位于世界原点的固定方块，边长 4，摩擦 0.7，弹性 0.4，**碰撞关闭**。它是机关关节的公共锚点。
- **处理占位物体**（Process Placeholder）：
  1. 给 `Sector_0N` 组里的占位物体加上“Sector”属性，读进 `PH` 表的行，然后删掉占位物体。
  2. 数出“单个小节里该类型最多几个”，把模板复制到这个数量（对象池）。
  3. 把池里的对象依次填进 `PH` 行。**不同小节的行会复用同一个对象**，所以“一个实体就是一个机关”不成立。
  4. 副本的实体名带数字后缀（`P_Modul_41_MF001`），脚本名不变，网格共享。
- **加载时每一行都激活一次、再全部停用**：所有摆放物先被放一遍，再全部拆掉。之后只有当前小节的是活的。

### `Gameplay_Ingame`：进关后的主逻辑【读】

- **Init Ingame / Set Physics Globals**：把重力设成 (0, -20, 0)，物理时间因子设成 2。在此之前世界用的是物理引擎的默认值 (0, -9.81, 0)。
- **Init Ingame / activate Scripts**：开局激活第 1 小节。
- **Ball Navigation**：五个 Set Physics Force。四个由方向键驱动（按下创建、松开关闭），第五个力为 0，由一个恒为假的调试开关控制。死亡时关闭。
- **BallManager**：
  - 死亡判定：球和 `DepthTestCubes` 组做包围盒相交。
  - Deactivate Ball：死亡后让小节管理器拆掉并重建当前小节。
  - New Ball：恢复球的初始状态，放到当前重生点，等 3 秒，物理化。
- **Trafo Manager**：找最近的变球器，距离小于 4.3 就变球。不看小节，也不看是否可见。

### `Gameplay_Events`：盘点和小节的切换【读】

- **activate Sektor**：过盘点时，让小节管理器停用旧小节 S、激活 S+1。
- **activate next Checkpoint**：只放置下一个盘点。两个盘点对象轮流用。触发条件是离大火焰 6.5 以内。

### `Gameplay_SectorManager`：小节管理器【读】

每次执行的顺序：

1. 读“要停用的小节”和“要激活的小节”。
2. 把阶段标志置为假。
3. 停用：遍历 `PH` 里属于该小节的行。有脚本的类型，重新激活它的脚本（脚本读到阶段为假，走自己的拆除分支）；没有脚本的，去掉刚体并隐藏，或只隐藏。停用时不看“Set?”列。
4. 一条延迟 4 帧的连线。
5. 把阶段标志置为真。
6. 激活：遍历属于该小节的行，先检查“Set?”是否为真，为假就跳过。然后放到这一行的矩阵位置、显示；有脚本的激活脚本（脚本自己建刚体和关节），没有脚本的直接物理化。

激活时的物理化每次都是新建的：刚体在这一行的位置重新造出来。

调用它的地方：

| 何时 | 效果 |
|---|---|
| 关卡开始 | 第 1 小节拆掉再建 |
| 过盘点 | 拆掉 S，建 S+1 |
| 死亡 | 拆掉当前小节再建 |
| 重开关卡、Game Over（`Event_handler` 的 reset Level） | 拆掉当前小节，把 `PH` 每一行的“Set?”恢复为真，再从第 1 小节开始 |

### 机关自己的脚本【读】

- 每个 `P_Modul_XX_MF Script` 开头读阶段标志：为真就物理化、建关节；为假就关关节、去掉刚体、恢复初始状态。没有单独的“复位”消息，复位就是这两步各走一遍。
- 同一类型的所有实例参数相同（比过 01、08、17、19、25、26、29、30、34、37、41）。所以每种机关只有一套数值，见 [mechanisms.md](mechanisms.md)。
- 多数机关的刚体“初始冻结”，球靠近到 50（水平距离）以内时由一个邻近检测唤醒。

### `Event_handler`

- reset Level：见上表【读】。
- 暂停：把物理时间因子写成 0，所以暂停时物理不走【读】。

### 没有读过的

`Gameplay_Energy`、`Gameplay_Blitz`、`Gameplay_Refresh`、`Gameplay_Sky`、`Gameplay_Tutorial`、`PE_Balloon Script`、球的各个脚本，本项目都没有读过内容。只知道第 1 关的教程会暂停世界【测】。

## 数据表

| 表 | 内容 |
|---|---|
| `PH` | 每关一张，每个摆放物一行：小节、类型名、矩阵、对象、Set? |
| `PH_Groups` | 每种类型一行：池的大小（Amount）、激活方式、复位方式 |
| `CurrentLevel` | 当前关的状态：`[0][1]` 当前球种（ActiveBall）、`[0][4]` 阶段标志（Activation Phase?）、当前重生点 |
| `IngameParameter` | 第 0 行有：当前变球器、要激活的小节、要停用的小节 |
| `Checkpoints`、`ResetPoints` | 盘点和重生点，行号就是顺序 |
| `Physicalize_GameBall` | 三种玩家球的摩擦、弹性、质量、阻尼、推力，见 [physics.md](physics.md) |
| `Physicalize_Floors` | 三个路面组的摩擦、弹性、碰撞组 |
| `Physicalize_Convex` | 凸包类摆放物的参数：箱子、道具纸球、Dome |
| `Physicalize_Balls` | 球形摆放物的参数：道具木球、道具石球 |
| `AllLevel` | 每个关卡槽一行，定这一关的奖励分等 |

`PH_Groups` 里各类型的处理方式（第 11 关读出；其他关只知道类型列表相同）【读】：

| 类型 | 激活 | 复位 |
|---|---|---|
| 分数球、生命球、机关 01/03/08/17/18/19/25/26/29/30/34/37/41 | 运行自己的脚本 | 运行自己的脚本 |
| 三种变球器 | 放到位、显示，没有脚本 | 只隐藏 |
| 道具纸球、箱子、Dome | 按凸包物理化 | 去掉刚体并隐藏 |
| 道具石球、道具木球 | 按球形物理化 | 去掉刚体并隐藏 |

`Checkpoints` 表的对象列在两个盘点对象之间交替，不能当作盘点的身份用【读】。

## 物理行为模块（BB）的含义

以下按重写版源码的实现【读】。原版 2002 年的模块有没有差别，除注明的以外没有核对。

- **Physicalize**：把物体变成刚体。参数有固定与否、摩擦、弹性、质量、碰撞组、初始冻结、是否参与碰撞、线性和旋转阻尼、碰撞形状（若干凸包、球、凹面网格）。“质心偏移”只在不自动算质心时生效，是物体自身坐标系里的偏移。
- **碰撞组**：两个刚体的碰撞组**相同且不为空**时互不碰撞。空的组和谁都碰。区分大小写。
- **Set Physics Force**：创建时把方向按参考物体变换成世界方向，之后**固定不变**，归一化后乘以力的数值。每个物理步对刚体施加一次**冲量**，速度变化是 数值 ÷ 质量，不乘时间步长。所以“力”的数值其实是每个物理步的冲量。从创建一直作用到关闭。
- **Set Physics Hinge**：铰链。锚点是参考框架在创建时的世界位置，轴是参考框架的世界 Z 轴。限位只在“Limitations”为真时生效。
- **Set Physics Slider**：滑轨。轴由两个点定出，只允许沿轴平移。限位同样要开关为真才生效。
- **Set Physics Spring**：弹簧。两个锚点在创建时固定；有弹性常数、自然长度、阻尼，能推也能拉。
- **Set Physics Ball Joint**：球铰，把两个刚体在一个点上连起来。
- **Physics WakeUp**：唤醒“初始冻结”的刚体。
- **Delayer**：每帧把帧时间累加，累加到设定值时触发。按**游戏时间**计，500 毫秒名义上是 33 帧。
- **TT Scaleable Proximity**：按距离做邻近检测，可以只看水平距离，检测间隔随距离变（远了就隔很多帧才查一次）。
- **TT Restore IC**：把物体恢复到初始状态。机关的拆除分支用它，所以非当前小节的机关会回到世界原点（见 [rules.md](rules.md)）。

机关脚本里没有用到马达、冲量模块、浮力和连续接触【读】。
