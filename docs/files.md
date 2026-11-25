# 输入和输出的文件

约定：程序的输出（日志和 JSON）使用英文和 ASCII 字符，图里的文字保留中文；`things` 命令显示的是中文目录正文。

## 输入：地图文件

输入是一张 Ballance 地图的 `.nmo` 文件，原版关卡和自制图都一样。直接读文件，不启动游戏。

```bash
python -m ballistics export <地图.nmo> --out data
```

它写出两个文件：

- `<名字>-mesh.npz`：玩家球会碰到的固定物体的三角面（世界坐标）、每个三角面属于哪个物体、是不是钢轨，以及摩擦和弹性系数。
- `<名字>-map.json`：出生点、重生点、各盘点、飞船、小节数、每个小节里放了什么（机关、道具球、分数球等占位物体的类型和位置）、死亡区；带 `--templates` 导出时还有 `hulls`：机关刚体的碰撞凸包（模板里不显示的那些网格），每个是顶点、三角面和它所属物体的坐标矩阵，放在地图摆放的位置；还有 `stoppers`：Phys_FloorStopper 组里的物体（挡机关和道具、不挡玩家球），按连通块拆开，每块是顶点和三角面。

哪些东西会碰到球，按游戏自己的规则来：`Phys_Floors` 和 `Phys_FloorRails` 两个组里的物体，以各自的网格作为碰撞形状；`Phys_FloorStopper` 只挡机关和道具球，不挡玩家球，所以不算。游戏在每一关还自带一个固定在原点的边长 4 的方块，也一并加上。

地图文件里没有出生时的球种（游戏按关卡槽位决定），也没有机关的部件（它们来自游戏自带的模板）。导出时加上模板目录，机关、箱子、道具球的部件就按静止姿态放进网格并标成“摆放物”；分数球、生命球、变球器和风扇不放（前两种和风扇没有刚体，变球器按触发距离算）。占位物体本身即使被地图列在路面组里也不当路面（13 关里只有第 3 关的一个风扇占位物体是这样，原版记录里球从它中间穿过）。

### 读 nmo 需要的库

`.nmo` 的格式由 libcmo21 项目的 BMap 库解析，通过它的 Python 绑定 `pybmap` 调用。这两样不在本仓库里。查找顺序：`--bmap` 给的目录、环境变量 `BALLISTICS_BMAP`、作为 Blender 扩展安装的 Ballance Blender Plugin（bbp_ng，它自带这个库）。

### 验证

- **和游戏里的碰撞体一致**：第 1 关从 `Level_01.NMO` 读出的 21,763 个地图三角面，和运行中的游戏为固定物体持有的三角面逐个相同，坐标差为 0。出生点和三个盘点的位置也相同。
- **和游戏里球的运动一致**：用一条完整的第 1 关引擎轨迹（20,483 帧）逐帧核对球心到三角面的距离，球从不进入三角面，全程最小间隙 0.0099；贴地滚动时间隙的中位数是 0.0141，这是引擎自己的接触余量。

## 输出：粗层的规划线

`coarse` 写出 `<名字>-coarse-plan.json`：每个小节一条只有位置的线（`points`，球心位置）、每一段是哪种边（`kinds`）、用到的起飞边和台阶、以及链接（粗层接不上的地方：位置、横向距离、高差、附近 9 个单位内摆放物的类型）。没有规划的小节写明原因。另有两项说明一条线不完整：`starts_off_reset_point`，这条线不是从盘点开始的，而是从离盘点这么远的地方开始，因为盘点附近即使加上链接也到不了终点；`short_of_goal`，这条线的终点离终点标记还有这么远（最后一个小节的路在飞船前就结束了）。

粗层缓存 `<名字>-coarse-solid.pkl` 里，`Graph.info["launches"][(u, v)]` 是助跑起点、落地点这对粗层节点的起飞边。一对节点可能有不同球种的代表，`variants` 列出按球种保留的边；点对上的顶层字段是代价最低的代表，状态图只用当前球种的代价与边。`launch_rule` 缓存标记变化时重建旧图。

| 字段 | 含义 |
|---|---|
| `ball` / `site` | 球种 / 崖边或轨端粗层节点（不是助跑起点） |
| `launch_position` / `launch_velocity` | 实际使用的起跳球心和带坡度的初速度 |
| `way` | 助跑、绕棱、闭式飞行和接触到粗层落地点的折线，世界坐标球心，间距目标为 1；供显示、路线导出和旧跟随器读取 |
| `runup` / `launch_speed` | 推算的水平助跑长度 / 起跳三维速度标量（每物理秒） |
| `lip_speed` / `rim_ticks` / `rail_support_prefix_ticks` | 唇部速度 / 绕棱 PSI 数 / 绕棱结束后按同轨仍受托的局部前缀 PSI 数；前缀不计作自由飞行 |
| `landing_position` / `landing_face` | 最早有效支撑接触时的球心 / 所碰网格三角面；缓存索引的 `v` 是就近映射的粗层位置 |
| `bounce_count` / `contacts` | 非支撑反弹次数 / 按顺序记录的固体接触（含最终支撑）：类型、三角面、物体、飞行相对 PSI、球心、法线、撞前速度；反弹另记撞后速度。双轨落地标为 `dual_rail`。 |
| `launch_rule` / `motion` | v10 与旧缓存分开。缓存里的 `motion` 保存助跑直段、闭式飞行、支撑和绕棱／侧翻弧，供机关及变球器守卫求交。风扇暂保存完整内核步内弦。`way` 不提供这些判定的几何。 |
| `rail_departures` | v8 沿轨串边的脱轨序列：方式、钢轨物体、位置、速度、轨上相对时刻、推力、瞄准粗层点和反解弧长区间；几何端点是零宽区间。缓存 `rail_departure_candidates` 保存映射成功的候选的紧凑行，测量脚本输出同内容的 CSV。 |

v13 另存 `terrain_ball_masks`（`(u,v,kind)` 的木／石／纸球位掩码，未列为三种都能走）；`terrain_uncertain` 单列实测区间内未定的坡／台阶；`terrain_support`、`terrain_lips` 对应共形支撑和实际台阶棱。`launch_runup_adj` 是继承的几何起飞提案，不是分球种滚动图。v15 的 `terrain_step_checks` 另列分球种的局部台阶候选、棱高、有限棱和理想入棱速度；续行的 `contact_margins` 记录入棱摩擦、承压和圆弧摩擦余量。新版 `launch_rule` 使旧图失效。

v12 的 `g.info["conformal"]` 与粗层图一起缓存：`patches` 保存平面法线、平面常数、原始面编号、公开边和 Shapely XZ 多边形；`face_patch`、`node_patch`、`node_patches` 保存面/落脚点索引（未归入共形路面的节点用 `-1`）；`regions` 按球种保存块 × 滚动区域的多边形和映射节点；`rails` 是 v11 真实轨心直段。重建时另外输出 `<名字>-conformal-candidates.csv`（经过包络核对的瞄准代表，未必飞得过）及 `<名字>-conformal-rail-candidates.csv`（已映射的钢轨续行候选，包括非最快代表）。旧图按 `launch_rule` 失效。

v14 的 `rail_itinerary` 保存按顺序实际承载的钢轨物体。`rail_departures` 中的 `joint` 是目标轨链和直段索引（没有瞄准钢轨时为空）；`source_piece` 是当前拟合直段索引，`clearance` 是原管可豁免的连续 PSI 前缀。`interval` 是对应释放区间沿原直段的弧长范围；链端固定释放时退化为一点。

`g.lanes` 的两个几何字段（v11）：`points` 是原粗层落脚点（用于配对和映射）；`segments` 是 `(N, 3)` 的球心路径折点，相邻两行是一段直线，供捕获、沿轨、接缝、助跑和反解。两个数组的长度不必相同。双轨如果没有由两支管共同形成的有效段，`segments` 可以为空。新 `launch_rule` 会使旧的求解图缓存重建。
| `segments` / `first_landing_node` | 有闭式续行的边记录首次飞行及后续 `rail`、`floor`、`rim`、`flight`、`release` 片段与首次落点节点；片段含物体、起止球心和速度、推力、PSI、游戏秒及结束原因。`landing_position` 指最终交回粗层的球心。命中同一节点对与球种的不同续行见证可共存于 `variants`。 |
| `across` / `down` | 自由飞行起点到接触球心的水平距离 / 带符号落差 |
| `keys_in_air` / `air_inputs` / `switch_tick` | 世界 X、Z 推力，以对应球种的单键推力为 1；两段代表另存两组向量和整数切换 PSI（从代表曲线起点计，即同轨受托前缀之前），`switch_tick=0` 表示恒定 |
| `flight_ticks` / `ticks` / `time` | 飞行 PSI 数 / 助跑、绕棱、同轨受托前缀与飞行的总 PSI 数 / 总游戏秒（`ticks / 132`），可含步内小数；当前搜索代价仍为 `ticks / 66 * FOLLOW_SPEED + STOP_COST` |

## 输出：区域和连接

`solve`（不带 `--roll`）分三步写出三份文件（怎么算的见 [solver.md](solver.md) 的“区域”），`sectors[]` 都是每个小节一项。

第一步，`<名字>-regions.json`：

| 内容 | 含义 |
|---|---|
| `entry_kind` | 这个小节从哪种球开始（前一个小节的走法以哪种球到达；各节的球种整关一起选，见 [solver.md](solver.md) 的“球种整关一起选”） |
| `mechanisms` | 这个小节粗层记状态的机关，按位的顺序（第二步的 `mechanism_bits` 的第 m 位是第 m 个：栅栏开了、推板倒了、跷跷板低在另一头） |
| `transformers` | 这个小节的变球器：名字、种类、位置 |
| `start_region`、`goal_regions` | 盘点所在的区域（按入口的球种）；目标所在的区域（每种球一个） |
| `regions` | 按球种（`wood`、`stone`、`paper`）分开，每个区域：名字（如 `W68`、`S54`、`P4`，首字母分别代表木球、石球、纸球）→ 点数、中心、范围（`lo`、`hi`） |

第二步，`<名字>-links.json`：

| 内容 | 含义 |
|---|---|
| `fewest_breaks` | 到目标最少要跨几个断点；到不了时为 `null` |
| `exit_kind` | 选中的走法以哪种球到达目标（下一个小节以它入场）；到不了时为 `null` |
| `breaks_by_entry_and_exit_kind` | 试过的每种入场球、每种到达球种各要几个断点，如 `{"wood": {"wood": 0, "paper": 0}}` |
| `way` | 断点最少、其次连接最少的走法：每一步的区域（名字带球种）、球种、机关位、怎么到的（`by`）和这条连接的说明（变球的连接写明进了哪种变球器的范围） |
| `coarse_plan` | 粗层自己的最短计划压成的区域步骤，对照用 |
| `links` | 区域之间的连接（不含断点）：从哪个区域到哪个、是什么、最小代价、有几条这样的边、一个例子的两端位置 |
| `breaks`、`breaks_all` | 30 个点以上的区域之间的断点式连接（`what` 以 `break (gap)` 或 `break (bridge)` 开头）；所有区域之间的个数 |

第三步，`<名字>-mechanisms.json`：

| 内容 | 含义 |
|---|---|
| `mechanisms[]` | 每个机关：名字、种类（`gate`、`board`、`seesaw`、`cork_bridge`、`falling_rock`、`t_board`、`flap`、`float_row`、`pendulum`、`elevator`、`swing`、`sandbag`、`box`）、状态（`states`）、开局的状态（`initial`）、每个状态放行的连接（`lets_through`：状态 → 从哪个区域到哪个、哪几种球）、在哪里由哪种球拨动（`switched`、`switch_note`；落石、T 型板和双向挡板的 `switched[].at` 是该区域里的操作处坐标列表）、固定在开局状态时到目标最少几个断点（`held_breaks`，到不了为 `null`）、结论（`need`：`needed` / `used, can be bypassed` / `not used by this way` / `goal not reached`） |
| `chain[]` | 第二步的走法里每次变球或拨动机关：第几步、做了什么（`did`）、靠哪条连接（`by`）、为了后面第几步的哪条连接（`for`，之后没用到时为 `null`） |
| `mechanisms[]` 中的 `flap` | 双向挡板：`pose A` / `pose B` 的通行连接；`switched[]` 另含 `from_state`、`to_state`、通过的 `lane`（A 或 B）和操作处 `at`；切换与通过是同一步 |
| `mechanisms[]` 中的 `float_row` | 一列箱式浮板：箱数（`boxes`）、成员摆放名（`placements`）、能否走完整列的球种上限（`passes`，不表示找到两端岸点）、侧面及末端离开明细（`exits[]`：入口区域 `from`、从入口数的箱号 `box`、离开方向 `direction`（`side -`、`side +`、`end`）、落点区域 `to`、球种 `balls`、入口和落点坐标 `at`）、走法是否用到（`on_way`）、整列拿掉后的最少断点数（`without_breaks`，同该列的 `held_breaks`）；`states` 为 `["no state bit"]`，列名用半角连字符，如 `P_Modul_41_01-08` |
| `mechanisms[]` 中的 `pendulum` | 单摆：`states` 为空；`crossings[]` 为有向推过连接（`from`、`to`、球种 `balls`、代表边的坐标 `at`、走法是否用这条边 `on_way`）；`blocked[]` 为限位挡掉的有向点对（木球区 `from`、`to`，同样限制其他球种，及 `at`）。`on_way` 表示该杆是否被用于推过或联动（`need` 相应为 `used by this way` 或 `not used by this way`），`without_breaks` 是拿掉杆后的最少断点数（到不了为 `null`）；另有 `stopper_deg`（无目录数据为 `null`）、`sweep_radius`、`hole_warning` 和 `candidates[]`（未接入的几何候选，摆放名 `name`、`type`、依据 `basis`（`ASSUMED`）和邻近的走法断点 `near_route_breaks`，不证明因果） |
| `mechanisms[]` 中的 `box` | 风扇上的一摞箱子，以排序后的第一个成员命名，`placements` 列全部成员，`fan` 是关联的风扇；开局 `present`，`cleared` 才放行对应的纸球上升边 |
| `mechanisms[]` 中的 `elevator` | 升降机状态为 `loaded` / `used`；`entries[]` 记门洞来路的 `regions` 和坐标 `at`。`exits[]` 按球种 `ball` 和出口样本列 `from` / `to` 区域、`blocks` 推掉几块、`walls` 挡墙门刚体名、世界高度 `platform_top` / `exit_top`、水平空档 `gap`、`flight` 是否需飞过、`with_speed` 是否需带速度上台阶、路唇或下一台平台边坐标 `exit_at`、图路点坐标 `at`、下一台摆放名 `target_elevator`（无则 `null`）及是否为选中走法的代表 `on_way`。机关总表的 `on_way` 表示是否用到这一台；`without_breaks` 是拿掉平台后最少断点数（到不了为 `null`）。 |
| `mechanisms[]` 中的 `swing` / `sandbag` | `states` 为空；`crossings[]` 含 `from`、`to`、`balls`、代表边坐标 `at`、样本数 `samples`、`on_way`；机关也有总的 `on_way`，以及 `without_breaks`（到不了为 `null`）、`switch_note`、`status`（`connected` / `fixed_bypass` / `no_matching_crossing`）、`safety_floor`（秋千是否有保护路面，沙袋为 `null`）。`timing` 共用 `period_game_seconds`、`windows_per_period`；秋千另有 `end_reach`、`amplitude_degrees`、`bank_window_steps_132`（空斗窗口，`lip_abs_x` 为模型岸沿，`minus_x` 跨周）和 `tas_on_bucket_game_seconds`；沙袋另有 `window_steps_132`（中心／包围盒）、`startup_closed_steps_132`、`first_window_after_death_steps_132`（从重建起算）、`impact_ball_speed`。未匹配沙袋只给横扫中线时序作参考。 |
| 单摆的 `linked[]`、推板和箱子的 `switched[]` 中的联动项 | 同一条原地操作：`pendulum` 为发起摆放名、`target` 为目标机关名、`effect` 为远程推倒或推开的说明；`from` / `to` 为同一球种区域，`balls` 为球种、`at` 为操作处坐标列表，`basis` 为 `OWNER`、`DERIVED`、`ASSUMED`，`on_way` 表示走法用到，`locked` 表示目标锁定时该操作不可用 |

状态字符串：栅栏 `closed` / `open`；推板和 T 型板 `up` / `down`；落石 `standing` / `dropped`；软木桥 `whole` / `broken`；双向挡板 `pose A` / `pose B`；箱子 `present` / `cleared`；跷跷板 `low end at (x, z)`（没有上板点的另一端用 `low end at the other end`）。`balls` 用英文球种，多个球种用逗号和空格分隔，如 `wood, stone`。说明中的依据标签为 `[READ]`、`[MEASURED]`、`[DERIVED]`、`[ASSUMED]`、`[WIKI]`、`[OWNER]`。

用了假设（`--without`、`--state`、`--lock`）时，三份文件名里多一个标签（`<名字>-<标签>-regions.json` 等），文件头的 `overrides` 记着用了哪些；不用时 `overrides` 为 `null`。

区域、连接和走法里的区域都用名字，名字的第一个字是球种：同一块路面，木球、石球、纸球的区域可以不一样（别种变球器的范围把它截开）。

## 输出：路线清单

`solve --roll` 在结果目录里写出 `<名字>-routes.json`：已经解过的每个小节的每一条路线，给别的工具读。它们是路线假说：手写物理核里的轨迹，没有在游戏里跑过。

| 内容 | 含义 |
|---|---|
| `pushes`、`push_names`、`ticks_per_game_second`、`ball` | 按键编号对应的推力方向（九种）、每游戏秒 132 步、球种 |
| `nudge` | 容错率是怎么量的（见 [solver.md](solver.md)） |
| `sectors[].entered` | 球是带着速度进入这个小节，还是静止起步 |
| `sectors[].routes[]` | 这个小节的路线，第一条 `R1` 是标准路线 |
| `routes[].reaches_goal`、`left_breaks`、`game_seconds` | 走没走通、留下的断点、预计时间（没走通时为空） |
| `routes[].tolerance` | 这条路线上细搜出来的走法有几处，以及每档扰动下最弱的一处 20 次里还能过几次 |
| `routes[].searched_passages` | 每一处细搜出来的走法各自的数 |
| `sectors[].ball`、`end_ball` | 球进入和离开这个小节时的球种（标准路线的） |
| `routes[].segments[]` | 重跑这条路线需要的东西：每一段的起始状态（位置、速度、自转）和按键（按键编号、步数）；留下的断点单独成一项；每一段带球种（`ball`），变球单独成一项（`type: transformation`：哪个变球器、从哪种变成哪种、占多少步、新球出现后的状态） |
| `routes[].extras` | 这条路线顺路吃到的分数球和生命球（规则和依据见 [mechanisms.md](mechanisms.md)）。`extras[]` 每个道具一行：名字、种类、所在小节、吃没吃到（`collected`）、在这个小节的第几步和第几游戏秒、吃到时和最近时的球距、是否贴边（`marginal`）、分数球的分（`points`，220：分数小球按过盘点前按 Esc 收齐算）；总的 `points` 是这条路线从分数球得的分，`lives` 是生命球的个数；`checkpoints` 是按规则判定过盘点的那一步；`basis` 是各个数在目录里标的依据。断点之后吃到的标 `after_left_break`，不计入 `points` 和 `lives` |
| `routes[].shortcut` | 备选路线才有：经过的孤岛、从标准路线的哪个状态离开、落在哪、上孤岛用了多久和试了多少次、这一段自己的容错率、在哪里回到标准路线以及那时领先多少、总共省多少 |
| `sectors[].islands_without_a_route` | 没给出路线的孤岛：没试的，和试了多少次没搜到的 |

每条路线逐步的球心位置在 `<名字>-staged-s<N>.pkl` 里（标准路线是结果本身，备选路线在它的 `alternatives` 下）。

## 输出：路线文件

```
python -m ballistics route --map data/Level_01-map.json --results data --level 1 --out data/routes
```

写出 `L01.npz`，一条“参考线”：

| 内容 | 含义 |
|---|---|
| `points` | 球心沿路线的位置，按弧长每 0.5 一个点 |
| `sector` | 每个点所在的小节（从 1 起），只增不减 |
| `t_ms` | 球经过每个点的游戏时间（毫秒），严格递增；读它的一方由此得到路线上的速度 |
| `meta` | 关号、出生点和各盘点的位置（读它的一方用来核对这条线属于哪一关）、点距、平滑参数、长度、各小节的弧长范围、来源说明 |

这就是 Ballrain 读参考线用的格式（它的 `ReferencePath` 资产，版本 1），所以两边都不需要知道对方：这里只写文件，那边按原样读。处理方式也和从录像造参考线时一样：位置先在时间上做 40 毫秒的高斯平滑（弹跳不增加弧长），再按弧长重采样。

线上放什么、不放什么：只放每个小节的标准路线，备选路线现在不写成参考线（各小节的备选路线怎么接成一整条线还没有定）。从关卡开头沿路线走到**第一个留下的断点**为止，就在那里结束。线不会跨过留下的断点，断点之后求解器按“从对面静止起步”列出来的部分也不在线上。`meta` 里的 `complete` 说明这条线是不是通到了头，`left_breaks` 列出留下的断点，`collected` 列出线上顺路吃到的分数球和生命球（名字、种类、所在小节、在那个小节的第几游戏秒、最近的球距、是否贴边、分数）。

## 输出：分段的路线文件（版本 2）

```
python -m ballistics route --map data/Level_02-map.json --results data --level 2 --out data/routes --mesh data/Level_02-mesh.npz --nmo data/maps/Level_02.NMO
```

给了 `--mesh`，除了 `L02.npz`，还在旁边写出 `L02.segments.json`：同一条路线，分成一段一段，每段写明球种、在线上的位置、出口、走廊和对机关做的事。版本 1 的线不变，只读它的一方照旧可以只读它。格式的草案在 [route-generator-plan.md](route-generator-plan.md)，Ballrain 那边看过一遍；这里是实际写出来的样子，和草案不同的地方列在这一节末尾。代码在 `ballistics/segments.py`。

文件头：

| 内容 | 含义 |
|---|---|
| `version` | 2 |
| `level`、`map` | 关号；地图文件的名字 |
| `line`、`line_sha256` | 同一次写出的 `L<nn>.npz` 和它的 sha256。读的一方用它核对这份文件和线是同一套（`segments.read` 也核对） |
| `map_sha256` | 地图文件（`.nmo`）的 sha256。用 `--nmo` 给出地图文件；不给时到地图 json 旁边的 `maps/` 和 `BALLISTICS_DATA/maps/` 下按名字找，找不到就是 `null` |
| `time`、`ticks_per_game_second` | 时间都是游戏时间的毫秒；物理步每游戏秒 132 步 |
| `coordinates` | 位置都是世界坐标里的球心，游戏单位 |
| `keys` | 按键名对应的推力方向（世界坐标）：`+x` 是朝 +x 推的那个键，`+x+z` 是两个键一起按 |
| `nudge` | 扰动重放怎么扰动（见下） |
| `routes[]` | 现在只有一条：`id` 为 `standard`，`complete` 和线的一样，`segments` 是它的各段 |

每段：

| 内容 | 含义 |
|---|---|
| `i`、`sector` | 第几段（从 0 起），在第几小节 |
| `kind` | `roll`：跟随器走的（连着的几段合成一段）；`flight`：从这一段里分出来的腾空（离开一切至少 40 步，起落之间水平至少 4）；`crossing`：细搜跨过的一段；`fan_rise`：乘风扇上升（按上升边自己的走法走的）；`operate`：推栅栏（按住一个键）；`transform`：变球；`break`：留下的断点 |
| `ball` | 这一段的球种：`wood`、`stone`、`paper`（断点段为 `null`） |
| `source` | 现在都是 `kernel`：物理核里的结果，没在游戏里跑过 |
| `s`、`t_ms` | 这一段在参考线上的弧长区间和游戏时间区间，和 `L<nn>.npz` 的 `points`（按弧长）、`t_ms` 对得上：第 k 步在线上的时间是 k × 1000/132 毫秒。一段的终点就是下一段的起点 |
| `own_s`、`own_t_ms` | 参考线停在第一个留下的断点，之后的段不在线上：那里 `s` 和 `t_ms` 为 `null`，改给这两项，从这一段自己的起点量起 |
| `ticks` | 这一段有多少物理步 |
| `start` | 这一段开始时球的位置和速度（`pos`、`vel`） |
| `assumed_rest_start` | 为真的段在第一个留下的断点之后（或在一个没走通的小节之后），是假定球在那里静止起步列出来的，**不能当作可行** |
| `rest_start` | 球被放到静止状态的那一段才有：`far_side_of_break`，放在断点对面；`reset_point`，前一个小节没走通，放在这个小节的盘点下 |
| `exit` | 出口，见下；变球段和断点段没有 |
| `corridor` | 走廊，见下；同上 |
| `mechanisms` | 这一段对机关做的事，见下 |
| `transform` | 只在变球段：`at`（变球器名）、`pos`、`from`、`to`；另有 `end`，新球出现后的位置和速度 |
| `no`、`arrives`、`arrives_by_nudge` | 只在细搜段：它跨过的是哪个断点；求解器量的扰动重放 20 次里还能到对面的次数（扰动 1 档），以及 1、3、10 档各自的次数。一个断点跨过以后球紧接着又过不去、又从前面重新跨过时，路线上会有它的几次跨越，只有最后一次带这个数 |
| `fan_bounds` | 纸球在风扇旁的段才有：用风扇推力的两档（盒子边上的球推得最少、最多）重跑，各到不到（`least`、`most`） |
| `break` | 只在断点段：`no`、`near`（这一侧最后的位置）、`far`（对面） |

**出口和走廊**都是粗略的（`rough: true`）：物理核里扰动重放的包络，不是证明出来的。扰动和求解器量容错率时一样大：起点在 x、z 上各偏正态 0.03，速度每个分量偏正态 0.3，每次换键提前或推后至多 2 步（均匀）。怎么重放分两种：

- 细搜段和腾空段整段重放 20 次，从这一段的起点扰动起。
- 跟随器走的段不能整段开环重放：它的按键是闭环选出来的，按原样从扰动过的起点重放，几秒内就会掉下去（第 1 关一段 30 游戏秒的，20 次全掉）。所以切成每 132 步（1 游戏秒）一窗，最后一窗止于段末，每一窗从路线在窗口开始时的状态扰动起，重放这一窗的按键 20 次。它说的是：开头还在路线上，一游戏秒里会偏多远。闭环的执行模型要跟上的正是这个。

`exit`：`at`、`vel` 是路线自己在段末的位置和速度；`trials`、`not_lost` 是最后一窗重放了几次、几次没丢球（没掉下去，也没误入变球器；止于变球器的段，进变球器算没丢）。下面几项取段末在没丢球的那几次加上路线本身里的范围；没丢球的不到 10 次时都为 `null`：

| 内容 | 含义 |
|---|---|
| `box` | 段末位置的盒子：`min`、`max` |
| `speed_h` | 水平速度大小的区间 |
| `vy` | 竖直速度的区间。只在腾空段、或段末有一次是悬空的时候写，否则为 `null` |
| `dir`、`dir_tol_deg` | 段末水平速度的方向（单位向量，`dy` 为 0）和各次与它的最大夹角。腾空段也只给水平方向：它的段末是落地前一步，扰动过的有的早一步落地、已经弹起，三维方向没有意义。路线在段末慢于 0.5 时不给方向；慢于 0.5 的那几次不参与夹角 |
| `from` | 是哪一种重放 |

`corridor`：`every` 为 1.0，`half_width[j]` 是这一段自己从起点量起的弧长在 [j, j+1) 那一截上，各窗没丢球的重放离路线（同一窗里路线各步的球心，取最近的一个）的最大距离。另有 `trials`、`lost`：所有窗合起来重放了几次、丢了几次。

**机关**：每项有 `name`（地图里物体的原名）、`type`、`does`、`ball`（能做这件事的球种，按 [mechanisms.md](mechanisms.md) 里量过的：栅栏只有石球推得开，推板木球和石球推得倒，跷跷板只有石球翻得过来）。机关的状态和粗层用的一样，从物理核的场景读：栅栏滑出 4.5 算开，推板转过 60° 算倒，跷跷板低于 −10° 算翻过来了。

| 情形 | 写法 |
|---|---|
| 推栅栏的段 | `{"type": "gate", "does": "push_open", "key": "+x", "hold_ms": 1500, "opened": true}`：按哪个键、按多久、推开了没有 |
| 一段里推倒了推板 | `{"type": "push_board", "does": "push_over", "state": [0, 1], "impact_speed": [低, 高], "pushed_over": 20, "trials": 20}`。撞上时的速度取撞上前一步球的速度：推板静止时一动不动，撞上的那一步它醒过来，球在这一步里就失去大部分速度，转角要到下一步才变，所以按它醒来的那一步算。区间是路线本身和从撞上前 66 步扰动起的 20 次里同样推倒了的那几次 |
| 一段里翻过了跷跷板 | `{"type": "seesaw", "does": "cross", "tips": true, "state": [0, 1]}` |
| 一段里别的状态变化 | `does` 为 `push_open`（不是推栅栏的段里推开了栅栏）或 `state_change` |
| 乘风扇上升的段 | 经过的每个风扇一项：`{"type": "fan", "does": "ride"}` |

`state` 是这一段开始和结束时那个机关的状态（0 或 1）。按键和按住多久只是提示，执行模型不照着回放。

和草案不同的地方（Ballrain 看过的是草案）：

- `exit` 多了 `at`、`vel`（路线自己的段末）和 `trials`、`not_lost`；包络不够 10 次时为 `null`，不硬给。
- `dir` 在腾空段也只给水平的（原因见上）；`dir_tol_deg` 是算出来的，不是固定的 15。
- `corridor` 多了 `trials`、`lost`、`from`；半宽是到同一窗里路线最近一步的距离。
- 每段多了 `ticks`、`start`、`assumed_rest_start`（在线上的为假）和 `rest_start`；细搜段多了 `no`、`arrives_by_nudge`；纸球在风扇旁的段多了 `fan_bounds`。
- 机关项多了 `state`；推栅栏多了 `opened`；推板的 `impact_speed` 带着 `pushed_over`、`trials`、`from`。乘风扇上升的段把风扇列成机关。
- 跷跷板只在翻过来时标；木球、纸球从跷跷板上过去而没有翻它，现在不标。
- 断点段的 `s` 为 `null`。
- 还没有的：段开始时机关应处的状态（`states`）、分支、SR 和 HS 的数，都排在后面（见计划）。
