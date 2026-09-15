# RLDemo — 交接说明（给本地 Claude Code）

用中文和用户交流。用户是仓库作者，2017 年写了原始示例；2026-09 起在 Claude Code 远程会话里把它做成了一个
"网络到底记住了还是学会了"的研究项目。本文件是远程会话的交接。分支：`claude/gifted-brahmagupta-mc64oz`
（PR #1 未合并，所有工作都在这个分支上，继续在这个分支提交）。

## 项目是什么

二阶魔方（2×2×2，固定 DBL 角块）：3,674,160 个状态，9 个动作（U/R/F × 90/180/270），God's number 11。
用 BFS 算出每个状态的真实距离，所以任何策略、任何价值函数都能和最优解逐状态对比。在这个测试床上比较：

- 查表 Q-learning（全状态空间，2 分钟到 100%，作为标准答案）
- 神经网络 Q-learning（numpy CPU 版；PyTorch 版支持 MPS / CUDA），研究它学到了什么

## 文件地图

| 文件 | 作用 |
|---|---|
| `rubik.py` | 环境、完美哈希 `encode/decode`（排列 Lehmer 码 × 729 + 朝向三进制）、`transitions()`（N×9 转移表，缓存在 `cache/transitions.npy`）、`bfs_distances()`、`cubies()`、`symmetries()`（6 个整体对称的状态映射，缓存 `cache/symmetries.npy`，首次构建约 30 s）、`EpisodeRecorder` |
| `play_rubik.py` | 主 CLI：`train / eval / solve / serve`，`--tag` 把实验放进 `cache/<tag>/`；查表 `Trainer`；pip 风格进度面板 `Progress`；`greedy_rollout`、`beam_solve`；HTTP 服务 `Serving`（/metrics /trace /spectator /policy.bin） |
| `qnet.py` | numpy MLP 和 `NetTrainer`；特征：`onehot`（144 维贴纸）、`cubie_onehot`（70 维）；`qfunction(net)` 按输入宽度自动选特征 |
| `qnet_torch.py` | `TorchNetTrainer`：全状态特征驻留设备、all-actions 目标、double DQN、课程、`--features`、`--promote-by`、`--symmetry-aug`、批量束搜索 `beam_success`；所有碰 GPU 的方法过一把 RLock（MPS 多线程会 abort） |
| `dashboard.html` | 训练监控（深色遥测风格，浅色可切）：tiles、可调纵轴的曲线、课程升级标记、两张热力图（解出率 × 距离 × 进度，已采样比例 × 距离 × 进度）、3D 魔方（实况回放 / 旁观者）、指标说明 |
| `playground.html`, `cube-model.js`, `cube-3d.js` | 3D 魔方页面（含真实魔方模式）和共享的前端模型 / three.js 视图 |
| `symmetry_test.py` | 诊断 D2：采样过的状态 vs 它们的未采样对称像 vs 随机未采样状态 |
| `capacity.py` | 诊断 D4：同一架构监督训练 n 个状态的记忆上限；`--subset` 只用某次 RL 采样过的状态；`--features` |
| `run_matrix.sh`, `compare.py` | 控制变量实验矩阵（base / onestep / beam / cubie / symaug）和对比表 |
| `paper/main.tex` | 论文初稿 v0.1（article 模板，投稿时换会议 sty） |
| `README.md` | 面向读者的完整说明，方法、结果、GPU、诊断、干预开关 |

产物都在 `cache/`（gitignore）。用户本机的关键目录：`cache/main/`（M1 Max 主跑，1024-1024-512，60 min，K 卡 7）。

## 已经确立的结论（论文和报告的核心，不要推翻除非有新证据）

M1 Max 主跑（`--tag main`）的四个诊断：

1. **见过 vs 没见过**：同一距离上采样过和未采样过的状态解出率几乎相同（d8：22.4% vs 17.3%；d9：8.4% vs 9.9%），价值误差也相同 → 不是按状态记忆。
2. **对称像测试**：采样过状态的未采样对称像，解出率等于随机未采样状态（比值 1.0）→ 没学到对称结构。
3. **贪心 vs 束搜索**：同一个网络，贪心约 16%（由分距离表推算），宽度 32 束搜索 90.3% → 价值函数在从未训练的区域里仍然有用，只是精度不够（d7 价值误差 0.95 ≈ 相邻动作分数差 1，贪心在抛硬币）。
4. **监督容量上限**：同一架构（173 万参数）监督训练能精确背 30 万到 100 万个状态（≈ 2 bit/参数）；喂全部 367 万时单步只对 56% 却贪心解出 79%（2 分钟）→ 瓶颈不在网络，在 RL 的学习信号（自举、非平稳）和数据覆盖（RL 只采样了 30%）。

合起来：网络学到的是贴纸空间里的平滑插值，既不记状态也不懂对称。K 卡在 7 是升级判据依赖贪心造成的自锁。
4090 跑（2048-2048-1024，5.3 h）同样卡 K=7，贪心 30%，采样 78.8%，未见过 12.6%：参数买的是精度不是记忆。

关键数字：状态按距离 [1, 9, 54, 321, 1847, 9992, 50136, 227536, 870072, 1887748, 623800, 2644]，92% 在 d8–10。
瞎猜单步准确率 25.4%（平均每状态 2.28 个最优动作）。查表：1.8 亿步，127 s，100%。

## 当前状态和待办

刚完成（已推送）：四个干预开关 + 实验矩阵。**用户接下来要在 M1 Max 上跑** `bash run_matrix.sh`
（默认每组 40,000 次更新、同 seed、1024-1024-512、batch 4096，每组约 30–40 min），然后 `python3 compare.py m_base m_onestep m_beam m_cubie m_symaug`，
以及 `python3 symmetry_test.py --tag m_symaug`。目标：看哪一个干预提升最大。

结果拿到后的待办：

1. 把矩阵结果写进 `paper/main.tex`（Planned experiments 一节要变成结果）和 `README.md` 的结果部分。
2. 论文 TODO：作者信息、两次跑的更新次数 / 吞吐、直接测的贪心随机解出率（`eval --agent net --tag main`）、仓库地址、逐条核对参考文献。
3. 论文补强顺序：3 个 seed → 宽度扫描（固定更新次数、每档调 lr）→ 监督容量二维扫描（参数 × n）→ 第二个可穷举域（如 8-puzzle）。投稿路径：arXiv + RLC / 分析类 workshop，之后 RLC 正会或 TMLR。
4. 公平性实验：`capacity.py --subset cache/main/net_seen.npy` 拆开"数据覆盖"和"学习信号"。
5. 可选叠加实验：`--features cubie --symmetry-aug --promote-by beam` 三个一起。
6. 三个 artifact（表格监控、DQN 监控、playground）是远程会话发布的，本地无法更新，不用管。

## 约定和坑

- 提交信息用英文，末尾附 `Co-Authored-By` 行；每完成一项就提交并推送到上面的分支；不要开新 PR。
- 每个实验用 `--tag`，文件在 `cache/<tag>/`：`net.npz`（numpy 格式权重，sizes 决定特征种类）、`net_policy.npy`、`net_metrics.json`、`net_seen.npy`（packbits 位图）。`--resume` 从这些续训，预算（`--minutes/--steps/--updates`）只算本次。
- `--features cubie`、`--symmetry-aug`、`--promote-by`、`--updates` 只有 torch 后端支持。
- MPS：训练线程和 HTTP 线程不能同时用 GPU，已用 RLock 串行；`--amp` 只在 CUDA/CPU 生效。`/policy.bin` 第一次会让训练停几秒（全策略导出）。
- 用户之前遇到过：pip 的 SOCKS 代理错误（`env -u ALL_PROXY -u all_proxy pip ...`）；Windows 上 `source` 无效、PyPI torch 是 CPU 版（用 cu124 index）。
- `--plain` 或重定向输出时每次评估一行；TTY 下是原地刷新的面板。日志里搜 `↑ 课程升级` 看 K 的变化。
- 评估指标定义（CLI 面板和 dashboard 的"指标说明"里有同样的文字）：解出率 = 每个距离抽 400 个贪心 30 步、按状态数加权；已采样 = 进过更新的状态比例；泛化 = 未采样状态的贪心解出率；|Q+d| = |max Q + 真实距离| 均值；beam_random / onestep_random 同样按距离加权。
- 沙盒里没有 LaTeX；本地编译 `paper/main.tex` 用 Overleaf 或 pdflatex。
