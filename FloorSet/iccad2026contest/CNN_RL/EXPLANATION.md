# CNN_RL — GNN + CNN 序列式 Floorplan Placer（ICCAD 2026 Contest C）

> 報告用整合文件。涵蓋整體架構、模型內部接線、合法化鏈、訓練方法（BC vs RL）、
> 診斷實驗、結果與結論。對應程式碼在 `FloorSet/iccad2026contest/CNN_RL/`，完整逐 phase
> 歷史見 [`ALGORITHM.md`](ALGORITHM.md)。

---

## 1. 問題與評分

**任務**：FloorSet-Lite floorplan，n = 21–120 個矩形 block，需擺成無重疊、滿足硬約束
（fixed 形狀、preplaced 位置、MIB、cluster、boundary）且面積/線長盡量小的合法佈局。

**官方 cost（單一 case，越低越好）**：
```
Cost = (1 + 0.5·(HPWL_gap + Area_gap)) · e^(2·V) · max(0.7, RuntimeFactor^0.3)
     = 10  （infeasible：任何重疊 / 面積誤差 >1%）
```
- `HPWL_gap`, `Area_gap`：相對 GT baseline 的相對差，`max(0,·)` 夾住（贏不過 GT 不額外加分）。
- `V`：soft 約束（boundary/grouping/MIB）相對違反量。
- `RuntimeFactor = your_runtime / 所有投稿在該 case 的中位數`。

**Total Score**：跨 100 case 按 `e^(n/12)` 加權（n = block 數）→ **大 case 主導**。

> **本機評分把 RuntimeFactor 釘成 1.0**（沒有跨投稿中位數），所以本機 Total Score 只反映
> quality（HPWL/Area/violation）；runtime 只對官方 leaderboard 與開發迭代有意義。

---

## 2. 整體架構（兩段式）

核心理念：**把組合問題交給學習、把幾何合法化交給解析法**。

```
┌─ 學習端 (policy) ────────────┐   ┌─ 解析端 (legalizer) ─────────────┐
│ GNN 編碼 netlist (誰連誰)     │   │ skyline_legalize_shaped           │
│ + CNN 讀 rasterized 盤面      │   │   + contour/height shaping        │
│ + 序列地一格格選每個 block    │──▶│   + cohesion cluster              │
│   的落點 (cx,cy) 與 aspect    │   │   + real-cost gate                │
└──────────────────────────────┘   │ → slide_boundary → enforce_hard   │
                                    │ → _detailed_place → 合法佈局      │
                                    └───────────────────────────────────┘
```

**最重要的設計洞察**：**legalizer 比 policy 準確度更重要**。同一個未訓練好的 checkpoint，
配自寫 row-packer 得 Total Score 10.05；改接 `analytic_legalizer` 的 skyline legalizer →
**2.09**。所以 policy 只需給「方向大致對」的 centers，重活由 legalizer 承擔。
`rl_skyline_optimizer.py` 只 **import** `analytic_legalizer` 的純函式，不修改它。

---

## 3. 模型內部：兩編碼器 + Canvas + CNN 怎麼接

實際維度（程式碼值）：`node_dim=128, node_channels=32, hidden=64, n_conv=4, G=64,
in_channels=5`。

### 3.1 原始輸入（每題一組）

| Tensor | 形狀 | 內容 |
|---|---|---|
| `area_target` | `[N]` | 各 block 目標面積 |
| `constraints` | `[N,5]` | (fixed, preplaced, mib, cluster, boundary_code) |
| `b2b` | `[E,3]` | block↔block net：(i, j, weight) |
| `p2b` | `[Ep,3]` | pin↔block：(pin_idx, block_idx, weight) |
| `pins_pos` | `[P,2]` | I/O pin 固定座標 |

### 3.2 GNN 路徑（`gnn_encoder.py`，每題算一次，靜態）

netlist 在擺放中不變，故整個 rollout 只跑一次。

1. **node features `[N,10]`**：每 block 10 維 —
   `[sqrt(area)正規化, area/total, fixed, preplaced, has_mib, has_cluster, bd_L,R,T,B]`。
   只描述 block 自己，不含連通性。
2. **edges**：`b2b` → 無向（雙向）`edge_index[2,2E]` + `edge_weight[2E]`，丟 padding/越界。
3. **3 層 SAGELayer（weighted-mean message passing）**：
   ```
   input_proj: Linear(10→128)+ReLU → h[N,128]
   每層: agg_i = Σ_{j∈鄰} (h_j·w_ij) / Σ w_ij      (weighted mean，對 weight scale 不敏感)
         h_i ← ReLU(W_self·h_i + W_neigh·agg_i)
   ×3 → 聚合 3-hop 鄰域
   ```
4. **輸出 `node_emb[N,128]`**：每 block 含「自身 + 拓樸鄰域」的 128 維向量。

> GNN = 「這個 block 是什麼、連誰、在什麼拓樸位置」。**每題一次**，rollout 重複取 `node_emb[cur]`。

### 3.3 Canvas / Rasterize 路徑（`canvas_raster.py`，每放一塊重算）

把**當前部分擺放**轉成 `[5,64,64]` 多通道影像（CNN 的視覺輸入）。座標：grid `[row,col]`，
row→y、col→x，cell 左下 =`(col·cell_w, row·cell_h)`。

| Ch | 名稱 | 內容 |
|---|---|---|
| 0 | OCCUPANCY | 已放 block 覆蓋的 cell=1 |
| 1 | DENSITY | 每 cell 覆蓋數（重疊熱點），用 2D 差分積分圖一次算完 |
| 2 | WL_PULL | 對當前 block：放各 cell 到其**已放 b2b 鄰居**的加權 Manhattan HPWL，正規化 [0,1]，低處值高 |
| 3 | FEASIBILITY | 當前 block(給定 w,h)左下放此格能否塞進 canvas =1 |
| 4 | PIN_PULL | 同 WL_PULL 但對象是連到的 **I/O pin**(p2b)，讓模型看見 pin ring |

> Canvas = 「盤面長怎樣 + 對當前這塊而言哪裡是好位置」。**每步重算**。

### 3.4 CNN trunk + heads（`policy_net.py`）—— 兩路在此合流

逐步追維度（單樣本，G=64）：
```
輸入: canvas[5,64,64]   node_emb[cur][128]

(1) node_proj: Linear(128→32)                  → node_vec[32]
(2) broadcast: → node_map[32,64,64]            （block 向量鋪滿每像素）
(3) 融合: cat([canvas,node_map])               → x[37,64,64]
           ↑ 每像素「知道」正在放哪塊 + 其拓樸脈絡
(4) stem: Conv2d(37→64,3x3,pad1)+ReLU          → x[64,64,64]
(5) 4×dilated 殘差 conv, dilation=1,2,4,8:
      x ← ReLU(Conv2d(64→64,3x3,dil=d)(x)+x)
      ↑ 感受野指數成長(1→2→4→8→16)，不降解析度即覆蓋全域
(6) 全域分支:
      g = global_mlp(x 在 HxW 平均)[64]
      x ← ReLU(fuse(Conv2d(128→64,1x1))(cat[x, g 廣播])) → x[64,64,64]
      ↑ 每 cell 額外拿一份「整張佈局摘要」(Phase 10 加；
        舊純 3x3 trunk 只看 ~9x9 鄰域→case99 鬆散擺放)
(7) heads:
      policy_head: Conv2d(64→1,1x1)            → logits[64,64]
      pooled = x 在 HxW 平均[64]
      value_head: [64]→[1]                     → V(s)  (PPO baseline)
      aspect_head:[64]→[5]                     → aspect bucket logits
```
**直覺**：GNN 說「我是連到右邊兩顆最大 macro 的大塊」；canvas 說「右半邊空、pin-pull 高」；
CNN 疊起來 → 右半邊 cell logits 拉高。**GNN=我是誰連誰；CNN=盤面長怎樣哪裡好**。

### 3.5 動作選擇（每步）
```
probs = softmax(logits 被 feasibility mask 蓋 -inf 後)   # 不合法格機率=0
greedy: argmax(probs)  (推論)   |  sample: multinomial(probs)  (RL/best-of-K)
free soft block: 另選 aspect_idx → 從 ASPECT_BUCKETS 取 area-preserving (w,h)
```

### 3.6 每步迴圈（兩路徑交織）
```
題目 → GNN.encode_problem() 一次 → node_emb[N,128]            ┐靜態
迴圈(待放 block，面積大→小):                                  │
  ├ rasterize_env(env) → canvas[5,64,64]   (每步重算盤面)      │
  ├ policy._trunk(canvas, node_emb[cur])   (兩路合流)          │
  ├ masked softmax → 選 cell(+aspect)                          │
  └ env.step() 放 block → 更新 positions ──────────────────────┘
迴圈結束 → 各 block (cx,cy)+aspect → legalizer 鏈 → 合法佈局 → 評分
```
**GNN 跑一次（netlist 靜態），CNN+canvas 每步跑一次（盤面在變）**。

### 3.7 維度速查
| 階段 | 張量 | 形狀 |
|---|---|---|
| GNN 輸入 | node features | `[N,10]` |
| GNN 輸出 | node_emb | `[N,128]` |
| Canvas | raster | `[5,64,64]` |
| node 投影+廣播 | node_map | `[32,64,64]` |
| 融合後 | x | `[37,64,64]` |
| stem 後 / trunk | x | `[64,64,64]` |
| policy 輸出 | logits | `[64,64]` |
| value / aspect | — | `[1]` / `[5]` |

---

## 4. MDP 形式（序列擺放）

| 元素 | 定義 |
|---|---|
| State | 已放 block + canvas raster + GNN embeddings + 下一個 block id |
| Action | grid cell index `[0,G*G)` = 下塊左下角；free soft block 另選 aspect bucket |
| 順序 | preplaced/fixed 先放(定位、排除決策)，其餘按面積大→小 |
| Reward | 僅 terminal：`-(合法化後 contest cost)` |

---

## 5. 合法化鏈（真正的分數來源，Phase 15）

centers → `parse_and_init`/`prepack_clusters` → `skyline_legalize_shaped` →
`slide_boundary` → `enforce_hard` → `_detailed_place`。所有大勝都在這層（Total 1.85→1.60）：

- **B1 contour-aware shaping**：skyline packer 落塊時選最貼合 contour 凹口的 area-constant aspect。
- **B1+ height-aware landing**（β=0.3）：landing 目標加 `β·h` 懲罰，避免高瘦塊頂高 skyline（最大 legalize 勝）。
- **real-cost gate**：用精確 contest cost 形式（含 `e^{2V}`、candidate-relative min baseline）選 reshape 候選，而非舊的 `bbox·HPWL` proxy。
- **obstacle-aware x-candidates**：讓寬 cluster super-block 卡進 preplaced 障礙**旁**而非被頂過去。
- **deformable cohesion clusters**（**最大單一勝 −9%**）：cluster 成員當可變形個體擺、加 cohesion 拉力形成連通 blob，貼合 contour 而非死板矩形。

---

## 6. 訓練方法：BC（已上線）vs RL（Phase 17）

**重要命名澄清**：「RL」在此專案指**架構**（MDP placer、policy/value net、rollout），
**不是訓練方法**。所有 shipping checkpoint（phase11, phase13）都是 **BC（行為模仿）**訓練。
「RL rollout」= 把 policy 跑過 MDP 做推論，BC 訓練的 policy 也能這樣 rollout。
Phase 17 才是第一次真正用 **RL 訓練**。

| | BC rollout（phase13, `train_network.py`） | RL fine-tune（phase17, `train_rl_finetune.py`） |
|---|---|---|
| 訓練訊號 | GT cell label（模仿資料集擺法） | **真實 post-legalize contest cost** |
| 目標 | 逐 block cross-entropy 對 GT cell(+aspect CE) | PPO policy-gradient：拉高降低 cost 的動作機率 |
| 看 legalizer? | **否**（從不合法化、看不到 cost） | **是**（reward = 完整 `solve()` pipeline） |
| 信用分配 | 每 block 獨立對其 GT cell 評分 | terminal reward 攤回整 episode（per-case baseline） |
| 探索 | 無（抄 GT） | stochastic rollout（位置+aspect 都採樣） |
| 資料 | GT label，~10k 即飽和 | 自產訊號；1M 實例當 RL 環境 |
| 穩定器 | soft label(σ=1.5) | KL-anchor 回 BC policy、per-case baseline、frozen GNN |
| reward(去 runtime) | — | `−(1+0.5(hpwl_gap+area_gap))·e^{2V}`，runtime=1.0 |

**兩者推論路徑完全相同**，只有權重不同。

---

## 7. RL 前置診斷（決定 RL 值不值得做）

reward 一律用 **runtime-free** contest cost（runtime=1.0 → runtime 項=1.0）。

### 7.1 更正一個錯誤前提
舊 README 稱 Phase 12 PPO 失敗是因 reward 用未合法化的 raw 佈局。**錯**：
`archive/train_ppo.py` 顯示 Phase 12 已用 legalized reward + KL-anchor + per-case baseline，
仍失敗。當時真正較弱處：(a) reward 走**舊 plain skyline**(~1.85)非現在的 shaped 鏈(1.60)；
(b) cost 用 `compute_training_loss_differentiable` 非真實 metric；(c) 從 phase11 暖啟動、
**aspect head 沒進 RL action**。

### 7.2 Step 0 — leverage 診斷（`poc_rl_leverage.py`）
問題：給定現在的強 legalizer，cost 對 centers 有多敏感？做法：greedy + K=16 sampled rollout
各過完整 pipeline，量 `greedy − best-of-K`。

| case 區段 | greedy 均 | best-of-16 均 | ceiling | best<greedy |
|---|---|---|---|---|
| 小(bc 21–40) | 1.8072 | 1.6379 | **0.169** | 18/20 |
| 大(bc 101–120) | 1.6265 | 1.5750 | **0.052** | 14/20 |

headroom 真實但大 case 縮小（密→更被 legalizer 主導）。按 `e^{n/12}` **加權後天花板僅 ~0.029**。
→ 不是死路，但加權 upside 微薄。

### 7.3 best-of-K 推論 + K-sweep（`RL_NSAMPLE`，全 100）
K 次 rollout（候選0=greedy + K−1 sampled），各合法化，用 candidate-relative gate 選最低真實 cost
（無 GT，inference-valid，永不劣於 greedy）。

| K | Total Score(quality) | Avg Cost | Avg Runtime | Δquality | runtime× |
|---|---|---|---|---|---|
| 1 (greedy) | 1.6039 | 1.6847 | 0.83s | — | 1.0× |
| 2 | 1.5956 | 1.6564 | 1.39s | −0.52% | 1.67× |
| 4 | 1.5748 | 1.6198 | 3.07s | −1.81% | 3.70× |
| 8 | **1.5623** | 1.5886 | 5.54s | **−2.59%** | 6.67× |

quality 可達 ~1.56，但 **官方 `runtime^0.3` 大概率吃掉**：若 runtime 在 `max(0.7,·)` 地板之上，
K=2 懲罰即 `1.675^0.3=1.174`(+17%)>賺的 0.52%。要「免費」需 field median >4.6s(K=2)/>18.5s(K=8)，
對 ~0.5–1s 解法不太可能。→ best-of-K 非安全 stopgap，但**證明 quality target ~1.56 可達**。

### 7.4 ★ 目前方法的上界（Method Upper Bound）

> **best-of-K 給出「目前這套 policy + legalizer」在 quality 上的實務上界 ≈ Total Score 1.56。**

意義：best-of-K = 用**同一個 policy**反覆採樣、每次都挑「合法化後真實 cost 最低」的那一版。
因此它代表「在不改 policy、不改 legalizer 的前提下，這套方法能擠出的最好 quality」——
即**這個 center-prediction 架構的天花板**。

```
analytic baseline   1.9788  ─────────────────────────────────┐
                                                              │ legalizer 端的累積改良
目前最佳 (greedy)    1.6039  ◀── 提交版，runtime 0.83s         │ (1.85 → 1.60)
                                                              │
方法上界 (best-of-8) 1.5623  ◀── ★ 這套方法的天花板            │ 只剩 ~0.04 的 policy headroom
                                  (代價: 6.7× runtime)         ┘
```

**給大家的一句話**：legalizer 端已把 quality 從 1.98 推到 1.60；剩下「擺位/centers」這條路
的全部潛力（即使用 oracle 般的 best-of-K 去榨）也只到 **~1.56**，再往下就得換槓桿（不是更好的
centers，而是更強的 legalizer 或新的決策維度）。而且這 ~0.04 還得付 6.7× runtime，RL 想把它
免費折進權重——目前（Phase 17）尚未成功。

---

## 8. Phase 17 — RL fine-tune（`train_rl_finetune.py`）

warm-start phase13、reward=完整 pipeline 真實 cost、aspect 入 action、KL-anchor + per-case
baseline（Phase 12 對的部分留著）、held-out 在驗證集 spread 上跑 greedy。

**結果（40 iters, batch 8×rollouts 8, lr 3e-5, kl 0.1, 全 100 評分）**：

| | Total Score | Avg Cost | Feasible | Runtime |
|---|---|---|---|---|
| phase13_aspect (BC, baseline) | **1.6039** | 1.6847 | 100/100 | 0.83s |
| phase17_rl (RL fine-tune) | 1.6271 | 1.7107 | 100/100 | **0.66s** |

- 訓練不發散（KL~0.005，不同於 Phase 12），但 held-out proxy 在 1.66–1.77 震盪、無清楚下降。
- **quality 沒贏 BC**（1.6039→1.6271, +1.4%），**但 runtime 更快**（0.66<0.83s，RL-tuned centers
  觸發較少 reshape gate 候選）。100/100 feasible。**保留為比較點**：quality 持平但更快的 checkpoint
  在官方 `runtime^0.3` 下未必被支配。
- **為何沒贏**（診斷已預測）：(1) 加權天花板僅 0.029；(2) RL 優化未加權平均，但 Total Score 加權
  大 case，而大 case headroom 最小（objective–metric 權重不一致）。
- **未試變體**（低信心）：按 `e^{n/12}` 加權 advantage / 多採樣大 case。

---

## 9. 結果總表

| Optimizer / checkpoint | Total Score | Avg Cost | Feasible | Runtime |
|---|---|---|---|---|
| `analytic_legalizer`（baseline） | 1.9788 | 2.0632 | 100/100 | — |
| **phase13_aspect + Phase15 legalizer（目前最佳，提交此）** | **1.6039** | 1.6847 | 100/100 | 0.83s |
| phase16_repro（從頭複刻 phase13） | 1.6506 | 1.7356 | 100/100 | — |
| phase16_wl_asp（+wl-aux） | 1.6855 | 1.7410 | 100/100 | — |
| phase17_rl（RL fine-tune，比較點） | 1.6271 | 1.7107 | 100/100 | 0.66s |
| best-of-8 推論（quality 探針） | 1.5623 | 1.5886 | 100/100 | 5.54s |

---

## 10. 關鍵結論（報告重點）

1. **legalizer > policy 準確度** —— 所有大勝在 legalizer 端（1.85→1.60）。
2. **BC 在 ~10k 飽和** —— BC 目標（模仿 GT）天花板 ~1.60–1.65；加資料、wl-aux 輔助 loss、
   砍 aspect head 都無效甚至退步。
3. **從頭重訓複刻不出 phase13**（1.65 vs 1.60）—— phase13 的品質來自跨 phase 累積；RL 等任何
   fine-tune 都應從 phase13 暖啟動，不要從頭。
4. **RL leverage 加權天花板僅 ~0.029** —— cost 對 centers 敏感（有梯度），但大 case 被 legalizer
   主導，而大 case 主導 Total Score。
5. **best-of-K = 目前方法的上界 ≈ 1.56** —— 用同一 policy 反覆採樣挑最佳，代表「不改
   policy/legalizer 下能擠出的最好 quality」。從 greedy 1.6039 只剩 ~0.04 的 centers headroom，
   且要付 6.7× runtime。再往下必須換槓桿（更強 legalizer / 新決策維度），不是更好的 centers。
6. **三次 RL 嘗試（Phase 4/12/17）quality 都沒贏 BC** —— 定論：**centers 這條 RL 槓桿撬不動，
   剩下的真槓桿在 legalizer 端**。phase17 runtime 更快，保留為比較點。

---

## 11. 檔案地圖

| 檔案 | 角色 |
|---|---|
| `rl_skyline_optimizer.py` | 主 optimizer（rollout → legalize 鏈），**提交此**；含 `RL_NSAMPLE` best-of-K、`ASPECT_PASS` 開關 |
| `gnn_encoder.py` | GNN（SAGE-style，`[N,10]→[N,128]`） |
| `canvas_raster.py` | 部分擺放 → `[5,G,G]` raster |
| `policy_net.py` | CNN trunk + policy/value/aspect heads |
| `placement_env.py` | MDP 環境（reset/step/reward） |
| `skyline_shape.py` | Phase 15 contour-aware in-packer shaping |
| `train_network.py` | BC 訓練（vectorised + parallel） |
| `train_rl_finetune.py` | Phase 17 RL fine-tune（PPO + KL-anchor） |
| `poc_rl_leverage.py` | RL leverage 診斷（greedy vs best-of-K 散布） |
| `plot_rl_skyline.py` | 視覺化（`--no-wires` 隱藏連線；`--ground-truth`） |
| `ALGORITHM.md` | 完整逐 phase 史與所有實驗細節 |
| `checkpoints/phase13_aspect.pt` | 目前 default checkpoint |
| `checkpoints/phase17_rl.pt` | RL fine-tune 比較點 |

---

## 12. 圖（已生成，無 wires 版）

| 檔案 | 內容 |
|---|---|
| `k1_case0_nowire.png` / `k8_case0_nowire.png` | case0 greedy vs best-of-8（K=8 明顯更矮更密） |
| `k1_case99_nowire.png` / `k8_case99_nowire.png` | case99（best-of-8 gate 保留 greedy，兩圖相同） |

重畫：`RL_CKPT=... ASPECT_PASS=1 RL_NSAMPLE=<K> python3 CNN_RL/plot_rl_skyline.py --test-id <id> --no-wires --out <path>`

---

## 13. 復現指令（從 `FloorSet/iccad2026contest/` 執行）

```bash
# 評分目前最佳
python3 iccad2026_evaluate.py --evaluate CNN_RL/rl_skyline_optimizer.py

# 評分指定 checkpoint（RL 比較點，需 ASPECT_PASS）
RL_CKPT=CNN_RL/checkpoints/phase17_rl.pt ASPECT_PASS=1 \
  python3 iccad2026_evaluate.py --evaluate CNN_RL/rl_skyline_optimizer.py

# best-of-K 推論（K 倍 runtime）
RL_NSAMPLE=8 python3 iccad2026_evaluate.py --evaluate CNN_RL/rl_skyline_optimizer.py

# BC 訓練
python3 CNN_RL/train_network.py --num-samples 20000 --epochs 1 --grid 64 --workers 16 \
  --soft-sigma 1.5 --aspect-weight 0.5 --ckpt-name phaseXX.pt

# RL fine-tune
python3 CNN_RL/train_rl_finetune.py --warmstart CNN_RL/checkpoints/phase13_aspect.pt \
  --iters 40 --batch-samples 8 --rollouts 8 --kl-coef 0.1 --ckpt-name phase17_rl.pt

# leverage 診斷
RL_CKPT=CNN_RL/checkpoints/phase13_aspect.pt ASPECT_PASS=1 \
  python3 CNN_RL/poc_rl_leverage.py --n-cases 20 --rollouts 16
```
