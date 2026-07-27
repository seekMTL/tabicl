## 声明
`experiments/exp3_joint_projection/backupv2/2_group`里只是 embedding.py、eval_joint_vs_baseline.py、regressor.py 与 `experiments/exp3_joint_projection/backupv2/1_combine` 里对应的文件有所改动，其余的都和 1_combine 目录的一样

## 原始实验记录(reward 和 delta_state 标签组合加性注入)
执行命令：
```bash
CUDA_VISIBLE_DEVICES=1 python3 experiments/exp3_joint_projection/eval_joint_vs_baseline.py 280
```

终端输出:
```
Epoch 280: 191620 train / 10000 test (limited to 10000)
y 范围: reward [-1.645,6.318]  delta [-6.934,7.289]

=== Baseline: 12 x TabICLRegressor(num_outputs=1) ===
耗时: 388.5s

=== Joint: 1 x TabICLRegressor(num_outputs=12) ===
耗时: 352.8s

======================================================================
   Dim Label          Baseline MSE      Joint MSE Ratio(J/B)
  ------------------------------------------------------------------
     0 reward             0.000546       0.000585       1.07x
     1 delta[0]           0.000000       0.000000       2.22x
     2 delta[1]           0.000000       0.000000       1.39x
     3 delta[2]           0.000000       0.000000       2.49x
     4 delta[3]           0.000000       0.000000       1.37x
     5 delta[4]           0.000001       0.000002       1.36x
     6 delta[5]           0.000142       0.000218       1.54x
     7 delta[6]           0.000181       0.000353       1.95x
     8 delta[7]           0.001758       0.002484       1.41x
     9 delta[8]           0.000625       0.001342       2.15x
    10 delta[9]           0.002393       0.003100       1.30x
    11 delta[10]          0.006073       0.012886       2.12x
  ------------------------------------------------------------------
         REWARD MSE       0.000546       0.000585       1.07x
     DELTA(avg) MSE       0.001016       0.001853       1.82x
        OVERALL MSE       0.000977       0.001748       1.79x

  加速比: 1.1x (388.5s → 352.8s)

```

## 原始实验记录(reward 和 delta_state 标签分组后加性注入)
### Epoch 0 (6K 样本)
执行命令：
```bash
CUDA_VISIBLE_DEVICES=1 python experiments/exp3_joint_projection/eval_joint_vs_baseline.py 0
```

终端输出：
```
Epoch 0: 4020 train / 1980 test (limited to 10000)
y 范围: reward [-1.645,3.373]  delta [-6.220,3.157]

=== Baseline: 12 x TabICLRegressor(num_outputs=1) ===
耗时: 5.4s

=== Joint: 1 x TabICLRegressor(num_outputs=12) ===
耗时: 1.1s

======================================================================
   Dim Label          Baseline MSE      Joint MSE Ratio(J/B)
  ------------------------------------------------------------------
     0 reward             0.009108       0.010206       1.12x
     1 delta[0]           0.000000       0.000000       1.43x
     2 delta[1]           0.000000       0.000000       1.28x
     3 delta[2]           0.000000       0.000000       4.86x
     4 delta[3]           0.000000       0.000001       1.65x
     5 delta[4]           0.000000       0.000000       1.69x
     6 delta[5]           0.000394       0.000467       1.19x
     7 delta[6]           0.000451       0.000547       1.21x
     8 delta[7]           0.005114       0.010669       2.09x
     9 delta[8]           0.001854       0.006837       3.69x
    10 delta[9]           0.009583       0.020015       2.09x
    11 delta[10]          0.014721       0.032217       2.19x
  ------------------------------------------------------------------
         REWARD MSE       0.009108       0.010206       1.12x
     DELTA(avg) MSE       0.002920       0.006432       2.20x
        OVERALL MSE       0.003435       0.006747       1.96x

  加速比: 5.0x (5.4s → 1.1s)
```

### Epoch 100 (106K 样本)
#### 不限制测试集上限
执行命令：
```bash
CUDA_VISIBLE_DEVICES=0 python experiments/exp3_joint_projection/eval_joint_vs_baseline.py 100 --samples 0
```

终端输出:
```
Epoch 100: 71020 train / 34980 test
y 范围: reward [-1.645,5.945]  delta [-6.934,7.289]

=== Baseline: 12 x TabICLRegressor(num_outputs=1) ===
耗时: 94.8s

=== Joint: 1 x TabICLRegressor(num_outputs=12) ===
耗时: 73.7s

======================================================================
   Dim Label          Baseline MSE      Joint MSE Ratio(J/B)
  ------------------------------------------------------------------
     0 reward             0.001424       0.001628       1.14x
     1 delta[0]           0.000000       0.000000       1.70x
     2 delta[1]           0.000000       0.000000       1.01x
     3 delta[2]           0.000000       0.000000       1.55x
     4 delta[3]           0.000000       0.000000       1.58x
     5 delta[4]           0.000002       0.000002       0.99x
     6 delta[5]           0.000219       0.000226       1.03x
     7 delta[6]           0.000298       0.000337       1.13x
     8 delta[7]           0.002822       0.003414       1.21x
     9 delta[8]           0.001012       0.001820       1.80x
    10 delta[9]           0.003085       0.004706       1.53x
    11 delta[10]          0.007746       0.016643       2.15x
  ------------------------------------------------------------------
         REWARD MSE       0.001424       0.001628       1.14x
     DELTA(avg) MSE       0.001380       0.002468       1.79x
        OVERALL MSE       0.001384       0.002398       1.73x

  加速比: 1.3x (94.8s → 73.7s)
```

#### 限制测试集上限 10k
执行命令：
```bash
CUDA_VISIBLE_DEVICES=1 python experiments/exp3_joint_projection/eval_joint_vs_baseline.py 100
```

终端输出:
```
Epoch 100: 71020 train / 10000 test (limited to 10000)
y 范围: reward [-1.645,5.945]  delta [-6.934,7.289]

=== Baseline: 12 x TabICLRegressor(num_outputs=1) ===
耗时: 76.0s

=== Joint: 1 x TabICLRegressor(num_outputs=12) ===
耗时: 58.8s

======================================================================
   Dim Label          Baseline MSE      Joint MSE Ratio(J/B)
  ------------------------------------------------------------------
     0 reward             0.001567       0.001794       1.15x
     1 delta[0]           0.000000       0.000000       1.77x
     2 delta[1]           0.000000       0.000000       1.00x
     3 delta[2]           0.000000       0.000000       1.44x
     4 delta[3]           0.000000       0.000000       1.62x
     5 delta[4]           0.000002       0.000002       1.00x
     6 delta[5]           0.000214       0.000230       1.07x
     7 delta[6]           0.000269       0.000338       1.26x
     8 delta[7]           0.002571       0.003447       1.34x
     9 delta[8]           0.001018       0.001907       1.87x
    10 delta[9]           0.003583       0.005094       1.42x
    11 delta[10]          0.008452       0.017098       2.02x
  ------------------------------------------------------------------
         REWARD MSE       0.001567       0.001794       1.15x
     DELTA(avg) MSE       0.001465       0.002556       1.75x
        OVERALL MSE       0.001473       0.002493       1.69x

  加速比: 1.3x (76.0s → 58.8s)
```

### Epoch 280 (286K 样本)
#### 不限制测试集上限

执行命令：
```bash
CUDA_VISIBLE_DEVICES=0 python experiments/exp3_joint_projection/eval_joint_vs_baseline.py 280 --samples 0
```

终端输出:
```
Epoch 280: 191620 train / 94380 test
y 范围: reward [-1.645,6.318]  delta [-6.934,7.289]

=== Baseline: 12 x TabICLRegressor(num_outputs=1) ===
耗时: 527.3s

=== Joint: 1 x TabICLRegressor(num_outputs=12) ===
耗时: 482.7s

======================================================================
   Dim Label          Baseline MSE      Joint MSE Ratio(J/B)
  ------------------------------------------------------------------
     0 reward             0.000791       0.000868       1.10x
     1 delta[0]           0.000000       0.000000       1.79x
     2 delta[1]           0.000000       0.000000       0.89x
     3 delta[2]           0.000000       0.000000       1.41x
     4 delta[3]           0.000000       0.000000       1.35x
     5 delta[4]           0.000001       0.000001       1.16x
     6 delta[5]           0.000135       0.000148       1.10x
     7 delta[6]           0.000170       0.000216       1.27x
     8 delta[7]           0.001647       0.001897       1.15x
     9 delta[8]           0.000552       0.000943       1.71x
    10 delta[9]           0.001894       0.002607       1.38x
    11 delta[10]          0.005522       0.012109       2.19x
  ------------------------------------------------------------------
         REWARD MSE       0.000791       0.000868       1.10x
     DELTA(avg) MSE       0.000902       0.001629       1.81x
        OVERALL MSE       0.000893       0.001566       1.75x

  加速比: 1.1x (527.3s → 482.7s)
```

#### 限制测试集上限 10k

执行命令：
```bash
CUDA_VISIBLE_DEVICES=1 python experiments/exp3_joint_projection/eval_joint_vs_baseline.py 280
```

终端输出:
```
Epoch 280: 191620 train / 10000 test (limited to 10000)
y 范围: reward [-1.645,6.318]  delta [-6.934,7.289]

=== Baseline: 12 x TabICLRegressor(num_outputs=1) ===
耗时: 387.9s

=== Joint: 1 x TabICLRegressor(num_outputs=12) ===
耗时: 352.3s

======================================================================
   Dim Label          Baseline MSE      Joint MSE Ratio(J/B)
  ------------------------------------------------------------------
     0 reward             0.000546       0.000616       1.13x
     1 delta[0]           0.000000       0.000000       1.85x
     2 delta[1]           0.000000       0.000000       0.78x
     3 delta[2]           0.000000       0.000000       1.38x
     4 delta[3]           0.000000       0.000000       1.54x
     5 delta[4]           0.000001       0.000002       1.26x
     6 delta[5]           0.000142       0.000148       1.04x
     7 delta[6]           0.000181       0.000259       1.43x
     8 delta[7]           0.001758       0.001989       1.13x
     9 delta[8]           0.000625       0.001115       1.78x
    10 delta[9]           0.002393       0.002269       0.95x
    11 delta[10]          0.006073       0.010025       1.65x
  ------------------------------------------------------------------
         REWARD MSE       0.000546       0.000616       1.13x
     DELTA(avg) MSE       0.001016       0.001437       1.41x
        OVERALL MSE       0.000977       0.001369       1.40x

  加速比: 1.1x (387.9s → 352.3s)
```

## 原始实验记录(reward 和 delta_state 标签分组后加性注入)-8/2划分训练集测试集
- epoch 0 测试结果：
```
Epoch 0: 4800 train / 1200 test (limited to 10000)
y 范围: reward [-1.645,3.373]  delta [-6.220,3.157]

=== Baseline: 12 x TabICLRegressor(num_outputs=1) ===
耗时: 5.4s

=== Joint: 1 x TabICLRegressor(num_outputs=12) ===
耗时: 1.1s

======================================================================
   Dim Label          Baseline MSE      Joint MSE Ratio(J/B)
  ------------------------------------------------------------------
     0 reward             0.005804       0.006104       1.05x
     1 delta[0]           0.000000       0.000000       1.11x
     2 delta[1]           0.000000       0.000000       1.14x
     3 delta[2]           0.000000       0.000000       4.86x
     4 delta[3]           0.000001       0.000000       0.55x
     5 delta[4]           0.000000       0.000000       1.50x
     6 delta[5]           0.000404       0.000463       1.15x
     7 delta[6]           0.000476       0.000545       1.15x
     8 delta[7]           0.004668       0.009336       2.00x
     9 delta[8]           0.001580       0.005394       3.41x
    10 delta[9]           0.010576       0.024797       2.34x
    11 delta[10]          0.012957       0.026304       2.03x
  ------------------------------------------------------------------
         REWARD MSE       0.005804       0.006104       1.05x
     DELTA(avg) MSE       0.002787       0.006076       2.18x
        OVERALL MSE       0.003039       0.006079       2.00x

  加速比: 5.0x (5.4s → 1.1s)
```

- epoch 100 测试结果：
```
Epoch 100: 84800 train / 10000 test (limited to 10000)
y 范围: reward [-1.645,5.945]  delta [-6.934,7.289]

=== Baseline: 12 x TabICLRegressor(num_outputs=1) ===
耗时: 97.8s

=== Joint: 1 x TabICLRegressor(num_outputs=12) ===
耗时: 77.5s

======================================================================
   Dim Label          Baseline MSE      Joint MSE Ratio(J/B)
  ------------------------------------------------------------------
     0 reward             0.001165       0.001305       1.12x
     1 delta[0]           0.000000       0.000000       1.79x
     2 delta[1]           0.000000       0.000000       0.99x
     3 delta[2]           0.000000       0.000000       1.65x
     4 delta[3]           0.000000       0.000000       1.58x
     5 delta[4]           0.000002       0.000002       0.91x
     6 delta[5]           0.000196       0.000214       1.09x
     7 delta[6]           0.000251       0.000306       1.22x
     8 delta[7]           0.002519       0.003005       1.19x
     9 delta[8]           0.000701       0.001427       2.03x
    10 delta[9]           0.002642       0.004389       1.66x
    11 delta[10]          0.006661       0.016914       2.54x
  ------------------------------------------------------------------
         REWARD MSE       0.001165       0.001305       1.12x
     DELTA(avg) MSE       0.001179       0.002387       2.02x
        OVERALL MSE       0.001178       0.002297       1.95x

  加速比: 1.3x (97.8s → 77.5s)
```

- epoch 280 测试结果：
```
Epoch 280: 228800 train / 10000 test (limited to 10000)
y 范围: reward [-1.645,6.318]  delta [-6.934,7.289]

=== Baseline: 12 x TabICLRegressor(num_outputs=1) ===
耗时: 535.0s

=== Joint: 1 x TabICLRegressor(num_outputs=12) ===
耗时: 477.3s

======================================================================
   Dim Label          Baseline MSE      Joint MSE Ratio(J/B)
  ------------------------------------------------------------------
     0 reward             0.001310       0.001382       1.06x
     1 delta[0]           0.000000       0.000000       2.04x
     2 delta[1]           0.000000       0.000000       0.89x
     3 delta[2]           0.000000       0.000000       1.57x
     4 delta[3]           0.000000       0.000000       1.30x
     5 delta[4]           0.000001       0.000001       1.16x
     6 delta[5]           0.000130       0.000130       1.00x
     7 delta[6]           0.000178       0.000192       1.08x
     8 delta[7]           0.001554       0.002003       1.29x
     9 delta[8]           0.000610       0.001255       2.06x
    10 delta[9]           0.002121       0.002354       1.11x
    11 delta[10]          0.005778       0.012470       2.16x
  ------------------------------------------------------------------
         REWARD MSE       0.001310       0.001382       1.06x
     DELTA(avg) MSE       0.000943       0.001673       1.77x
        OVERALL MSE       0.000974       0.001649       1.69x

  加速比: 1.1x (535.0s → 477.3s)
```

