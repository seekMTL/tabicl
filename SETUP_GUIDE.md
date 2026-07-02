# 服务器项目搭建与远程仓库关联操作指南

## 背景

将从GitHub fork的仓库zip代码包上传到服务器后，需要完成以下目标：
1. 在服务器上搭建Git项目
2. 关联到个人fork的远程仓库
3. 实现本地开发与远程同步

## 环境信息

| 项目 | 值 |
|------|-----|
| 服务器路径 | `/home/lizitao/project/tabicl` |
| GitHub用户名 | seekMTL |
| 远程仓库 | `git@github.com:seekMTL/tabicl.git` |
| 分支 | main |

---

## 实际执行的完整流程

### 第一步：初始化Git仓库

```bash
cd /home/lizitao/project/tabicl
git init
```

### 第二步：配置本地Git用户信息

使用 `--local` 参数，只对当前仓库生效，不影响其他项目：

```bash
git config --local user.name "seekMTL"
git config --local user.email "3491889397@qq.com"
```

### 第三步：重命名默认分支为main

```bash
git branch -m main
```

### 第四步：添加远程仓库

**实际执行（HTTPS格式，后来发现问题）**：
```bash
git remote add origin https://github.com/seekMTL/tabicl.git
```

**推荐做法（直接使用SSH格式）**：
```bash
git remote add origin git@github.com:seekMTL/tabicl.git
```

### 第五步：添加文件并创建初始提交

```bash
git add .
git commit -m "Initial commit: 添加tabicl项目代码"
```

### 第六步：推送失败与排查

**第一次尝试推送（HTTPS格式）**：
```bash
git push -u origin main
# 失败：fatal: could not read Username for 'https://github.com'
```

**排查SSH连接**：
```bash
ssh -T git@github.com
# 成功：Hi seekMTL! You've successfully authenticated...
```

**发现问题**：远程URL是HTTPS格式，改为SSH格式：
```bash
git remote set-url origin git@github.com:seekMTL/tabicl.git
```

**第二次尝试推送**：
```bash
git push -u origin main
# 失败：! [rejected] main -> main (fetch first)
# error: 无法推送一些引用到 'github.com:seekMTL/tabicl.git'
# 提示：更新被拒绝，因为远程仓库包含您本地尚不存在的提交。这通常是因为另外
# 提示：一个仓库已向该引用进行了推送。再次推送前，您可能需要先整合远程变更
# 提示：（如 'git pull ...'）。
# 提示：详见 'git push --help' 中的 'Note about fast-forwards' 小节。
```

**发现问题**：远程仓库已有内容（fork时带过来的），需要处理

### 第七步：合并远程内容

```bash
git pull origin main --allow-unrelated-histories --no-rebase --no-edit
```

**参数说明**：
- `--allow-unrelated-histories`：允许合并不相关的历史（本地是新初始化的，和远程没有共同祖先）
- `--no-rebase`：使用合并策略，不使用变基
- `--no-edit`：不打开编辑器，直接使用默认的合并信息

### 第八步：推送

```bash
git push -u origin main --force
```

**说明**：执行完 `git pull` 合并后，本地已经是远程的"超集"，此时用 `git push` 和 `git push --force` 效果一样，都能成功。

---

## 最终结果

### Commit历史

```
13e4acd Merge branch 'main' of github.com:seekMTL/tabicl  （合并时自动生成）
aba7962 Initial commit: 添加tabicl项目代码                  （本地初始提交）
8f665ed n_threads is set to the maximum...                  （原始仓库提交）
...（原始仓库的其他提交记录）
```

### 仓库状态

```bash
git status
# 位于分支 main
# 您的分支与上游分支 'origin/main' 一致。
# 无文件要提交，干净的工作区

git remote -v
# origin	git@github.com:seekMTL/tabicl.git (fetch)
# origin	git@github.com:seekMTL/tabicl.git (push)
```

---

## 关键问题解答

### 问题1：为什么要先pull再push？

因为远程仓库（fork时带过来的）已有提交历史，而本地是新初始化的仓库，两者没有共同祖先。Git拒绝直接推送，要求先整合远程内容。

### 问题2：合并后用 `git push` 还是 `git push --force`？

**两者效果一样**。

执行 `git pull` 合并后：
- 本地分支包含远程的所有提交（是远程的"超集"）
- 推送时Git发现本地包含了远程的所有内容，是"fast-forward"操作
- 不需要强制推送也能成功

### 问题3：如果不想保留远程历史，正确的做法是什么？

**直接强制推送，不要先pull**：
```bash
# 不执行 git pull
git push -u origin main --force
```

这样远程仓库只有你自己的提交记录，没有原始仓库的历史。

### 问题4：两种方案对比

| 方案 | 操作步骤 | 效果 |
|------|---------|------|
| **保留远程历史** | `git pull` → `git push` | 保留原始仓库的所有提交记录，多一个merge记录 |
| **覆盖远程仓库** | 直接 `git push --force` | 丢弃远程历史，只有自己的提交，历史干净 |

---

## 问题排查记录

### 问题1：SSH连接失败（假象）

**现象**：
```
kex_exchange_identification: Connection closed by remote host
Connection closed by 198.18.0.39 port 22
```

**原因**：
- 198.18.0.x 是RFC 2544保留IP地址，网络中存在DNS劫持
- 但实际测试 `ssh -T git@github.com` 可以成功

**结论**：SSH配置正常，问题在于远程URL配置成了HTTPS格式

### 问题2：HTTPS认证失败

**现象**：
```
fatal: could not read Username for 'https://github.com': 没有那个设备或地址
```

**原因**：
- GitHub已不支持密码认证，需要Personal Access Token
- 服务器环境应直接使用SSH格式

**解决方案**：
```bash
git remote set-url origin git@github.com:seekMTL/tabicl.git
```

### 问题3：推送被拒绝

**现象**：
```
! [rejected] main -> main (fetch first)
error: 无法推送一些引用到 'github.com:seekMTL/tabicl.git'
```

**原因**：远程仓库已有内容，本地没有远程的提交历史

**解决方案**：
- 方案A：先合并再推送 `git pull` → `git push`
- 方案B：直接强制推送 `git push --force`

---

## 常用命令参考

### 查看状态
```bash
git status                    # 查看工作区状态
git remote -v                 # 查看远程仓库配置
git log --oneline             # 查看提交历史
git config --local --list     # 查看本地git配置
```

### SSH相关
```bash
ssh -T git@github.com         # 测试SSH连接到GitHub
cat ~/.ssh/id_ed25519.pub     # 查看公钥
```

### 远程仓库操作
```bash
# 修改远程URL格式
git remote set-url origin git@github.com:用户名/仓库名.git      # SSH格式
git remote set-url origin https://github.com/用户名/仓库名.git   # HTTPS格式

# 推送
git push -u origin main       # 正常推送并设置跟踪
git push -u origin main --force  # 强制推送（覆盖远程）
```

### 合并相关
```bash
# 拉取并合并（允许不相关历史）
git pull origin main --allow-unrelated-histories --no-rebase --no-edit
```

---

## 后续开发工作流

### 日常开发

```bash
# 1. 修改代码
# 2. 添加修改
git add .

# 3. 提交修改
git commit -m "描述你的修改"

# 4. 推送到远程
git push
```

### 同步上游仓库更新（如需要）

```bash
# 1. 添加上游仓库（只需执行一次）
git remote add upstream https://github.com/原作者/tabicl.git

# 2. 获取上游更新
git fetch upstream

# 3. 合并上游更新
git merge upstream/main

# 4. 推送到自己的fork
git push
```

---

## 注意事项

1. **服务器环境直接用SSH格式**：避免HTTPS认证问题
2. **本地配置优先**：使用 `--local` 参数配置git用户信息
3. **方案选择明确**：
   - 保留远程历史：先 `git pull` 再 `git push`
   - 覆盖远程仓库：直接 `git push --force`，不要先pull
4. **fork仓库独立**：推送到的是个人fork仓库，不影响上游原始仓库

---

## 操作记录

- **日期**: 2026-07-01
- **操作人**: seekMTL
- **服务器**: /home/lizitao/project/tabicl
- **实际采用方案**: 保留远程历史（先合并再推送）