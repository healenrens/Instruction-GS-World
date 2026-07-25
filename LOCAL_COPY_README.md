# Instruct-GS-World 本地代码镜像

本目录于 2026-07-25 从以下位置同步：

- SSH：`Dev1_BaiduqiyuanA100`
- 远端：`/root/xuhaoming/public/instruct_gs_world`

这是代码与文档镜像，不是远端 296GB 工作目录的完整备份。远端原目录没有被移动、删除或改写。

## 已保留

- 主项目源码：`code/`
- 脚本：`scripts/`、根目录 `*.sh`
- 文档与研究记录：根目录 `*.md`、`docs/`、`notes/`
- 第三方依赖的源码和文本说明：`third_party/`
- 少量远端已有的源码暂存/备份目录

## 未同步

- `data/`、`checkpoints/`、`outputs/`、`logs/`、`eval/`、`viz/`
- `.venv/`、`.uv_cache/`、`__pycache__/`、`.git/`
- 模型权重、实验输出、图片、视频、二进制数据与日志
- 超过 10MB 的单个文件

第三方仓库的来源和固定版本见 `THIRD_PARTY_REVISIONS.md`。
