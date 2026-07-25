# 第三方源码版本

同步时没有复制嵌套 `.git/` 对象；下列 commit 用于恢复各依赖的版本。

| 目录 | Origin | Commit |
|---|---|---|
| CUT3R | https://github.com/CUT3R/CUT3R.git | `8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf` |
| GPSToken | https://github.com/xtudbxk/GPSToken.git | `372231d1de74cf759dc20cef12455265010d803e` |
| LIBERO | https://github.com/Lifelong-Robot-Learning/LIBERO.git | `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| Pi3 | https://github.com/yyfz/Pi3.git | `9fa3ddb3f8d53041f8b2738df404f62223bbaa7b` |
| St4RTrack | https://github.com/HavenFeng/St4RTrack.git | `0f9a3f44a7ebac76600cd31ec9eea5228ad7db91` |
| co-tracker | https://github.com/facebookresearch/co-tracker.git | `82e02e8029753ad4ef13cf06be7f4fc5facdda4d` |
| diff-gaussian-rasterization-w-depth | https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth | `cb65e4b86bc3bd8ed42174b72a62e8d3a3a71110` |
| monst3r | https://github.com/Junyi42/monst3r.git | `574cc77ad278bad582f470e5382624e01f8769a7` |
| vggt | https://github.com/facebookresearch/vggt.git | `a288dd0f14786c93483e45524328726ab7b1b4ce` |

`Dynamic3DGaussians` 在远端没有 `.git/` 元数据，因此这里只保留其源码快照。

远端显示为 dirty 的 GPSToken、St4RTrack 和 co-tracker 只有未跟踪的缓存、权重或生成目录；这些内容均未同步，没有发现已跟踪源码修改。
