# GenieX 手机验收实验

从 `/home/lyyyy/edge_side_work/geniex_validation` 整体迁入本目录，旧位置保留兼容链接。

- `run_phone_smoke.py`：通过 ADB 调用手机上已部署的 GenieX smoke 脚本，保存原始日志、输出及 SDK JSON。
- `CURRENT_RUN.txt`：早期 smoke 结果目录。
- `CURRENT_ROUNDTRIP.txt`：猫/狗图片相关历史验证目录。
- `results/`：保留的历史验收结果。

```bash
cd /home/lyyyy/edge_side_work/code/geniex_validation
../android_world_phone/.venv/bin/python run_phone_smoke.py --help
```

该脚本默认手机部署目录为 `/data/local/tmp/qwen3_vl_geniex_sm8750`，对应早期验收配置。当前三档16K包验收与官方M3A实验入口位于 [android_world_phone](../android_world_phone/README.md)。
