# Instruct-GS-World Source Recovery Manifest

## Recovery inputs

- Source-side backup: `/Users/hela/Downloads/instruct_gs_world/`
- Existing authoritative checkout: `/Users/hela/Instruct-GS-World/`
- Isolated recovered checkout: `/Users/hela/Instruct-GS-World-recovered-20260725/`
- Git base: branch `gpstoken-2dgs`, commit `8f9ba789678b82b4590c8496342fc36da2fabc6e`
- Recovery branch: `codex/restore-source-backup-20260725`

The source-side backup was copied on 2026-07-25 from
`/root/xuhaoming/public/instruct_gs_world` on SSH host
`Dev1_BaiduqiyuanA100`. It intentionally excludes datasets, checkpoints,
runtime logs, generated media, virtual environments, Git metadata, and files
larger than 10 MB.

## Merge policy

The recovered checkout is a non-destructive union. Existing local-only files
were retained, and source-backup-only files were restored at their original
paths.

- `/Users/hela/Downloads/instruct_gs_world/code/igsw/model.py` was selected for
  `/Users/hela/Instruct-GS-World-recovered-20260725/code/igsw/model.py` because
  it contains the source-side entity-level SE(3) implementation absent from
  the existing checkout.
- `/Users/hela/Instruct-GS-World/code/scripts/gen_libero_pi3_v2.sh` was retained
  because it is newer and preserves the held-seed generation branch removed
  from the backup copy.
- `/Users/hela/Instruct-GS-World/agent.md` was retained because the backup is
  an exact 150-line-shorter prefix of the existing document.
- Existing local-only source and research notes were retained. No file was
  deleted from the existing authoritative checkout.

## Third-party boundary

The source snapshots under
`/Users/hela/Instruct-GS-World-recovered-20260725/third_party/` were restored
locally but remain ignored by the main Git repository. Their upstream origins
and pinned commits are recorded in
`/Users/hela/Instruct-GS-World-recovered-20260725/THIRD_PARTY_REVISIONS.md`.
This avoids vendoring approximately 245,000 lines of upstream code into the
first-party recovery commit.

## Pre-import inventory

- Source-side backup first-party code: 535 files, 87,704 lines.
- Existing authoritative checkout code: 493 files, 80,079 lines.
- Identical common files: 485.
- Backup-only source files: 48 files, 7,963 lines.
- Existing-checkout-only source files: 6 files, 393 lines.
- Common source paths with different content: 2.

This manifest records source recovery only. It is not evidence that remote
datasets, checkpoints, experiment outputs, or Python environments were
recovered.
