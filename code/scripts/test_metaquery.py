"""GPU validation for MetaQuery conditioning (research_G).

Runs the 5 acceptance checks from the implementation spec on ONE shared GPU
(loads ~12-15GB alongside the running train_stream job; load nothing onto an
already-busy 80GB GPU beyond this). No training, no checkpoints touched.

Checks:
  1. processor returns mm_token_type_ids (HF 5.x; required by get_rope_index).
  2. language_model(output_hidden_states=True) -> len(hidden_states)==29, each [1,L+N,2048].
  3. grad flows to meta_query (non-None, finite) and to NO Qwen param (all .grad is None).
  4. instruction sensitivity: two different instructions on the same image ->
     query_hidden mean-abs-diff > 1e-3.
  5. visual_pos_masks extension does not break _deepstack_process; full forward runs.
Also reports peak GPU memory of one forward_metaquery.

Run on server:
  cd $WS/code && CUDA_VISIBLE_DEVICES=3 HF_HOME=$WS/hf_cache HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 $WS/.venv/bin/python scripts/test_metaquery.py
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from igsw.dynamics.conditioning import QwenVLEncoder  # noqa: E402


def banner(s):
    print("\n" + "=" * 70 + f"\n{s}\n" + "=" * 70, flush=True)


def main():
    assert torch.cuda.is_available(), "need a GPU"
    dev = "cuda"
    dtype = torch.bfloat16
    N = 64

    banner("load QwenVLEncoder (frozen)")
    enc = QwenVLEncoder().to(dev).eval()
    print(f"hidden_size={enc.hidden_size} num_layers={enc.num_layers} "
          f"image_token_id={enc.image_token_id}", flush=True)
    # meta_query as it lives in InstructGSWorldModel (nn.Parameter [N,2048])
    meta_query = torch.nn.Parameter(
        torch.randn(N, enc.hidden_size, device=dev, dtype=dtype) * 0.02)

    # a deterministic synthetic image (uint8 HxWx3) — shape mirrors AgiBot frames
    rng = np.random.RandomState(0)
    image = (rng.rand(434, 574, 3) * 255).astype("uint8")
    instr_a = "Pick up the red block from the table."
    instr_b = "Open the top drawer slowly."

    inputs_a = enc.build_inputs(instr_a, image)
    inputs_a = {k: (v.to(dev, dtype=dtype) if torch.is_tensor(v) and v.is_floating_point()
                    else (v.to(dev) if torch.is_tensor(v) else v))
                for k, v in inputs_a.items()}

    # ---------------------------------------------------------------- CHECK 1
    banner("CHECK 1 — processor output keys (mm_token_type_ids present?)")
    keys = list(inputs_a.keys())
    print("inputs.keys() =", keys, flush=True)
    has_mm = "mm_token_type_ids" in keys
    print(f"mm_token_type_ids present: {has_mm}", flush=True)
    L = int(inputs_a["input_ids"].shape[1])
    print(f"L (real tokens) = {L}; image_grid_thw = "
          f"{inputs_a.get('image_grid_thw').tolist() if 'image_grid_thw' in inputs_a else None}",
          flush=True)
    check1 = has_mm

    # ---------------------------------------------------------------- CHECK 2
    # Re-run the metaquery prep here (independent of forward_metaquery) and call
    # language_model directly to inspect the full hidden_states tuple length/shapes.
    banner("CHECK 2 — len(hidden_states)==29, each [1, L+N, 2048]")
    qm = enc.model.model
    with torch.no_grad():
        ie = qm.get_input_embeddings()(inputs_a["input_ids"].to(dev))
        iout = qm.get_image_features(inputs_a["pixel_values"].to(dev, dtype),
                                     inputs_a["image_grid_thw"].to(dev), return_dict=True)
        iemb = torch.cat(iout.pooler_output, dim=0).to(dev, dtype)
        dse = iout.deepstack_features
        imask, _ = qm.get_placeholder_mask(inputs_a["input_ids"].to(dev), inputs_embeds=ie,
                                           image_features=iemb)
        ie = ie.masked_scatter(imask, iemb)
        vpm = imask[..., 0]
        pos, _ = qm.get_rope_index(input_ids=inputs_a["input_ids"].to(dev),
                                   mm_token_type_ids=inputs_a["mm_token_type_ids"].to(dev),
                                   image_grid_thw=inputs_a["image_grid_thw"].to(dev),
                                   attention_mask=inputs_a["attention_mask"].to(dev))
        maxp = int(pos.max().item())
        qpos = (torch.arange(N, device=dev, dtype=pos.dtype) + maxp + 1)[None, None].expand(3, 1, N)
        pos_ext = torch.cat([pos, qpos], dim=2)
        vpm_ext = torch.cat([vpm, torch.zeros(1, N, dtype=torch.bool, device=dev)], dim=1)
        am_ext = torch.cat([inputs_a["attention_mask"].to(dev),
                            torch.ones(1, N, dtype=inputs_a["attention_mask"].dtype, device=dev)], dim=1)
        ie_ext = torch.cat([ie, meta_query.detach()[None].to(dtype)], dim=1)
        out = qm.language_model(input_ids=None, inputs_embeds=ie_ext, attention_mask=am_ext,
                                position_ids=pos_ext, visual_pos_masks=vpm_ext,
                                deepstack_visual_embeds=dse, output_hidden_states=True,
                                use_cache=False)
    hs = out.hidden_states
    n_hs = len(hs)
    shape0 = tuple(hs[0].shape)
    shape_last = tuple(hs[-1].shape)
    print(f"len(hidden_states) = {n_hs} (expect {enc.num_layers + 1})", flush=True)
    print(f"hidden_states[0].shape  = {shape0}", flush=True)
    print(f"hidden_states[-1].shape = {shape_last}", flush=True)
    all_shapes_ok = all(tuple(h.shape) == (1, L + N, enc.hidden_size) for h in hs)
    print(f"all entries == [1, {L + N}, {enc.hidden_size}]: {all_shapes_ok}", flush=True)
    # also print the position ranges to confirm no collision (queries start at max+1)
    print(f"text/img max_pos = {maxp}; query positions = "
          f"[{maxp + 1} .. {maxp + N}] (text-like, identical on all 3 axes)", flush=True)
    check2 = (n_hs == enc.num_layers + 1) and all_shapes_ok
    del out, hs, ie_ext, ie, iemb, iout, dse
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------- CHECK 3
    banner("CHECK 3 — grad to meta_query (finite), NO grad to any Qwen param")
    torch.cuda.reset_peak_memory_stats(dev)
    if meta_query.grad is not None:
        meta_query.grad = None
    for p in enc.parameters():
        p.grad = None
    # WITH grad (forward_metaquery has NO @no_grad — caller controls context)
    query_hidden = enc.forward_metaquery(inputs_a, meta_query)   # [28,N,2048]
    print(f"query_hidden.shape = {tuple(query_hidden.shape)} dtype={query_hidden.dtype} "
          f"requires_grad={query_hidden.requires_grad}", flush=True)
    shape_ok = tuple(query_hidden.shape) == (enc.num_layers, N, enc.hidden_size)
    peak_fwd = torch.cuda.max_memory_allocated(dev) / 1e9
    loss = query_hidden.float().sum()
    loss.backward()
    mq_grad_ok = (meta_query.grad is not None
                  and torch.isfinite(meta_query.grad).all().item()
                  and meta_query.grad.abs().max().item() > 0)
    mq_gnorm = meta_query.grad.norm().item() if meta_query.grad is not None else float("nan")
    print(f"meta_query.grad: non-None={meta_query.grad is not None} "
          f"finite={torch.isfinite(meta_query.grad).all().item() if meta_query.grad is not None else False} "
          f"|grad|_2={mq_gnorm:.4e} max={meta_query.grad.abs().max().item() if meta_query.grad is not None else 0:.4e}",
          flush=True)
    qwen_with_grad = [n for n, p in enc.named_parameters() if p.grad is not None]
    n_qwen_params = sum(1 for _ in enc.parameters())
    print(f"Qwen params with non-None .grad: {len(qwen_with_grad)} / {n_qwen_params} "
          f"(expect 0)", flush=True)
    if qwen_with_grad:
        print("  OFFENDERS (first 5):", qwen_with_grad[:5], flush=True)
    check3 = mq_grad_ok and (len(qwen_with_grad) == 0) and shape_ok
    peak_total = torch.cuda.max_memory_allocated(dev) / 1e9
    print(f"peak GPU mem: forward={peak_fwd:.2f} GB, forward+backward={peak_total:.2f} GB", flush=True)

    # ---------------------------------------------------------------- CHECK 4
    banner("CHECK 4 — instruction sensitivity (diff > 1e-3)")
    inputs_b = enc.build_inputs(instr_b, image)
    inputs_b = {k: (v.to(dev, dtype=dtype) if torch.is_tensor(v) and v.is_floating_point()
                    else (v.to(dev) if torch.is_tensor(v) else v))
                for k, v in inputs_b.items()}
    with torch.no_grad():
        qh_a = enc.forward_metaquery(inputs_a, meta_query)
        qh_b = enc.forward_metaquery(inputs_b, meta_query)
    diff = (qh_a.float() - qh_b.float()).abs().mean().item()
    # per-layer breakdown (early vs late) to confirm the text influence propagates
    pl = [(qh_a[j].float() - qh_b[j].float()).abs().mean().item() for j in (0, 13, 27)]
    print(f"mean |query_hidden(A) - query_hidden(B)| = {diff:.4e}", flush=True)
    print(f"  per-layer diff  L0={pl[0]:.3e}  L13={pl[1]:.3e}  L27={pl[2]:.3e}", flush=True)
    check4 = diff > 1e-3

    # ---------------------------------------------------------------- CHECK 5
    banner("CHECK 5 — deepstack + full forward end-to-end (no shape error)")
    # forward_metaquery already exercises _deepstack_process at layers 0,1,2 with the
    # extended visual_pos_masks; if it returned a [28,N,2048] tensor above, deepstack
    # did not raise. Re-run once more cleanly and confirm finite output.
    with torch.no_grad():
        qh_final = enc.forward_metaquery(inputs_a, meta_query)
    finite = torch.isfinite(qh_final).all().item()
    print(f"end-to-end forward_metaquery: shape={tuple(qh_final.shape)} "
          f"finite={finite} (deepstack at layers 0/1/2 did not raise)", flush=True)
    check5 = finite and tuple(qh_final.shape) == (enc.num_layers, N, enc.hidden_size)

    # ---------------------------------------------------------------- SUMMARY
    banner("SUMMARY")
    res = {
        "1 mm_token_type_ids present": check1,
        "2 hidden_states==29, shapes": check2,
        "3 grad->meta_query, none->Qwen": check3,
        "4 instruction sensitivity": check4,
        "5 deepstack/full forward": check5,
    }
    for k, v in res.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}", flush=True)
    print(f"\npeak GPU memory (one forward_metaquery, fwd only): {peak_fwd:.2f} GB", flush=True)
    print(f"peak GPU memory (fwd + backward):                  {peak_total:.2f} GB", flush=True)
    allok = all(res.values())
    print(f"\n{'[ALL 5 CHECKS PASS]' if allok else '[SOME CHECKS FAILED]'}", flush=True)
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
