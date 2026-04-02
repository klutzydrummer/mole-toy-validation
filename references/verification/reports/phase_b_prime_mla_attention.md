# Phase B' Verification Report: mla_attention.md

**Date:** 2026-04-01 (updated; original 2026-03-31)
**Verdict:** PASS

---

## Equations

| Claim | Source | Status | Notes |
|-------|--------|--------|-------|
| Eq. 9: `c_t^{KV} = W^{DKV} h_t` | `mla_deepseek_v2_2405.04434.md` line 74 | VERIFIED | Verbatim match |
| Eq. 10: `k_t^C = W^{UK} c_t^{KV}` | `mla_deepseek_v2_2405.04434.md` line 75 | VERIFIED | Verbatim match |
| Eq. 11: `v_t^C = W^{UV} c_t^{KV}` | `mla_deepseek_v2_2405.04434.md` line 76 | VERIFIED | Verbatim match |
| Eq. 12: `c_t^Q = W^{DQ} h_t` | `mla_deepseek_v2_2405.04434.md` line 96 | VERIFIED | Verbatim match |
| Eq. 13: `q_t^C = W^{UQ} c_t^Q` | `mla_deepseek_v2_2405.04434.md` line 97 | VERIFIED | Verbatim match |

---

## Implementation claims

| Claim | Line | Status | Notes |
|-------|------|--------|-------|
| `W_DKV`: `nn.Linear(d, d_c, bias=False)` | 116 | VERIFIED | Exact match |
| `W_UK`: `nn.Linear(d_c, n_heads * self.d_head, bias=False)` | 117 | VERIFIED | Exact match |
| `W_UV`: `nn.Linear(d_c, n_heads * self.d_head, bias=False)` | 118 | VERIFIED | Exact match |
| `W_DQ`: `nn.Linear(d, d_c_q, bias=False)` | 121 | VERIFIED | Exact match |
| `W_UQ`: `nn.Linear(d_c_q, n_heads * self.d_head, bias=False)` | 122 | VERIFIED | Exact match |
| `d_c = d // 4` default | 110 | VERIFIED | Exact match; comment "(128 at d=512)" present |
| `d_c_q = d // 2` default | 112 | VERIFIED | Exact match; comment "(256 at d=512)" present |
| KV path at lines 134–137: `c_kv = self.W_DKV(x)`, `k = self.W_UK(c_kv).reshape(...)`, `v = self.W_UV(c_kv).reshape(...)` | 134–137 | VERIFIED | Code matches spec snippet exactly; both K and V read from same `c_kv` |
| Q path at lines 139–141: `c_q = self.W_DQ(x)`, `q = self.W_UQ(c_q).reshape(...)` | 139–141 | VERIFIED | Exact match (note: blank comment line at 139, actual code at 140–141) |
| `apply_rope` on `q` and `k` at lines 143–144 | 143–144 | VERIFIED | Both calls present; `v` not rotated |
| SDPA and output at lines 146–148: `F.scaled_dot_product_attention(q, k, v, is_causal=True)`, `out.transpose(1,2).reshape(B, L, D)`, `return self.out(out)` | 146–148 | VERIFIED | Exact match |
| Class spans lines 81–148 | 81–148 | VERIFIED | `class MLACausalAttention` starts at 81; last line of `forward` is 148 |

---

## Intentional deviations

| Deviation | Status | Notes |
|-----------|--------|-------|
| No RMSNorm on `c_t^{KV}` or `c_t^Q` | VERIFIED | Paper (`mla_deepseek_v2_2405.04434.md` lines 105, 204–205) explicitly states "An RMSNorm is applied after c_t^Q (and c_t^{KV}) for training stability." Reference code (`mla_attention.py` lines 90, 96) confirms: `self.q_norm = nn.RMSNorm(...)` and `self.kv_norm = nn.RMSNorm(...)`. Our code has neither; deviation is accurately documented and code confirms absence. |
| No decoupled RoPE (Eq. 14–19) | VERIFIED | Paper source (`mla_deepseek_v2_2405.04434.md` lines 116–136) contains full decoupled RoPE equations 14–19. Our code at lines 143–144 applies standard `apply_rope` to the full Q and K tensors. Deviation accurately described. |
| `d_c = d//4 = 128` (not `4·d_h`) | VERIFIED | Paper (`mla_deepseek_v2_2405.04434.md` line 83) states "d_c = 4 d_h for DeepSeek-V2, i.e., 512". At toy scale d=512, n_heads=8, d_head=64: `4·d_h = 256`, not 128. Our choice `d//4 = 128` is a different scaling rule (d//4 vs 4·d_h). The spec correctly notes the deviation is a toy-scale adaptation. Numerically: spec says "d_c = d//4 = 128" at d=512, which is accurate. |

---

## Verification checklist

| Item | Status | Notes |
|------|--------|-------|
| 1. KV shared latent: `W_DKV` single linear `d → d_c` at line 116; both `W_UK` and `W_UV` read from same `c_kv` at line 135 | VERIFIED | Line 116: `self.W_DKV = nn.Linear(d, d_c, bias=False)`. Lines 135–137: `c_kv = self.W_DKV(x)`, then `self.W_UK(c_kv)` and `self.W_UV(c_kv)`. |
| 2. Q separate latent: `W_DQ` independent of `W_DKV` at line 121 | VERIFIED | Line 121: `self.W_DQ = nn.Linear(d, d_c_q, bias=False)` — entirely separate parameter. |
| 3. `d_c` defaults at lines 110, 112 | VERIFIED | Line 110: `d_c = d // 4`. Line 112: `d_c_q = d // 2`. |
| 4. No shared parameters between K and V paths | VERIFIED | Lines 117–118: `self.W_UK` and `self.W_UV` are independent `nn.Linear` instances. |
| 5. RoPE on both Q and K | VERIFIED | Lines 143–144: `apply_rope` called on both `q` and `k`. `v` is not rotated. |
| 6. Output shape correctness | VERIFIED | Line 147: `out.transpose(1, 2).reshape(B, L, D)`. With `D = d = n_heads * d_head`, produces `[B, L, D]`. |
| 7. No RMSNorm on latents | VERIFIED | No `RMSNorm` or `LayerNorm` call on `c_kv` or `c_q` in forward pass (lines 134–148). |
| 8. Parameter count: 458,752 | VERIFIED | Arithmetic: W_DKV(512×128=65,536) + W_UK(128×512=65,536) + W_UV(128×512=65,536) + W_DQ(512×256=131,072) + W_UQ(256×512=131,072) = 458,752. Also: `self.out = nn.Linear(d, d, bias=False)` adds 512²=262,144 params to total layer count, but the spec's 458,752 figure covers only the five compression projections (not the output projection), which is consistent with the spec's intent to measure "KV/Q compression" params. |

---

## Issues found

None. The reference snippet field name issue (paraphrased `kv_a_proj`/`kv_b_proj` vs source
`wkv_a`/`wkv_b`) identified in the original report has been corrected in the spec (2026-04-01).
The spec's "Reference implementation" section now uses verbatim field names from
`mla_attention.py` lines 127–146 with explicit line citations.

---

## Summary

All equations (9–13) match the primary source verbatim. All implementation claims at the cited
line numbers are accurate. All intentional deviations are present in the code and accurately
described. The parameter count arithmetic is correct. All 8 verification checklist items pass.
Reference snippet corrected to use verbatim field names from the source.

**Overall verdict: PASS**
