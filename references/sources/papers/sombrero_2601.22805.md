# SOMBRERO: Measuring and Steering Boundary Placement in End-to-End Hierarchical Sequence Models

## Full Citation

**Title:** SOMBRERO: Measuring and Steering Boundary Placement in End-to-End Hierarchical Sequence Models
**arXiv ID:** 2601.22805
**Category:** cs.LG
**Year:** 2026

---

## Abstract Summary

SOMBRERO introduces a diagnostic framework for evaluating *where* a hierarchical model places chunk boundaries, independent of downstream task performance. The central claim: in a well-trained hierarchical model, boundaries should align with positions of **high next-token surprisal** (predictive difficulty). A model that places boundaries at "easy" positions is not leveraging the hierarchy — it is simply subsampling tokens uniformly.

---

## Key Metric: Boundary Enrichment (B)

The **Boundary Enrichment** metric measures how strongly boundary positions concentrate on high-surprisal tokens:

```
B = E[surprisal(t) | b_t = 1] / E[surprisal(t)]
```

where `surprisal(t)` is the negative log-probability of the next token given context, and `b_t = 1` indicates a boundary.

- **B > 1**: Boundaries preferentially land on hard-to-predict positions — semantically meaningful chunking.
- **B ≈ 1**: Boundaries are uniformly distributed — no semantic selectivity (equivalent to fixed stride).
- **B < 1**: Boundaries avoid hard positions — pathological inverse selection.

---

## Key Findings

1. **Randomly initialized hierarchical models** produce B ≈ 1 (no preference). Training drives B > 1 in well-converged models.

2. **Boundary entropy is correlated with B but not equivalent.** Low binary entropy (p_t near 0 or 1) is necessary but not sufficient — the router must also place confident boundaries *at the right positions*. A model can be confident (low entropy) while placing boundaries uniformly (B ≈ 1).

3. **CRL/recurrent encoders** tend to produce higher B than fixed-stride baselines. Attention-based encoders can match or exceed CRL if given sufficient capacity.

4. **Steering:** The paper also proposes auxiliary losses that reward B > 1 (boundaries at high-surprisal positions). Not used in our implementation — B is used as a diagnostic only.

---

## Relevance to Phase 2

| SOMBRERO concept | Phase 2 diagnostic |
|---|---|
| Boundary Enrichment B | Not currently computed — requires a base language model to estimate surprisal; could be approximated using the current model's own logits |
| Binary boundary entropy H(p_t) | `boundary_entropy()` in train.py — logged per step as `hdc/boundary_entropy` |
| "Confident and selective" as quality criterion | STE (H-Net Eq. 7) + H-Net Eq. 10 ratio loss both incentivize this |

**Interpretation guidance for Phase 2 results:**
- `boundary_entropy` decreasing over training = router becoming more decisive. Good.
- If `boundary_entropy` is low but `val_bpc` is not improving: check whether the model is placing boundaries at semantically meaningful positions (would require B computation).
- `outer_strided` will have B = 1 by construction — serves as the B = 1 lower bound.
- `outer_crl` (cosine rule) should produce B > 1 if the CRL encoder is creating meaningful position diversity.

---

## Sources

- arXiv:2601.22805 (primary)
- Companion to H-Net (arXiv:2507.07955) — uses H-Net as the evaluation testbed
