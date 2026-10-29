"""
verify_fusion.py -- Phase 5.1-5.3 verification
================================================
Standalone CPU-only test script for belief_fusion.py and dms.py.
No model checkpoints required.

Run locally:
    python scripts/verify_fusion.py

Exits 0 if all tests pass, 1 if any fail.
"""

import sys
from pathlib import Path

# Allow running from repo root or from scripts/
repo_root = Path(__file__).resolve().parent.parent
edtc_utils = repo_root / "resources" / "EDTC" / "lib" / "utils"
sys.path.insert(0, str(edtc_utils))

import torch
from belief_fusion import abf, compute_beliefs, compute_conflict, dbf, discount
from dms import dms

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "[PASS]"
FAIL = "[FAIL]"
_failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    tag = PASS if cond else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  {tag} {name}{suffix}")
    if not cond:
        _failures.append(name)


def allclose(a: torch.Tensor, b: torch.Tensor, atol: float = 1e-5) -> bool:
    return torch.allclose(a.float(), b.float(), atol=atol)


def section(title: str) -> None:
    print(f"\n{'-' * 60}")
    print(f"  {title}")
    print(f"{'-' * 60}")


def make_opinion(b_vals: list[float]) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (b, u) from explicit belief values; u = 1 - sum(b)."""
    b = torch.tensor(b_vals)
    u = 1.0 - b.sum()
    return b, u


# ===========================================================================
# Section 1 -- ABF correctness
# ===========================================================================

section("ABF -- correctness")

# 1a. Identity: vacuous RGB (u=1,b=0) + dogmatic IR (u=0, b sums to 1)
#     => b_fused should equal b_th exactly.
#     NOTE: This identity only holds when the certain stream has u=0.
#     With u_rgb=1, u_th>0: b_fused = b_th/(1+u_th) != b_th.
b_th_dog, _ = make_opinion([0.9, 0.1])   # dogmatic IR: u=0
u_th_dog = torch.tensor(0.0)
b_rgb_vac, u_rgb_vac = torch.zeros(2), torch.tensor(1.0)
b_f_id, u_f_id = abf(b_rgb_vac, u_rgb_vac, b_th_dog, u_th_dog)
check("ABF identity: vacuous RGB + dogmatic IR -> b_fused=b_th",
      allclose(b_f_id, b_th_dog) and allclose(u_f_id, u_th_dog))

# 1b. Symmetry: swapping streams must not change result
b_a, u_a = make_opinion([0.5, 0.3])
b_b, u_b = make_opinion([0.7, 0.1])
bf_ab, uf_ab = abf(b_a, u_a, b_b, u_b)
bf_ba, uf_ba = abf(b_b, u_b, b_a, u_a)
check("ABF symmetry (swap rgb/th -> same result)",
      allclose(bf_ab, bf_ba) and allclose(uf_ab, uf_ba))

# 1c. Partition of unity: b_fused.sum(-1) + u_fused ~= 1
torch.manual_seed(0)
b_r = torch.rand(8, 2); b_r = b_r / (b_r.sum(-1, keepdim=True) + 0.5)
u_r = 1.0 - b_r.sum(-1)
b_t = torch.rand(8, 2); b_t = b_t / (b_t.sum(-1, keepdim=True) + 0.5)
u_t = 1.0 - b_t.sum(-1)
bf, uf = abf(b_r, u_r, b_t, u_t)
pou = bf.sum(-1) + uf
check("ABF partition of unity (sum_b + u ~= 1)",
      allclose(pou, torch.ones(8)),
      f"max_err={(pou - 1).abs().max():.2e}")

# 1d. Both vacuous -> u_fused = 1
b_vac = torch.zeros(2)
b_f2, u_f2 = abf(b_vac, torch.tensor(1.0), b_vac, torch.tensor(1.0))
check("ABF both vacuous -> u_fused=1, b_fused=0",
      allclose(u_f2, torch.tensor(1.0)) and allclose(b_f2, torch.zeros(2)))

# 1e. Consistent modalities: both certain about UAV -> fused also UAV-certain
b_uav_cert = torch.tensor([1.0, 0.0])
u_cert = torch.tensor(0.0)
bf_cc, uf_cc = abf(b_uav_cert, u_cert, b_uav_cert, u_cert)
check("ABF consistent UAV opinions -> fused is UAV-certain",
      allclose(bf_cc, b_uav_cert) and allclose(uf_cc, u_cert))


# ===========================================================================
# Section 2 -- compute_conflict
# ===========================================================================

section("compute_conflict")

# 2a. Zero conflict: dogmatic identical UAV opinions (both b=[1,0])
#     Non-dogmatic identical opinions have C>0 because of cross products.
b_dog = torch.tensor([1.0, 0.0])
C_dog = compute_conflict(b_dog, b_dog)
check("Conflict C=0 for dogmatic identical opinions (b=[1,0])",
      allclose(C_dog, torch.tensor(0.0)),
      f"C={C_dog.item():.4f}")

# 2b. Non-zero conflict for same non-dogmatic opinion (expected by formula)
#     b=[0.8, 0.1] sum=0.9; C = 0.9^2 - (0.64+0.01) = 0.81-0.65 = 0.16
b_same, _ = make_opinion([0.8, 0.1])
C_same = compute_conflict(b_same, b_same)
C_same_expected = b_same.sum()**2 - (b_same**2).sum()
check("Conflict formula correct for non-dogmatic identical opinions",
      allclose(C_same, C_same_expected),
      f"C={C_same.item():.4f}, expected={C_same_expected.item():.4f}")

# 2c. Total conflict: perfectly opposed certain opinions
b_uav = torch.tensor([1.0, 0.0])
b_bg  = torch.tensor([0.0, 1.0])
C_max = compute_conflict(b_uav, b_bg)
check("Conflict C=1 for perfectly opposed dogmatic opinions",
      allclose(C_max, torch.tensor(1.0)),
      f"C={C_max.item():.4f}")

# 2d. Partial conflict: manual cross-product check (K=2)
b1 = torch.tensor([0.6, 0.2])
b2 = torch.tensor([0.1, 0.7])
C_part = compute_conflict(b1, b2)
C_expected = 0.6 * 0.7 + 0.2 * 0.1   # manual sum of off-diag products
check("Conflict partial case (manual K=2 cross-product check)",
      allclose(C_part, torch.tensor(C_expected)),
      f"C={C_part.item():.4f}, expected={C_expected:.4f}")

# 2e. C in [0, 1] for random normalised inputs
torch.manual_seed(1)
b_x = torch.rand(100, 2); b_x = b_x / b_x.sum(-1, keepdim=True).clamp(1.0)
b_y = torch.rand(100, 2); b_y = b_y / b_y.sum(-1, keepdim=True).clamp(1.0)
C_rnd = compute_conflict(b_x, b_y)
check("Conflict C in [0,1] for random normalised inputs",
      bool((C_rnd >= 0).all() and (C_rnd <= 1).all()),
      f"range=[{C_rnd.min():.3f}, {C_rnd.max():.3f}]")


# ===========================================================================
# Section 3 -- discount
# ===========================================================================

section("discount")

# 3a. C=0 -> eta=1, opinion unchanged
b0, u0 = make_opinion([0.6, 0.2])
C0 = torch.tensor(0.0)
bt0, ut0 = discount(b0, u0, C0, lam=1.0)
check("Discount C=0 -> opinion unchanged",
      allclose(bt0, b0) and allclose(ut0, u0))

# 3b. C=1 -> eta=0, all belief collapsed to vacuity
b1v, u1v = make_opinion([0.6, 0.2])
C1 = torch.tensor(1.0)
bt1, ut1 = discount(b1v, u1v, C1, lam=1.0)
check("Discount C=1 -> b_tilde=0 and u_tilde=1",
      allclose(bt1, torch.zeros_like(bt1)) and allclose(ut1, torch.tensor(1.0)),
      f"bt1={[round(x,4) for x in bt1.tolist()]}, ut1={ut1.item():.4f}")

# 3c. Partition of unity preserved after discounting
b_d, u_d = make_opinion([0.5, 0.3])
for c_val in [0.0, 0.3, 0.7, 1.0]:
    C_v = torch.tensor(c_val)
    bt_v, ut_v = discount(b_d, u_d, C_v, lam=1.0)
    pou_v = bt_v.sum() + ut_v
    check(f"Discount partition of unity at C={c_val}",
          allclose(pou_v, torch.tensor(1.0)),
          f"sum_b_tilde + u_tilde = {pou_v.item():.6f}")

# 3d. Closed-form check (lam=1): u_tilde = u + C*(1 - u)
b_cf, u_cf = make_opinion([0.4, 0.3])
C_cf = torch.tensor(0.5)
_, ut_cf = discount(b_cf, u_cf, C_cf, lam=1.0)
ut_expected = u_cf + C_cf * (1.0 - u_cf)
check("Discount lam=1 closed form: u_tilde = u + C*(1-u)",
      allclose(ut_cf, ut_expected),
      f"got={ut_cf.item():.5f}, expected={ut_expected.item():.5f}")


# ===========================================================================
# Section 4 -- DBF correctness
# ===========================================================================

section("DBF -- correctness")

# 4a. Single-class opinions (C=0 exactly) -> DBF == ABF
#     C=0 only when beliefs are concentrated on a single class: b=[1,0] or b=[0,1].
#     For multi-class beliefs (e.g. b=[0.9,0.1]), C>0 due to cross-products.
b_single = torch.tensor([1.0, 0.0])   # only UAV class has belief; C=0
u_single = torch.tensor(0.0)
bf_dbf_id, uf_dbf_id = dbf(b_single, u_single, b_single, u_single, lam=1.0)
bf_abf_id, uf_abf_id = abf(b_single, u_single, b_single, u_single)
C_check = compute_conflict(b_single, b_single)
check("DBF with C=0 (single-class beliefs b=[1,0]) == ABF",
      allclose(bf_dbf_id, bf_abf_id) and allclose(uf_dbf_id, uf_abf_id),
      f"C={C_check.item():.4f}")

# 4b. Full conflict: C->1 => u_fused->1
b_uav2 = torch.tensor([1.0, 0.0])
b_bg2  = torch.tensor([0.0, 1.0])
u_cert = torch.tensor(0.0)
bf_conf, uf_conf = dbf(b_uav2, u_cert, b_bg2, u_cert, lam=1.0)
check("DBF full conflict -> u_fused -> 1",
      allclose(uf_conf, torch.tensor(1.0), atol=1e-4),
      f"u_fused={uf_conf.item():.6f}")
check("DBF full conflict -> b_fused -> 0",
      allclose(bf_conf, torch.zeros(2), atol=1e-4),
      f"b_fused={[round(x,6) for x in bf_conf.tolist()]}")

# 4c. Partition of unity
torch.manual_seed(2)
b_r3 = torch.rand(32, 2); b_r3 = b_r3 / (b_r3.sum(-1, keepdim=True) + 0.5)
u_r3 = 1.0 - b_r3.sum(-1)
b_t3 = torch.rand(32, 2); b_t3 = b_t3 / (b_t3.sum(-1, keepdim=True) + 0.5)
u_t3 = 1.0 - b_t3.sum(-1)
bf3, uf3 = dbf(b_r3, u_r3, b_t3, u_t3, lam=1.0)
pou3 = bf3.sum(-1) + uf3
check("DBF partition of unity (sum_b + u ~= 1)",
      allclose(pou3, torch.ones(32)),
      f"max_err={(pou3 - 1).abs().max():.2e}")

# 4d. Monotone sweep: u_fused increases as cross-stream conflict increases
#     Fix RGB opinion at [0.9, 0.05]. Sweep IR from same -> opposed.
b_fix = torch.tensor([0.9, 0.05]); u_fix = torch.tensor(0.05)
u_prev = -1.0
monotone = True
steps = torch.linspace(0.0, 0.85, 10)
for alpha_val in steps:
    # Blend IR opinion from aligned ([0.9,0.05]) toward opposed ([0.05,0.9])
    b_ir_sweep = (1 - alpha_val) * torch.tensor([0.9, 0.05]) + alpha_val * torch.tensor([0.05, 0.9])
    u_ir_sweep = torch.tensor(0.05)
    _, uf_sw = dbf(b_fix, u_fix, b_ir_sweep, u_ir_sweep, lam=1.0)
    if uf_sw.item() < u_prev - 1e-4:
        monotone = False
        break
    u_prev = uf_sw.item()
check("DBF u_fused monotone increases with growing conflict", monotone)

# 4e. Symmetry
b_p, u_p = make_opinion([0.5, 0.3])
b_q, u_q = make_opinion([0.2, 0.6])
bf_pq, uf_pq = dbf(b_p, u_p, b_q, u_q)
bf_qp, uf_qp = dbf(b_q, u_q, b_p, u_p)
check("DBF symmetry (swap rgb/th -> same result)",
      allclose(bf_pq, bf_qp) and allclose(uf_pq, uf_qp))


# ===========================================================================
# Section 5 -- Shape tests: SiamCAR spatial path (B, K, H, W)
# ===========================================================================

section("Shape tests -- SiamCAR spatial path (B,K,H,W)")

torch.manual_seed(3)
evi_ir  = torch.randn(1, 2, 17, 17)
evi_rgb = torch.randn(1, 2, 17, 17)

b_ir,  u_ir   = compute_beliefs(evi_ir,  K=2)
b_rgb2, u_rgb2 = compute_beliefs(evi_rgb, K=2)

check("compute_beliefs b shape == (1,2,17,17)", b_ir.shape == (1, 2, 17, 17))
check("compute_beliefs u shape == (1,17,17)",   u_ir.shape == (1, 17, 17))
check("compute_beliefs b >= 0",   bool((b_ir >= 0).all()))
check("compute_beliefs b < 1",    bool((b_ir < 1 + 1e-5).all()))
check("compute_beliefs u in (0,1]",
      bool((u_ir > 0).all() and (u_ir <= 1 + 1e-5).all()))
check("compute_beliefs partition of unity (b.sum(dim=1) + u ~= 1)",
      allclose(b_ir.sum(dim=1) + u_ir, torch.ones(1, 17, 17)))

# For abf/dbf: class dim must be LAST -> permute (B,K,H,W) -> (B,H,W,K)
b_ir_p  = b_ir.permute(0, 2, 3, 1)    # (1, 17, 17, 2)
u_ir_p  = u_ir                          # (1, 17, 17)
b_rgb_p = b_rgb2.permute(0, 2, 3, 1)
u_rgb_p = u_rgb2

bf_sp, uf_sp = dbf(b_rgb_p, u_rgb_p, b_ir_p, u_ir_p, lam=1.0)
check("DBF spatial b_fused shape == (1,17,17,2)", bf_sp.shape == (1, 17, 17, 2))
check("DBF spatial u_fused shape == (1,17,17)",   uf_sp.shape == (1, 17, 17))
check("DBF spatial u_fused in (0,1]",
      bool((uf_sp > 0).all() and (uf_sp <= 1 + 1e-5).all()))
pou_sp = bf_sp.sum(-1) + uf_sp
check("DBF spatial partition of unity",
      allclose(pou_sp, torch.ones(1, 17, 17)),
      f"max_err={(pou_sp - 1).abs().max():.2e}")


# ===========================================================================
# Section 6 -- Shape tests: YOLO scalar path (floats)
# ===========================================================================

section("Shape tests -- YOLO scalar path (floats)")

# Simulate _extract_best_anchor_unc() output
b_ir_uav  = 0.72    # b[..., 0] from YOLO IR
u_ir_s    = 0.18
b_rgb_uav = 0.45
u_rgb_s   = 0.41

# Reconstruct full belief vectors (K=2): b[1] = 1 - b[0] - u
b_ir_t  = torch.tensor([b_ir_uav,  1.0 - b_ir_uav  - u_ir_s])
b_rgb_t = torch.tensor([b_rgb_uav, 1.0 - b_rgb_uav - u_rgb_s])
u_ir_t  = torch.tensor(u_ir_s)
u_rgb_t = torch.tensor(u_rgb_s)

# DBF on scalars
bf_sc, uf_sc = dbf(b_rgb_t, u_rgb_t, b_ir_t, u_ir_t, lam=1.0)
check("YOLO scalar DBF: b_fused shape == (2,)",   bf_sc.shape == (2,))
check("YOLO scalar DBF: u_fused is 0-dim tensor", uf_sc.shape == ())
check("YOLO scalar DBF: u_fused in (0,1]",
      bool(0 < uf_sc.item() <= 1 + 1e-5))
check("YOLO scalar DBF: partition of unity",
      allclose(bf_sc.sum() + uf_sc, torch.tensor(1.0)))

# ABF on scalars
bf_abf_sc, uf_abf_sc = abf(b_rgb_t, u_rgb_t, b_ir_t, u_ir_t)
check("YOLO scalar ABF: partition of unity",
      allclose(bf_abf_sc.sum() + uf_abf_sc, torch.tensor(1.0)))
check("YOLO scalar ABF: b_fused(UAV) in [0,1]",
      bool(0 <= bf_abf_sc[0].item() <= 1))


# ===========================================================================
# Section 7 -- DMS (Dynamic Modality Selection)
# ===========================================================================

section("DMS -- Dynamic Modality Selection")

# Shared boxes for multiple tests
box_rgb_7 = torch.tensor([100.0, 100.0, 40.0, 30.0])   # cx, cy, w, h
box_th_7  = torch.tensor([102.0, 101.0, 38.0, 32.0])

# 7a. RGB has lower variance -> RGB box selected; sigma2_best matches RGB
var_rgb_7a = torch.tensor([0.1, 0.1, 0.1, 0.1])
var_th_7a  = torch.tensor([0.5, 0.5, 0.5, 0.5])
best_7a, s2_7a = dms(box_rgb_7, var_rgb_7a, box_th_7, var_th_7a)
check("DMS RGB lower variance -> RGB box selected",
      allclose(best_7a, box_rgb_7))
sigma2_rgb_ref = torch.tensor([0.1/40, 0.1/30, 0.1/40, 0.1/30]).mean()
check("DMS sigma2_best equals RGB normalized variance",
      allclose(s2_7a, sigma2_rgb_ref))

# 7b. IR has lower variance -> IR box selected
var_rgb_7b = torch.tensor([1.0, 1.0, 1.0, 1.0])
var_th_7b  = torch.tensor([0.05, 0.05, 0.05, 0.05])
best_7b, _ = dms(box_rgb_7, var_rgb_7b, box_th_7, var_th_7b)
check("DMS IR lower variance -> IR box selected",
      allclose(best_7b, box_th_7))

# 7c. Equal normalized variance -> RGB selected (tie-break via <=)
#     Use identical boxes so raw equal variances yield equal normalized variances.
box_same = torch.tensor([100.0, 100.0, 40.0, 30.0])
var_eq    = torch.tensor([0.2, 0.2, 0.2, 0.2])
best_7c, _ = dms(box_same, var_eq, box_same, var_eq)
check("DMS equal variance -> RGB selected (tie-break)",
      allclose(best_7c, box_same))

# 7d. Scale invariance: doubling both box and variances preserves selection
best_orig, _ = dms(box_rgb_7, var_rgb_7a, box_th_7, var_th_7a)
best_scaled, _ = dms(box_rgb_7 * 2, var_rgb_7a * 2, box_th_7 * 2, var_th_7a * 2)
rgb_chosen_orig   = allclose(best_orig,   box_rgb_7)
rgb_chosen_scaled = allclose(best_scaled, box_rgb_7 * 2)
check("DMS scale invariance: proportional scaling preserves selection",
      rgb_chosen_orig == rgb_chosen_scaled)

# 7e. sigma2_best >= 0 for random inputs
torch.manual_seed(4)
box_rnd_rgb = torch.rand(50, 4).abs() + 1.0
box_rnd_th  = torch.rand(50, 4).abs() + 1.0
var_rnd_rgb = torch.rand(50, 4).abs()
var_rnd_th  = torch.rand(50, 4).abs()
_, s2_rnd = dms(box_rnd_rgb, var_rnd_rgb, box_rnd_th, var_rnd_th)
check("DMS sigma2_best >= 0 for random inputs",
      bool((s2_rnd >= 0).all()),
      f"min={s2_rnd.min().item():.4f}")

# 7f. Batch input (B, 4) -> best_box (B, 4) and sigma2_best (B,)
B = 16
torch.manual_seed(5)
box_b_rgb = torch.rand(B, 4).abs() + 1.0
box_b_th  = torch.rand(B, 4).abs() + 1.0
var_b_rgb = torch.rand(B, 4).abs()
var_b_th  = torch.rand(B, 4).abs()
best_b, s2_b = dms(box_b_rgb, var_b_rgb, box_b_th, var_b_th)
check("DMS batch best_box shape == (B, 4)", best_b.shape == (B, 4))
check("DMS batch sigma2_best shape == (B,)", s2_b.shape == (B,))


# ===========================================================================
# Summary
# ===========================================================================

print(f"\n{'=' * 60}")
if _failures:
    print(f"  {FAIL}  {len(_failures)} test(s) failed:")
    for f in _failures:
        print(f"    - {f}")
    sys.exit(1)
else:
    print(f"  {PASS}  All tests passed.")
print(f"{'=' * 60}\n")
