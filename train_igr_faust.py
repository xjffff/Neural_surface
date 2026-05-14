"""
IGR-style neural SDF training for a single shape (FAUST / any mesh)
===================================================================
Replaces the DiGS pipeline.  Follows Gropp et al., "Implicit Geometric
Regularization for Learning Shapes" (ICML 2020):

  * loss = |f(x_surf)|              (manifold: zero level set)
         + lambda_n * |grad f - n|  (normal supervision -- you HAVE mesh normals)
         + lambda_e * (||grad f|| - 1)^2   (Eikonal, on the sampled distribution)
    NO divergence term, NO div clamp.

  * non-manifold points = per-point near-surface Gaussian  +  a little global
    uniform.  (Not whole-bbox uniform like DiGS -- supervision is concentrated
    where it matters: the thin shell around the surface.)

  * softplus MLP with IGR geometric initialization (starts as ~a sphere SDF).

Coordinate frame
----------------
The mesh is centered and scaled EXACTLY as in the eigenfunction notebook:
    cp    = V.mean(0)
    scale = abs(V - cp).max()        # L-inf
    V_norm = (V - cp) / scale
so the trained SDF's zero level set lives in the same frame as the cotangent
LBO reference mesh `V_norm`.  cp / scale are stored in the checkpoint.

Two things this fixes vs the old DiGS run
-----------------------------------------
  1. The mesh VERTICES are added to the manifold supervision set, so the very
     points you later evaluate |F(v)| on are directly constrained to f = 0.
  2. Normal supervision is actually used (DiGS `wo_n` threw it away).

Usage
-----
    python train_igr_faust.py --input_path FAUST_r/off/tr_reg_000.off \
        --outdir shape000 --n_iters 10000 --gpu 0

Loading the result in the eigenfunction notebook (replace the DiGS loader cell):

    from train_igr_faust import IGRNetwork
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    sdf_model = IGRNetwork(**ckpt['arch']).to(device)
    sdf_model.load_state_dict(ckpt['state_dict'])
    sdf_model.eval()
    def sdf_torch(x):
        out = sdf_model(x)
        return out[:, 0] if out.dim() == 2 else out
    # ckpt['cp'], ckpt['scale'] are the SAME normalization used for V_norm.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad


# ======================================================================
# 1. NETWORK  --  IGR implicit net: softplus MLP + skip + geometric init
# ======================================================================
class IGRNetwork(nn.Module):
    def __init__(self, d_in=3, hidden_dim=512, n_layers=8,
                 skip_in=(4,), radius_init=1.0, beta=100.0, geometric_init=True):
        """8x512 by default, single skip connection at layer 4, Softplus(beta).

        geometric_init makes f(x) ~= ||x|| - radius_init at initialization,
        which is what lets IGR converge reliably without divergence tricks."""
        super().__init__()
        dims = [d_in] + [hidden_dim] * n_layers + [1]
        self.num_layers = len(dims)
        self.skip_in = set(skip_in)
        self.d_in = d_in
        # keep the config so the module can be reconstructed exactly on load
        # (a skip layer shrinks its feeding layer's width, so we cannot
        #  reliably read hidden_dim back off the weight shapes).
        self._cfg = dict(d_in=d_in, hidden_dim=hidden_dim, n_layers=n_layers,
                         skip_in=tuple(sorted(skip_in)), radius_init=radius_init,
                         beta=beta)

        for layer in range(self.num_layers - 1):
            # a layer that feeds INTO a skip layer outputs fewer features,
            # so that after concatenating the input the width is restored.
            if (layer + 1) in self.skip_in:
                out_dim = dims[layer + 1] - d_in
            else:
                out_dim = dims[layer + 1]
            lin = nn.Linear(dims[layer], out_dim)

            if geometric_init:
                if layer == self.num_layers - 2:          # last linear layer
                    nn.init.normal_(lin.weight,
                                    mean=np.sqrt(np.pi) / np.sqrt(dims[layer]),
                                    std=1e-5)
                    nn.init.constant_(lin.bias, -radius_init)
                else:
                    nn.init.constant_(lin.bias, 0.0)
                    nn.init.normal_(lin.weight, 0.0,
                                    np.sqrt(2) / np.sqrt(out_dim))
            setattr(self, f"lin{layer}", lin)

        self.activation = nn.Softplus(beta=beta)

    def forward(self, x):
        inp = x
        for layer in range(self.num_layers - 1):
            lin = getattr(self, f"lin{layer}")
            if layer in self.skip_in:
                x = torch.cat([x, inp], dim=-1) / np.sqrt(2)
            x = lin(x)
            if layer < self.num_layers - 2:
                x = self.activation(x)
        return x                                          # (N, 1)

    def arch_kwargs(self):
        """Everything needed to reconstruct the module for loading."""
        return dict(d_in=self.d_in,
                    hidden_dim=getattr(self, "lin1").weight.shape[0],
                    n_layers=self.num_layers - 2,
                    skip_in=tuple(sorted(self.skip_in)))


def gradient(inputs, outputs):
    """d outputs / d inputs, shape (N, 3).  outputs must come from inputs
    with requires_grad=True and create_graph=True upstream."""
    d = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    g = grad(outputs=outputs, inputs=inputs, grad_outputs=d,
             create_graph=True, retain_graph=True, only_inputs=True)[0]
    return g


# ======================================================================
# 2. DATA  --  load mesh, normalize, build manifold point+normal set
# ======================================================================
def load_and_prepare(input_path, n_surface_samples=100_000, knn_sigma=50):
    """Returns a dict with everything the training loop needs.

    Manifold set = (face-area-weighted surface samples) UNION (mesh vertices).
    Including the vertices removes the train/eval mismatch: the points you
    later test |F(v)| on are themselves supervised to f = 0."""
    import trimesh
    from scipy.spatial import cKDTree

    mesh = trimesh.load(input_path, force="mesh", process=False)
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int64)

    # --- normalization: identical to the eigenfunction notebook ---
    cp = V.mean(axis=0)
    scale = float(np.abs(V - cp).max())
    V_norm = ((V - cp) / scale).astype(np.float32)

    mesh_norm = trimesh.Trimesh(V_norm, F, process=False)

    # --- face-area-weighted surface samples + their face normals ---
    samp_pts, face_idx = trimesh.sample.sample_surface(mesh_norm, n_surface_samples)
    samp_nrm = mesh_norm.face_normals[face_idx]

    # --- mesh vertices + vertex normals, appended to the manifold set ---
    vtx_pts = V_norm
    vtx_nrm = np.asarray(mesh_norm.vertex_normals, dtype=np.float32)

    mnfld_pts = np.concatenate([samp_pts, vtx_pts], axis=0).astype(np.float32)
    mnfld_nrm = np.concatenate([samp_nrm, vtx_nrm], axis=0).astype(np.float32)
    # renormalize normals (sampling / averaging can leave them slightly off-unit)
    mnfld_nrm /= (np.linalg.norm(mnfld_nrm, axis=1, keepdims=True) + 1e-12)

    # --- per-point local sigma = distance to the knn_sigma-th neighbor ---
    # used for the near-surface non-manifold sampling (IGR `sample_local`).
    tree = cKDTree(mnfld_pts)
    d_knn = tree.query(mnfld_pts, k=knn_sigma + 1)[0][:, -1]
    local_sigma = d_knn.astype(np.float32)[:, None]            # (M, 1)

    print(f"Loaded {input_path}")
    print(f"  vertices                : {V.shape[0]}")
    print(f"  surface samples         : {n_surface_samples}")
    print(f"  manifold set (samp+vtx) : {mnfld_pts.shape[0]}")
    print(f"  normalization           : cp={np.round(cp,4)}  scale={scale:.4f}")
    print(f"  local_sigma  mean/max   : {local_sigma.mean():.4f} / {local_sigma.max():.4f}")

    return dict(V_norm=V_norm, F=F, cp=cp, scale=scale,
                mnfld_pts=mnfld_pts, mnfld_nrm=mnfld_nrm,
                local_sigma=local_sigma)


def sample_nonmanifold(mnfld_pts, local_sigma, idx, global_ratio=8, global_sigma=1.1):
    """IGR non-manifold sampler for one batch.

    near-surface : manifold points + N(0, local_sigma^2)   (the bulk)
    global       : a smaller uniform sample over the bounding box
    Eikonal is evaluated on this combined set, so it is enforced both right
    next to the surface and out in free space."""
    base = mnfld_pts[idx]                                     # (B, 3)
    sig = local_sigma[idx]                                    # (B, 1)
    near = base + torch.randn_like(base) * sig
    n_global = max(1, base.shape[0] // global_ratio)
    glob = (torch.rand(n_global, 3, device=base.device) * 2 - 1) * global_sigma
    return torch.cat([near, glob], dim=0)


# ======================================================================
# 3. LOSS
# ======================================================================
def igr_loss(model, mnfld_pts, mnfld_nrm, nonmnfld_pts,
             lambda_normal=1.0, lambda_eik=0.1):
    """manifold |f| + normal alignment + Eikonal.  No divergence."""
    mnfld_pts = mnfld_pts.clone().requires_grad_(True)
    nonmnfld_pts = nonmnfld_pts.clone().requires_grad_(True)

    f_mnfld = model(mnfld_pts)
    f_nonmnfld = model(nonmnfld_pts)

    g_mnfld = gradient(mnfld_pts, f_mnfld)
    g_nonmnfld = gradient(nonmnfld_pts, f_nonmnfld)

    # zero level set: f = 0 on the surface
    manifold_loss = f_mnfld.abs().mean()

    # normal supervision: grad f == n on the surface  (IGR uses the L2 diff)
    normals_loss = (g_mnfld - mnfld_nrm).norm(2, dim=-1).mean()

    # Eikonal: ||grad f|| == 1 on the sampled distribution
    eik_loss = ((g_nonmnfld.norm(2, dim=-1) - 1.0) ** 2).mean()

    loss = manifold_loss + lambda_normal * normals_loss + lambda_eik * eik_loss
    return loss, dict(manifold=manifold_loss.item(),
                      normal=normals_loss.item(),
                      eikonal=eik_loss.item())


# ======================================================================
# 4. DIAGNOSTICS  --  the numbers you actually care about for the LBO step
# ======================================================================
@torch.no_grad()
def _f_only(model, x, batch=100_000):
    return torch.cat([model(x[i:i + batch]) for i in range(0, len(x), batch)])


def diagnostics(model, data, device):
    """|F| on the true mesh vertices  +  ||grad F|| stats on the surface."""
    V = torch.from_numpy(data["V_norm"]).float().to(device)

    f_v = _f_only(model, V).squeeze(-1).cpu().numpy()
    af = np.abs(f_v)

    Vg = V.clone().requires_grad_(True)
    g = gradient(Vg, model(Vg))
    gn = g.norm(2, dim=-1).detach().cpu().numpy()

    print("  --- diagnostics on mesh vertices ---")
    print(f"  |F(v)|   mean={af.mean():.5f}  max={af.max():.5f}  "
          f"p95={np.quantile(af,0.95):.5f}   (target mean < 1e-3)")
    print(f"  |grad F| mean={gn.mean():.4f}  std={gn.std():.4f}  "
          f"min={gn.min():.4f}                 (target mean~1, std small)")
    return af.mean(), gn.mean(), gn.std()


# ======================================================================
# 5. MARCHING CUBES  --  same-surface reference mesh for the LBO comparison
# ======================================================================
@torch.no_grad()
def extract_mesh(model, device, resolution=256, bound=1.1):
    """Marching cubes in the NORMALIZED frame (no un-scaling), so the result
    is directly comparable to the cotangent-LBO reference mesh V_norm."""
    from skimage import measure
    lin = np.linspace(-bound, bound, resolution, dtype=np.float32)
    xx, yy, zz = np.meshgrid(lin, lin, lin, indexing="ij")
    grid = torch.from_numpy(np.stack([xx.ravel(), yy.ravel(), zz.ravel()], 1)).to(device)
    vals = _f_only(model, grid).squeeze(-1).cpu().numpy()
    vals = vals.reshape(resolution, resolution, resolution).astype(np.float64)
    if vals.min() > 0 or vals.max() < 0:
        print("  marching cubes: SDF does not cross zero -- skipped.")
        return None
    spacing = (2 * bound / (resolution - 1),) * 3
    verts, faces, _, _ = measure.marching_cubes(vals, level=0.0, spacing=spacing)
    verts += np.array([-bound, -bound, -bound])
    import trimesh
    return trimesh.Trimesh(verts, faces, process=False)


# ======================================================================
# 6. TRAINING
# ======================================================================
def train(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = load_and_prepare(args.input_path,
                            n_surface_samples=args.n_surface_samples,
                            knn_sigma=args.knn_sigma)

    mnfld_pts = torch.from_numpy(data["mnfld_pts"]).to(device)
    mnfld_nrm = torch.from_numpy(data["mnfld_nrm"]).to(device)
    local_sigma = torch.from_numpy(data["local_sigma"]).to(device)
    M = mnfld_pts.shape[0]

    model = IGRNetwork(d_in=3, hidden_dim=args.hidden_dim, n_layers=args.n_layers,
                       skip_in=tuple(args.skip_in), radius_init=args.radius_init,
                       beta=args.beta).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: IGRNetwork {args.n_layers}x{args.hidden_dim}  ({n_params:,} params)")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[int(args.n_iters * m) for m in (0.5, 0.75, 0.9)], gamma=0.5)

    os.makedirs(args.outdir, exist_ok=True)

    for it in range(args.n_iters + 1):
        model.train()
        idx = torch.randint(0, M, (args.batch_size,), device=device)
        b_pts, b_nrm = mnfld_pts[idx], mnfld_nrm[idx]
        nonmnfld = sample_nonmanifold(mnfld_pts, local_sigma, idx,
                                      global_ratio=args.global_ratio,
                                      global_sigma=args.global_sigma)

        loss, parts = igr_loss(model, b_pts, b_nrm, nonmnfld,
                               lambda_normal=args.lambda_normal,
                               lambda_eik=args.lambda_eik)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        sched.step()

        if it % args.print_every == 0:
            print(f"[iter {it:6d}] loss={loss.item():.5f}  "
                  f"mnfld={parts['manifold']:.5f}  "
                  f"normal={parts['normal']:.5f}  "
                  f"eik={parts['eikonal']:.5f}")
        if it > 0 and it % args.diag_every == 0:
            model.eval()
            diagnostics(model, data, device)

    # ---- final diagnostics ----
    print("\n=== final ===")
    model.eval()
    mean_f, mean_g, std_g = diagnostics(model, data, device)

    # ---- save checkpoint (state + normalization + arch) ----
    ckpt_path = os.path.join(args.outdir, "igr_model.pth")
    torch.save(dict(state_dict=model.state_dict(),
                    arch=model.arch_kwargs(),
                    cp=data["cp"], scale=data["scale"],
                    final_mean_absF=mean_f, final_mean_gradnorm=mean_g),
               ckpt_path)
    print(f"checkpoint saved: {ckpt_path}")

    # ---- same-surface reference mesh for the LBO comparison ----
    if args.extract_mesh:
        mc = extract_mesh(model, device, resolution=args.mc_res)
        if mc is not None:
            mc_path = os.path.join(args.outdir, "mc_mesh_normalized.ply")
            mc.export(mc_path)
            print(f"marching-cubes mesh (normalized frame): {mc_path}")
            print(f"  {len(mc.vertices)} verts, {len(mc.faces)} faces")
            print("  -> run cotangent LBO on THIS mesh: it is the same-surface,"
                  " same-scale reference for the neural eigenvalues.")
    return model, data


def get_args():
    p = argparse.ArgumentParser(description="IGR-style single-shape SDF training")
    p.add_argument("--input_path", type=str, required=True,
                   help="mesh file (.off/.ply/.obj)")
    p.add_argument("--outdir", type=str, default="./igr_output")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)

    # data
    p.add_argument("--n_surface_samples", type=int, default=100_000)
    p.add_argument("--knn_sigma", type=int, default=50,
                   help="local_sigma = distance to this-th nearest neighbor")

    # network
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--n_layers", type=int, default=8)
    p.add_argument("--skip_in", nargs="+", type=int, default=[4])
    p.add_argument("--radius_init", type=float, default=1.0,
                   help="radius of the sphere the net approximates at init")
    p.add_argument("--beta", type=float, default=100.0, help="softplus beta")

    # optimization
    p.add_argument("--n_iters", type=int, default=10_000)
    p.add_argument("--batch_size", type=int, default=16_384)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad_clip", type=float, default=10.0)

    # loss weights
    p.add_argument("--lambda_normal", type=float, default=1.0)
    p.add_argument("--lambda_eik", type=float, default=0.1)

    # non-manifold sampling
    p.add_argument("--global_ratio", type=int, default=8,
                   help="near-surface : global  ==  global_ratio : 1")
    p.add_argument("--global_sigma", type=float, default=1.1,
                   help="half-width of the global uniform box")

    # logging / output
    p.add_argument("--print_every", type=int, default=200)
    p.add_argument("--diag_every", type=int, default=2000)
    p.add_argument("--extract_mesh", action="store_true",
                   help="run marching cubes at the end (same-surface LBO reference)")
    p.add_argument("--mc_res", type=int, default=256)
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    print("=" * 60)
    print("IGR-style SDF training")
    print("=" * 60)
    print(f"input        : {args.input_path}")
    print(f"network      : {args.n_layers} x {args.hidden_dim}  skip@{args.skip_in}")
    print(f"loss         : |f| + {args.lambda_normal}*normal + {args.lambda_eik}*eikonal"
          "   (no divergence)")
    print(f"iters/batch  : {args.n_iters} / {args.batch_size}")
    print("=" * 60)
    train(args)