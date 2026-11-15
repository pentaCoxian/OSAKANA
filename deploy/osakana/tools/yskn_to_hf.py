"""Dequantize a YASAKANI .yskn (v1) main model into a Hugging Face style folder (F16 matrices, BF16 vectors) that
convert_hf_to_gguf.py reads like the original Qwen3.8-27B checkpoint (use --no-mtp: the .yskn has no MTP block).
Every tensor is compared with the original BF16 checkpoint (cosine, relative MSE) and logged.
usage: yskn_to_hf.py <in.yskn> <out dir> <original HF dir> [--limit N]
The format is read as the YASAKANI engine source describes it (src/format.hpp, loader.cpp, v2_decode.cuh)."""
import json, os, shutil, struct, sys, time
import numpy as np, torch
from safetensors import safe_open
from safetensors.torch import save_file

torch.set_num_threads(min(48, os.cpu_count() or 8))
SHARD_BYTES = 5 << 30


def mulberry_signs(seed, offset, n):
    if offset < 0 or n < 0:
        raise ValueError("sign offset and count must be non-negative")
    s = (seed * 7919 + 13) & 0xFFFFFFFF or 0x9E3779B9
    # The state advances by a constant, so skip the unused prefix before mixing.
    s = np.uint32((s + offset * 0x6D2B79F5) & 0xFFFFFFFF)
    states = s + (np.arange(n, dtype=np.uint32) + np.uint32(1)) * np.uint32(0x6D2B79F5)
    t = (states ^ (states >> 15)) * (states | np.uint32(1))
    t ^= t + ((t ^ (t >> 7)) * (t | np.uint32(61)))
    t ^= t >> 14
    return torch.from_numpy(np.where(t < 0x80000000, -1.0, 1.0).astype(np.float32))


def signs(rot, n):
    assert rot["kind"] == "H512" and rot["rule"] == "mulberry32_7919_13", rot
    return mulberry_signs(rot["sign_seed"], rot["sign_offset"], n)


def hadamard(n=512):
    H = torch.ones(1, 1, dtype=torch.float64)
    while H.shape[0] < n: H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / n ** 0.5).float()


H512 = hadamard()


def step_tables(K):
    e = K * (torch.arange(128) + 1); start = e - 12
    word = torch.clamp(start, min=0) >> 5; sh = torch.clamp(start, min=0) & 31
    return e, start, word, torch.clamp(word + 1, max=4 * K - 1), sh


def decode_rows(words, K, lut, scale, s_in, s_out, start_kind):
    """words: int64 [R, seqs, 4K] (one row per stream set). Returns float32 [R, cols]."""
    assert start_kind == "zero"
    e, start, w0, w1, sh = step_tables(K)
    lo = words[..., w0]; hi = words[..., w1] & 0xFFF
    st = ((lo >> sh) | (hi << (32 - sh))) & 0xFFF
    early = start < 0                                                     # first steps: stream bits shifted up, zeros below
    if early.any():
        ee = e[early]
        st[..., early] = (words[..., :1] & ((1 << ee) - 1)) << (12 - ee)
    R, seqs = words.shape[:2]
    X = lut[st].reshape(R, seqs * 256) * scale[:, None]                   # step j -> columns 2j, 2j+1 of each 256-column sequence
    X = (X.view(R, -1, 512) @ H512).view(R, -1) * s_in[None, :]           # W = S_out H diag(scale) Q H S_in
    if s_out is not None:
        X = (H512 @ X.view(R // 512, 512, -1)).view(R, -1) * s_out[:, None]
    return X


def decode_tensor(mm, t):
    rows, cols, K = t["physical_rows"], t["cols"], t["bits"]; seqs, wps = cols // 256, 4 * t["bits"]
    assert t["d"] == 2 and t["t"] == 256 and t["l"] == 12 and not t.get("rar") and rows == t["rows"]
    words = np.frombuffer(mm, "<u4", t["stream_bytes"] // 4, t["stream_off"])
    assert words.size == rows * seqs * wps
    lut = torch.from_numpy(np.frombuffer(mm, "<f2", 8192, t["lut_off"]).astype(np.float32)).view(4096, 2)
    scale = torch.from_numpy(np.frombuffer(mm, "<f2", rows, t["scale_off"]).astype(np.float32))
    s_in = signs(t["rot_in"], cols)
    ro = t.get("rot_out"); s_out = signs(ro, rows) if ro and ro["kind"] != "none" else None
    chunk = max(512, (1 << 26) // (seqs * 128) // 512 * 512)               # about 64M states per chunk
    out = torch.empty(rows, cols, dtype=torch.float16)
    if t["layout"] == "warp":                                             # file: [seq][32-row tile][16-byte chunk][lane][4 words]
        w5 = words.reshape(seqs, rows // 32, wps // 4, 32, 4)
    else:
        assert t["layout"] == "row"; w5 = words.reshape(rows, seqs, wps)
    for r0 in range(0, rows, chunk):
        r1 = min(rows, r0 + chunk)
        if t["layout"] == "warp":
            blk = w5[:, r0 // 32:r1 // 32].transpose(1, 3, 0, 2, 4).reshape(r1 - r0, seqs, wps)
        else:
            blk = w5[r0:r1]
        X = decode_rows(torch.from_numpy(blk.astype(np.int64)), K, lut, scale[r0:r1], s_in, None if s_out is None else s_out[r0:r1], t["start"])
        out[r0:r1] = X.half()
    return out


def hf_name(n):
    if n == "lm_head": return "lm_head.weight"
    return "model.language_model." + n + ("" if n.endswith(("A_log", "dt_bias")) else ".weight")


def main():
    src, out_dir, orig = sys.argv[1:4]
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    os.makedirs(out_dir, exist_ok=True)
    with open(src, "rb") as f: hdr = f.read(0x20)
    magic, ver, jl = struct.unpack_from("<4sII", hdr)
    assert magic == b"YSKN" and ver == 1
    mm = np.memmap(src, np.uint8, "r")
    j = json.loads(bytes(mm[0x20:0x20 + jl]))
    oidx = json.load(open(f"{orig}/model.safetensors.index.json"))["weight_map"]
    handles = {}
    def orig_tensor(name):
        sh = oidx[name]
        if sh not in handles: handles[sh] = safe_open(f"{orig}/{sh}", "pt")
        return handles[sh].get_tensor(name)
    log = open(f"{out_dir}/compare.jsonl", "w")
    shard, shard_bytes, files, wmap = {}, 0, [], {}
    def flush():
        nonlocal shard, shard_bytes
        if not shard: return
        fn = f"model-{len(files) + 1:05d}.safetensors"; save_file(shard, f"{out_dir}/{fn}", metadata={"format": "pt"})
        files.append(fn); wmap.update({k: fn for k in shard}); shard, shard_bytes = {}, 0
    def add(name, ten):
        nonlocal shard_bytes
        shard[name] = ten.contiguous(); shard_bytes += ten.numel() * ten.element_size()
        if shard_bytes >= SHARD_BYTES: flush()
    t0 = time.time(); items = j["tensors"][:limit] if limit else j["tensors"]
    for i, t in enumerate(items):
        name = hf_name(t["name"]); ts = time.time()
        W = decode_tensor(mm, t)
        R = orig_tensor(name)
        assert tuple(R.shape) == tuple(W.shape), (name, R.shape, W.shape)
        num = ww = rr = ee = 0.0                                          # float64 sums by row chunks (float32 norm() is off by 0.2% at 30M+ elements)
        for r0 in range(0, R.shape[0], 8192):
            a = W[r0:r0 + 8192].double(); b = R[r0:r0 + 8192].double()
            num += (a * b).sum().item(); ww += (a * a).sum().item(); rr += (b * b).sum().item(); ee += ((a - b) ** 2).sum().item()
        rec = {"name": name, "bits": t["bits"], "bpw": t["bits"] / 2, "cos": num / (ww * rr) ** 0.5, "rel_mse": ee / rr,
               "proxy": t.get("proxy"), "secs": round(time.time() - ts, 1)}
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(f"[{i + 1}/{len(items)}] {name} {t['bits'] / 2:.0f} bpw cos {rec['cos']:.5f} rel_mse {rec['rel_mse']:.5f} ({rec['secs']} s, total {time.time() - t0:.0f} s)", flush=True)
        add(name, W); del R
    vec_diff = 0.0
    for v in j["vectors"]:
        assert v["dtype"] == "bf16", v
        raw = np.frombuffer(mm, "<u2", v["n"], v["off"]).copy()
        ten = torch.from_numpy(raw.view(np.int16)).view(torch.bfloat16)
        name = hf_name(v["name"]); R = orig_tensor(name)
        ten = ten.reshape(R.shape)
        d = (ten.float() - R.float()).abs().max().item(); vec_diff = max(vec_diff, d)
        if d: print(f"vector differs from the original: {name} max abs diff {d}", flush=True)
        add(name, ten)
    flush()
    json.dump({"metadata": {"total_size": sum(os.path.getsize(f"{out_dir}/{f}") for f in files)}, "weight_map": wmap},
              open(f"{out_dir}/model.safetensors.index.json", "w"), indent=1)
    for f in os.listdir(orig):                                            # config, tokenizer and template files of the original
        if not f.endswith(".safetensors") and f != "model.safetensors.index.json" and os.path.isfile(f"{orig}/{f}"):
            shutil.copy(f"{orig}/{f}", f"{out_dir}/{f}")
    print(f"done: {len(items)} matrices, {len(j['vectors'])} vectors (max abs diff from the original {vec_diff}), {len(files)} shards, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
