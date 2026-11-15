"""Write a GGUF whose 402 matrices (embedding included) keep the exact YASAKANI trellis codes (ggml types YTR3 / YTR4 of this engine).
The token embedding is YTR too (read by the CPU get_rows path). Metadata, tensor order and every other tensor (norms, ...) come from the F16 GGUF made by yskn_to_hf.py + convert_hf_to_gguf.py.
Each YTR tensor is decoded back the way the CUDA kernel reads it and compared with the F16 GGUF tensor.
usage: yskn_to_ytr_gguf.py <in.yskn> <template f16.gguf> <out.gguf> [--no-check]
Tensor data layout: see ggml/src/ggml-cuda/ytr.cu."""
import json, struct, sys, time
import numpy as np, torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3] / "gguf-py"))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import gguf
from yskn_to_hf import decode_rows, mulberry_signs

MAGIC = 0x31525459  # "YTR1"
GT = {6: gguf.GGMLQuantizationType.YTR3, 8: gguf.GGMLQuantizationType.YTR4}


def gguf_name(n):
    if n == "lm_head": return "output.weight"
    if n == "embed_tokens": return "token_embd.weight"
    m = {"linear_attn.in_proj_qkv": "attn_qkv", "linear_attn.in_proj_z": "attn_gate", "linear_attn.out_proj": "ssm_out",
         "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down",
         "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output"}
    _, layer, rest = n.split(".", 2)
    return f"blk.{layer}.{m[rest]}.weight"


def reorder_idx(n_k, n_v_per_k, head_dim):
    """index form of convert_hf_to_gguf's _reorder_v_heads along one dimension: gguf[i] = hf[p[i]]"""
    return np.arange(n_k * n_v_per_k * head_dim).reshape(n_k, n_v_per_k, head_dim).transpose(1, 0, 2).reshape(-1)


_sign_cache = {}
def sign_bits(rot, n):
    key = (rot["sign_seed"], rot["sign_offset"], n)
    if key not in _sign_cache:
        assert rot["kind"] == "H512" and rot["rule"] == "mulberry32_7919_13" and rot.get("qm_off", 0) == 0, rot
        s = mulberry_signs(rot["sign_seed"], rot["sign_offset"], n).numpy()
        _sign_cache[key] = (np.packbits(s < 0, bitorder="little"), s)
    return _sign_cache[key]


def build(mm, t, rmap, cmap):
    rows, cols, K = t["physical_rows"], t["cols"], t["bits"]; seqs, wps = cols // 256, 4 * K
    assert rows == t["rows"] and t["d"] == 2 and t["t"] == 256 and t["l"] == 12 and t["start"] == "zero" and not t.get("rar") and t["layout"] in ("warp", "row")
    total = rows * cols // 256 * (16 * K + 1)
    out = np.zeros(total, np.uint8)
    ro = t.get("rot_out"); has_out = bool(ro) and ro["kind"] != "none"
    parts = [("lut", np.frombuffer(mm, np.uint8, 16384, t["lut_off"])), ("scale", np.frombuffer(mm, np.uint8, 2 * rows, t["scale_off"])),
             ("sin", sign_bits(t["rot_in"], cols)[0])]
    if has_out: parts.append(("sout", sign_bits(ro, rows)[0]))
    if rmap is not None: parts.append(("rmap", rmap.astype("<i4").view(np.uint8)))
    if cmap is not None: parts.append(("cmap", cmap.astype("<i4").view(np.uint8)))
    off, offs = 64, {}
    for name, arr in parts:
        off = (off + 15) // 16 * 16; offs[name] = off; out[off:off + arr.size] = arr; off += arr.size
    code_off = (off + 255) // 256 * 256
    words = np.frombuffer(mm, "<u4", t["stream_bytes"] // 4, t["stream_off"])
    tiled = t["layout"] == "warp"           # matrices: lane layout (flag 16); the embedding stays row major for the CPU row lookup
    if tiled:
        # lane layout of the fast kernel: per 16-row tile and 256-column sequence, lane l = 4 gi + q holds words K q .. K q + K - 1
        # of rows gi and gi + 8 (2K words = K/2 uint4); uint4 u of lane l at ((tile * seqs + s) * K/2 + u) * 32 + l
        rm = words.reshape(seqs, rows // 32, wps // 4, 32, 4).transpose(1, 3, 0, 2, 4).reshape(rows, seqs, wps)   # [row][seq][word]
        a = rm.reshape(rows // 16, 2, 8, seqs, 4, K)            # [tile][rr][gi][seq][q][k]
        a = a.transpose(0, 3, 2, 4, 1, 5).reshape(rows // 16, seqs, 32, 2 * K)   # [tile][seq][lane][rr * K + k]
        codes = a.reshape(rows // 16, seqs, 32, K // 2, 4).transpose(0, 1, 3, 2, 4).reshape(-1)   # [tile][seq][u][lane][4]
    else:
        codes = words                                                                                # "row" is [row][seq][word] already
    assert code_off + codes.nbytes <= total, (t["name"], code_off, codes.nbytes, total)
    out[code_off:code_off + codes.nbytes] = codes.view(np.uint8)
    flags = (1 if has_out else 0) | (2 if rmap is not None else 0) | (4 if cmap is not None else 0) | (16 if tiled else 0)
    hdr = struct.pack("<IHHII7I5I", MAGIC, K, flags, rows, cols, offs["lut"], offs["scale"], offs["sin"], offs.get("sout", 0),
                      offs.get("rmap", 0), offs.get("cmap", 0), code_off, 0, 0, 0, 0, 0)
    out[:64] = np.frombuffer(hdr, np.uint8)
    return out.reshape(rows, -1)


def decode_ytr(b):
    """read a YTR tensor the way ytr.cu does; returns float32 [rows, cols] in GGUF order"""
    flat = b.reshape(-1)
    magic, K, flags, rows, cols, lut_o, sc_o, sin_o, sout_o, rm_o, cm_o, code_o = struct.unpack_from("<IHHII7I", flat)
    assert magic == MAGIC
    seqs, wps = cols // 256, 4 * K
    lut = torch.from_numpy(np.frombuffer(flat, "<f2", 8192, lut_o).astype(np.float32)).view(4096, 2)
    scale = torch.from_numpy(np.frombuffer(flat, "<f2", rows, sc_o).astype(np.float32))
    unbits = lambda o, n: torch.from_numpy(np.where(np.unpackbits(np.frombuffer(flat, np.uint8, n // 8, o), bitorder="little")[:n] == 1, -1.0, 1.0).astype(np.float32))
    s_in = unbits(sin_o, cols); s_out = unbits(sout_o, rows) if flags & 1 else None
    words = np.frombuffer(flat, "<u4", rows * seqs * wps, code_o)
    if flags & 8:
        words = words.reshape(rows // 16, seqs, wps // 4, 16, 4).transpose(0, 3, 1, 2, 4)
    elif flags & 16:
        a = words.reshape(rows // 16, seqs, K // 2, 32, 4).transpose(0, 1, 3, 2, 4).reshape(rows // 16, seqs, 8, 4, 2, K)   # [tile][seq][gi][q][rr][k]
        words = a.transpose(0, 4, 2, 1, 3, 5)                   # [tile][rr][gi][seq][q][k]
    words = words.reshape(rows, seqs, wps)
    W = torch.empty(rows, cols)
    chunk = 8192
    for r0 in range(0, rows, chunk):
        r1 = min(rows, r0 + chunk)
        W[r0:r1] = decode_rows(torch.from_numpy(words[r0:r1].astype(np.int64)), K, lut, scale[r0:r1], s_in, None if s_out is None else s_out[r0:r1], "zero")
    if flags & 2:
        W = W[torch.from_numpy(np.frombuffer(flat, "<i4", rows, rm_o).astype(np.int64))]
    if flags & 4:
        cm = np.frombuffer(flat, "<i4", cols, cm_o).astype(np.int64)
        W = W[:, torch.from_numpy(np.argsort(cm))]          # x_code[k] = x[cmap[k]]  ->  W_gguf[:, cmap[k]] = W_code[:, k]
    return W


def main():
    src, tmpl, out_path = sys.argv[1:4]; check = "--no-check" not in sys.argv
    mm = np.memmap(src, np.uint8, "r")
    jl = struct.unpack_from("<I", bytes(mm[:12]), 8)[0]; j = json.loads(bytes(mm[0x20:0x20 + jl])); md = j["model"]
    nk, nv, hd, kd = md["linear_num_key_heads"], md["linear_num_value_heads"], md["linear_value_head_dim"], md["linear_key_head_dim"]
    vperm = reorder_idx(nk, nv // nk, hd)
    ytr = {}
    for t in j["tensors"]:
        name = gguf_name(t["name"]); rmap = cmap = None
        if name.endswith("attn_qkv.weight"):
            qk = 2 * nk * kd; rmap = np.concatenate([np.arange(qk), qk + vperm])
        elif name.endswith("attn_gate.weight") and "linear_attn" in t["name"]:
            rmap = vperm
        elif name.endswith("ssm_out.weight"):
            cmap = np.argsort(vperm)
        ytr[name] = (t, rmap, cmap)
    reader = gguf.GGUFReader(tmpl)
    rt = {t.name: t for t in reader.tensors}
    missing = [n for n in ytr if n not in rt]; assert not missing, missing[:5]
    writer = gguf.GGUFWriter(out_path, arch=reader.fields["general.architecture"].contents())
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF.") or field.name == "general.file_type":
            continue
        vt = field.types[0]; st = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), vt, sub_type=st)
    writer.add_key_value("general.description", "Qwen3.8-27B YASAKANI speed weights with the original trellis codes (YTR3/YTR4)", gguf.GGUFValueType.STRING)
    for t in reader.tensors:
        if t.name in ytr:
            d = ytr[t.name][0]; K = d["bits"]; rb = d["cols"] // 256 * (16 * K + 1)
            writer.add_tensor_info(t.name, (d["physical_rows"], rb), np.dtype(np.uint8), d["physical_rows"] * rb, raw_dtype=GT[K])
        else:
            writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_ti_data_to_file()
    t0 = time.time(); worst = 0.0
    for i, t in enumerate(reader.tensors):
        if t.name not in ytr:
            writer.write_tensor_data(t.data, tensor_endianess=reader.endianess); continue
        d, rmap, cmap = ytr[t.name]
        b = build(mm, d, rmap, cmap)
        if check:
            W = decode_ytr(b); R = torch.from_numpy(np.asarray(t.data, dtype=np.float32).reshape(W.shape))
            rel = ((W - R).abs().max() / R.abs().max()).item(); worst = max(worst, rel)
            assert rel < 2e-3, (t.name, rel)
            print(f"[{i}] {t.name} {d['bits'] // 2} bpw flags rows/cols maps {rmap is not None}/{cmap is not None} max rel diff vs F16 {rel:.2e} ({time.time() - t0:.0f} s)", flush=True)
        writer.write_tensor_data(b)
    writer.close()
    print(f"done: {len(ytr)} YTR tensors, worst max rel diff {worst:.2e}, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
