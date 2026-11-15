"""Add the QSA indexers of a QSA GGUF (80 tensors, 4 metadata keys) to another GGUF of the same base model.
usage: add_qsa_indexers.py <model.gguf> <qsa.gguf> <out.gguf>"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3] / "gguf-py"))
import gguf

src, qsa, out = sys.argv[1:4]
R, Q = gguf.GGUFReader(src), gguf.GGUFReader(qsa)
KEYS = [k for k in Q.fields if ".attention.indexer." in k or k.endswith(".attention.compress_ratios")]
idx = [t for t in Q.tensors if ".indexer." in t.name]
assert len(idx) == 80 and len(KEYS) == 4, (len(idx), KEYS)
assert not any(".indexer." in t.name for t in R.tensors)
W = gguf.GGUFWriter(out, arch=R.fields["general.architecture"].contents())
def put(f):
    vt = f.types[0]
    W.add_key_value(f.name, f.contents(), vt, sub_type=f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None)
for f in R.fields.values():
    if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."):
        continue
    put(f)
n_layer = [f.contents() for f in R.fields.values() if f.name.endswith(".block_count")][0]
for k in KEYS:
    f = Q.fields[k]
    if k.endswith(".compress_ratios"):   # one entry per layer; the QSA file also counts its MTP layer
        W.add_key_value(k, list(f.contents())[:n_layer], gguf.GGUFValueType.ARRAY, sub_type=f.types[-1])
    else:
        put(f)
ts = list(R.tensors) + idx
for t in ts:
    W.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
W.write_header_to_file(); W.write_kv_data_to_file(); W.write_ti_data_to_file()
for reader, tensors in ((R, R.tensors), (Q, idx)):
    for t in tensors:
        W.write_tensor_data(t.data, tensor_endianess=reader.endianess)
W.close()
print("wrote", out, len(ts), "tensors, keys", KEYS)
