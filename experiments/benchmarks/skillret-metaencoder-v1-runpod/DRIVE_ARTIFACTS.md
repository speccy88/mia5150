# Drive artifact delivery

Parent path: **MEM / MIA5150 / MetaEncoder-v1-SkillRet-6006-4392**.

[Open the verified folder](https://drive.google.com/drive/folders/16zcgqtaJSVOAv46mWwf7zli1GOhdf-h4).

## Stable embedding delivery verified

Sixteen files totaling **499,581,427 bytes** were listed from the connected Drive account and sizes matched the transfer originals. The three lossless FP32 NumPy packages are split into seven byte-exact parts (90 MiB maximum per part) because the connector limits individual uploads. Nine metadata/helper files provide instructions, ID mappings, checksums and reassembly. No lossy quantization was used.

- `full_embeddings.npz`: document `(6006,6656)` and query `(4392,6656)` vectors.
- `document_chunks.npz`: content-preserving chunk vectors and separate first-chunk document vectors.
- `exploratory_dimensions.npz`: document/query prefix-reduced and renormalized vectors at 256, 512, 1,024, 2,048, 4,096 dimensions. These are not a native dimensionality feature.
- Full original dimensions are retained independently.

Follow `DOWNLOAD_AND_REASSEMBLE.txt` and `reassemble.py`; do not pass a part file directly to NumPy. Reassembly was verified byte-for-byte for all three packages without evaluating vectors on the transfer computer. Array shape, dtype, normalization and finite-value validation occurred on RunPod.

A representative part was downloaded again through Drive: `exploratory_dimensions.npz.part02`, **1,413,949 bytes**, SHA-256 `a53803620f859fcf318dd03cbbd8873e6c3aa431d2fadf9655e7b83b8ccf79ee`. It matched the source exactly.

## Final evidence delivery verified

The final folder contains **20 files totaling 642,211,029 bytes**. All sizes were checked against a fresh Drive folder listing. The final evidence archive contains source, results, full rankings, figures, manifests, logs and resumable decision checkpoints. It is split into three lossless byte parts: `evidence.tar.gz.part00a`, `evidence.tar.gz.part00b`, `evidence.tar.gz.part01`. Concatenate these in that exact order; the embedding-only `reassemble.py` does not reconstruct this archive.

Reassembled `evidence.tar.gz` is 142,628,820 bytes, SHA-256 `63ff20c22029d97b21fffd24812e1d9d73559f0db93ca2a75ba1a16216fff6c2`, matching the RunPod-generated export manifest. `FINAL_EVIDENCE_MANIFEST.json` records part sizes and SHA-256 values. Its updated contents were read back. Representative `part00a` was downloaded from Drive and its 47,185,920 bytes matched SHA-256 `dc9c21a42ca1412d444ec7086e70eabe729fb49a5e8f6196b0e6a1c57cbea647`.

A 90 MiB upload timed out twice and was confirmed absent before smaller parts were used. No duplicate successful upload was retained. Provider checksum fields were not exposed; verification consists of source hashing, listing/size checks, and representative full-byte readback. The connected account can access the private folder; sharing permissions were not expanded.

The extracted final archive passed a credential scan over 427.94 MB. Public GitHub metadata omits account credit balances and funding amounts; the original private archive preserves evidence bytes. The archive manifest retains original source meanings; publication hashes cover sanitized files separately.
