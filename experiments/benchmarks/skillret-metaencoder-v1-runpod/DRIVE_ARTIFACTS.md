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

## Still pending

The final `evidence.tar.gz` and final source manifest will be uploaded after decision experiments and diagnostics finish. The stable embeddings are delivered, but the complete requested Drive evidence package is not yet marked complete. Do not mistake an early evidence archive for the final report. Final file count/storage will be revised after those additions.
