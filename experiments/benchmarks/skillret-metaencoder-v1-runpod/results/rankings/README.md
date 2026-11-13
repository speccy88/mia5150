# Primary rankings

Complete primary top-50 and top-100 rankings are line-boundary shards to fit connector request limits. Concatenate each set in numeric part order to restore the exact original JSONL bytes; verify the SHA-256 in REASSEMBLY.json. No ranking or score was changed during packaging. All alternative-dimension and truncation rankings are retained in the Drive evidence archive.

```sh
cat full_6656_top100.part*.jsonl > full_6656_top100.jsonl
cat full_6656_top50.part*.jsonl > full_6656_top50.jsonl
```
