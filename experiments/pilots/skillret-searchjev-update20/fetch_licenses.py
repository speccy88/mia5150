import pathlib,urllib.request,hashlib,json
root=pathlib.Path(__file__).resolve().parent
out=root/'licenses';out.mkdir(exist_ok=True)
sources=[
('Qwen-base-LICENSE.txt','https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/2fc06364715b967f1860aea9cf38778875588b17/LICENSE'),
('SearchJev-code-LICENSE.txt','https://raw.githubusercontent.com/EvoScientist/SearchJev/054ea81d43071b93e2d75b7396a729a2598ac6ac/LICENSE')]
records=[]
for name,url in sources:
 with urllib.request.urlopen(url,timeout=30) as r:data=r.read(65537)
 assert len(data)<=65536
 (out/name).write_bytes(data)
 records.append(dict(file=name,url=url,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)))
(out/'manifest.json').write_text(json.dumps(records,indent=2));print(json.dumps(records))
