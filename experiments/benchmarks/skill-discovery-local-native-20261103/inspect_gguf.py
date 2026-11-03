"""Read GGUF identity/shape metadata without executing model code."""
import json
from pathlib import Path
import struct

FMT={0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}

def inspect(path, include_chat_template=False):
    with Path(path).open('rb') as f:
        def num(fmt):
            return struct.unpack('<'+fmt,f.read(struct.calcsize('<'+fmt)))[0]
        def string(keep=True):
            n=num('Q')
            if keep:
                return f.read(n).decode('utf-8','replace')
            f.seek(n,1)
        def value(t,keep=True):
            if t==8:
                return string(keep)
            if t==9:
                child,n=num('I'),num('Q')
                for _ in range(n):
                    value(child,False)
                return {'array_type':child,'length':n}
            if t in FMT:
                return num(FMT[t])
            raise ValueError('Unknown GGUF metadata type '+str(t))
        if f.read(4)!=b'GGUF':
            raise ValueError('Not GGUF')
        version,tensors,nmeta=num('I'),num('Q'),num('Q')
        info={'path':str(path),'version':version,'tensor_count':tensors,'metadata':{}}
        for _ in range(nmeta):
            key=string()
            t=num('I')
            keep=(key.startswith('general.') or not key.startswith('tokenizer.') or
                  (include_chat_template and key=='tokenizer.chat_template'))
            v=value(t,keep)
            if keep:
                info['metadata'][key]=v
        return info

if __name__=='__main__':
    import sys
    for p in sys.argv[1:]:
        print(json.dumps(inspect(p),indent=2))
