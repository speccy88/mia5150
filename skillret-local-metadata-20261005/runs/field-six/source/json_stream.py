"""Bounded stdlib JSON-array reader for large public corpus files."""
import json
def items(path,chunk_size=65536):
 decoder=json.JSONDecoder()
 with open(path,encoding='utf-8') as f:
  assert f.read(1)=='[','Expected a JSON array';buffer='';eof=False;need_item=True;seen=False
  while True:
   buffer=buffer.lstrip()
   if not buffer and not eof:
    new=f.read(chunk_size);buffer+=new;eof=not new;continue
   if buffer.startswith(']'):
    assert not need_item or not seen,'Trailing comma'
    assert not (buffer[1:]+f.read()).strip(),'Trailing content';return
   if not need_item:
    assert buffer.startswith(','),'Missing JSON separator';buffer=buffer[1:];need_item=True;continue
   try:value,end=decoder.raw_decode(buffer)
   except json.JSONDecodeError:
    if eof:raise
    new=f.read(chunk_size);buffer+=new;eof=not new;continue
   yield value;buffer=buffer[end:];need_item=False;seen=True
