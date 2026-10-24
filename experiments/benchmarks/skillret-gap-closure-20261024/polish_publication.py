"""Prose typography only; preserve raw evidence, URLs, code spans and audit receipts."""
import pathlib,re
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[2]
def polish(text):
    parts=re.split(r'(`[^`]*`|\]\([^\n)]*\)|https?://[^\s]+)',text)
    prefixes='on|retrieves|produces|retain|contain|of|freeze|with|nominal|seed|using|about|at|from|over|for|across|contains|planned|full|exposed|all|exact|requires|cap|and|before|after|within|difference|scope|under|mean|primary|balanced|additional|sensitivity'
    for i in range(0,len(parts),2):
        s=re.sub(r'\b('+prefixes+r')(?=\d)(?!\d+_)',r'\1 ',parts[i],flags=re.I)
        s=re.sub(r'(?<=\d)(queries|skills|points|tokens|resamples|calls|planned|valid|seconds|positive|percentage|million|minutes)\b',r' \1',s)
        s=re.sub(r',(?=\d)',', ',s) if not re.search(r'\d,\d{3}',s) else s
        for a,b in {'allfive':'all five','topfive':'top five','perN':'per N','byN':'by N','smallmodel':'small model','fullcatalog':'full catalog','balancedmacro':'balanced macro','pairedbootstrap':'paired bootstrap','samepanel':'same panel','sameselector':'same selector','freeVRAM':'free VRAM','guestRAM':'guest RAM','hostRAM':'host RAM','atn200':'at n=200','atn24':'at n=24','a95':'a 95','is+1.02':'is +1.02','=4000':'= 4,000','contain 10,100,1000,6006':'contain 10, 100, 1,000 and 6,006','IDF,k1=1.5,b=0.75':'IDF, k1=1.5, b=0.75','using 2000':'using 2,000',r'proposal\textquotesingle s':"proposal's"}.items():s=s.replace(a,b)
        parts[i]=s
    return ''.join(parts)
def main():
    files=[HERE/'report/RESEARCH_DRAFT.md',ROOT/'paper/final_report.tex',ROOT/'paper/presentation/research_draft.md',HERE/'README.md',HERE/'SPEC.md',HERE/'GAP_CLOSURE.md']
    for path in files:
        text=path.read_text();updated=polish(text)
        if updated!=text:path.write_text(updated)
    print('Publication prose spacing normalized; evidence bytes untouched')
if __name__=='__main__':main()
