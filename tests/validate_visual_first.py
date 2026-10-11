"""Check the published HTML contract, including the no-JavaScript overview."""
from html.parser import HTMLParser
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
VOID={'area','base','br','col','embed','hr','img','input','link','meta','param','source','track','wbr'}
class Node:
    def __init__(self,tag='',attrs=None,parent=None):
        self.tag,self.attrs,self.parent,self.children,self.text=tag,attrs or {},parent,[],''
    def ancestors(self):
        node=self.parent
        while node:
            yield node
            node=node.parent
    def descendants(self):
        for child in self.children:
            yield child
            yield from child.descendants()
class Page(HTMLParser):
    def __init__(self,source):
        super().__init__();self.root=Node();self.stack=[self.root];self.feed(source)
    def handle_starttag(self,tag,attrs):
        node=Node(tag,dict(attrs),self.stack[-1]);self.stack[-1].children.append(node)
        if tag not in VOID:self.stack.append(node)
    def handle_endtag(self,tag):
        for i in range(len(self.stack)-1,0,-1):
            if self.stack[i].tag==tag:
                self.stack=self.stack[:i];return
    def handle_data(self,data):self.stack[-1].text+=data

def check_site(base,dazu=False):
    pages=sorted(base.rglob('index.html'))
    assert len(pages)==(7 if dazu else 24),(base,len(pages))
    overviews=0;labs=[]
    for file in pages:
        page=Page(file.read_text(encoding='utf-8-sig'))
        main=next(n for n in page.root.descendants() if n.tag==('main' if dazu else 'article'))
        nodes=list(main.descendants());figures=[n for n in nodes if 'data-overview' in n.attrs]
        overviews+=len(figures)
        for figure in figures:
            assert not any(n.tag=='details' for n in figure.ancestors()),(file,'Hidden overview')
            assert not any(n.tag in {'input','select','button','canvas'} for n in figure.descendants()),(file,'Overview needs JS or controls')
            for svg in (n for n in figure.descendants() if n.tag=='svg'):
                assert svg.attrs.get('role')=='img' and svg.attrs.get('aria-label'),(file,'Missing SVG description')
        for node in nodes:
            kind=node.attrs.get('data-learning-lab')
            if kind:
                assert kind in ({'threshold'} if dazu else {'threshold','convolution','accumulation','attention','quantization'}),(file,kind)
                if kind not in {'threshold','convolution'}:
                    assert any(n.tag=='details' for n in node.ancestors()),(file,'Advanced lab is exposed')
                labs.append(kind)
            if node.tag in {'input','select'}:
                # Editable legacy calculators must be optional, including project tools.
                assert any(n.tag=='details' for n in node.ancestors()),(file,'Exposed calculator')
        if '/guides/' in file.as_posix() or (dazu and file.parent!=base):
            assert figures,(file,'Guide lacks a visible overview')
        assert not any('data-curve-lab' in n.attrs or 'data-regression-lab' in n.attrs or 'data-pipeline-lab' in n.attrs for n in nodes),(file,'Retired control')
    assert len(labs)==(1 if dazu else 5),(base,labs)
    return len(pages),overviews

subprocess.run(['node',str(ROOT/'scripts/render_learning_overviews.cjs'),'--check'],check=True,cwd=ROOT)
pages,overviews=check_site(ROOT/'site')
dazu=ROOT.parent/'site/learn/pytorch'
if dazu.exists():
    p,o=check_site(dazu,True);pages+=p;overviews+=o
print(f'Visual-first contract passed: {pages} pages, {overviews} visible HTML/SVG overviews; optional tools and no-JS content verified.')
