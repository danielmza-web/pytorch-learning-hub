"""Validate intrinsic teaching-image sizes, icons and retired bundle removal."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit, unquote
import xml.etree.ElementTree as ET
import re
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT/'site'
assert (SITE/'index.html').exists(), 'Build the library first'
assert not (SITE/'assets/javascripts/mermaid.min.js').exists()
checked = 0
class Parser(HTMLParser):
    def __init__(self): super().__init__(); self.images=[]; self.icons=[]
    def handle_starttag(self, tag, attributes):
        attrs=dict(attributes)
        if tag=='img': self.images.append(attrs)
        if tag=='link' and attrs.get('rel') in ('icon','apple-touch-icon'): self.icons.append(attrs)

for page in SITE.rglob('*.html'):
    parser=Parser(); parser.feed(page.read_text(encoding='utf-8'))
    assert {i['rel'] for i in parser.icons} >= {'icon','apple-touch-icon'}, page
    for attrs in parser.images:
        src=unquote(urlsplit(attrs.get('src','')).path)
        if '/assets/images/' not in '/'+src: continue
        path=(page.parent/src).resolve() if not src.startswith('/') else SITE/src.lstrip('/')
        if path.name=='dz-pytorch-mark.png': continue  # Theme owns header logo sizing.
        if path.suffix=='.svg':
            size=tuple(round(float(n)) for n in ET.parse(path).getroot().get('viewBox').split()[2:])
        else:
            with Image.open(path) as image: size=image.size
        assert (int(attrs.get('width',0)),int(attrs.get('height',0)))==size, (page,src,attrs)
        checked+=1
expected = sum(len(re.findall(r'!\[[^\]]*\]\([^)]+\)|<img\b', page.read_text(encoding='utf-8'))) for page in (ROOT/'docs').rglob('*.md'))
assert checked == expected and checked > 0, (checked, expected)
print(f'Presentation passed: {checked} teaching images with correct intrinsic sizes, both icons on every page, no retired bundle')
