from pathlib import Path
here=Path(__file__).resolve().parent
html=(here/'hybrid-demo.html').read_text();qa=(here/'demo_qa.js').read_text()
(here/'hybrid-demo-qa.html').write_text(html.replace('</body>','<script>'+qa+'</script></body>'))
