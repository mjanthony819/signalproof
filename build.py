from pathlib import Path
import json
root=Path(__file__).parent
parts={'weights':'weights.json','results':'results.json','samples':'samples.json','predictions':'test_predictions.json'}
data={k:json.loads((root/'web'/v).read_text()) for k,v in parts.items()}
html=(root/'web/template.html').read_text().replace('/*EMBED_DATA*/','const DATA='+json.dumps(data,separators=(',',':'))+';').replace('/*INFERENCE*/',(root/'web/inference.js').read_text())
(root/'index.html').write_text(html)
print('Built index.html',len(html),'characters')
