"""
Chat web local com grafico de similaridade (mostra o "ciclo de pensamento" do RAG).
Reutiliza o rag_local.py (deixe os dois arquivos na mesma pasta). Sem dependencias extras.

Uso:
    python rag_web.py --arquivo Epilogo2.txt
    python rag_web.py --arquivo Epilogo2.txt --so-busca --janela 5 --passo 3
Abre em http://localhost:8000
"""
import argparse
import json
import sys
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import rag_local as R

E = {}  # estado global: modelo, indice, trechos, gerador, limiar...


def responder(pergunta, k):
    n = len(E["trechos"])
    t0 = time.time()
    q = E["modelo"].encode([pergunta], normalize_embeddings=True).astype("float32")
    t_emb = time.time() - t0

    t0 = time.time()
    scores, ids = E["indice"].search(q, n)  # similaridade de TODOS os trechos
    t_busca = time.time() - t0

    todos = [0.0] * n
    for s, i in zip(scores[0], ids[0]):
        todos[int(i)] = float(s)
    top = [{"idx": int(i), "score": float(s), "texto": E["trechos"][int(i)]}
           for s, i in zip(scores[0][:k], ids[0][:k])]
    passou = top[0]["score"] >= E["limiar"]

    resposta, t_llm, prompt = "", 0.0, ""
    if E["gerador"] is None:
        resposta = "(modo --so-busca: LLM desativado)"
    elif not passou:
        resposta = "O texto não informa isso."
    else:
        pares = [(x["score"], x["texto"]) for x in top]
        prompt = "\n\n".join(f"[Trecho {i}]\n{t.strip()}" for i, (_, t) in enumerate(pares, 1))
        t0 = time.time()
        resposta = R.gerar_resposta(E["gerador"], pergunta, pares, E["max_tokens"])
        t_llm = time.time() - t0

    return {"resposta": resposta, "todos": todos, "top": top, "limiar": E["limiar"],
            "passou": passou, "prompt": prompt,
            "tempos": {"emb": t_emb, "busca": t_busca, "llm": t_llm}}


class H(BaseHTTPRequestHandler):
    def _send(self, code, body, tipo):
        self.send_response(code)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/info":
            info = {"trechos": len(E["trechos"]), "k": E["k"], "arquivo": E["arquivo"]}
            self._send(200, json.dumps(info).encode(), "application/json")
        else:
            self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")

    def do_POST(self):
        try:
            dados = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            out = responder(dados["pergunta"], max(1, int(dados.get("k", E["k"]))))
            self._send(200, json.dumps(out).encode(), "application/json")
        except Exception as e:
            self._send(500, json.dumps({"erro": str(e)}).encode(), "application/json")

    def log_message(self, *a):
        pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arquivo", required=True)
    p.add_argument("--emb", default=R.MODELO_EMB)
    p.add_argument("--llm", default=R.MODELO_LLM)
    p.add_argument("--janela", type=int, default=R.JANELA)
    p.add_argument("--passo", type=int, default=R.PASSO)
    p.add_argument("--k", type=int, default=R.K)
    p.add_argument("--limiar", type=float, default=R.LIMIAR)
    p.add_argument("--max-tokens", type=int, default=R.MAX_TOKENS, dest="max_tokens")
    p.add_argument("--so-busca", action="store_true")
    p.add_argument("--porta", type=int, default=8000)
    a = p.parse_args()
    if a.passo < 1 or a.passo > a.janela:
        sys.exit("--passo deve estar entre 1 e --janela.")

    texto = R.carregar_texto(a.arquivo)
    trechos = R.dividir_em_trechos(texto, a.janela, a.passo)
    modelo, indice = R.construir_indice(trechos, a.emb)
    gerador = None if a.so_busca else R.carregar_llm(a.llm)
    E.update(modelo=modelo, indice=indice, trechos=trechos, gerador=gerador, k=a.k,
             limiar=a.limiar, max_tokens=a.max_tokens, arquivo=a.arquivo)

    url = f"http://localhost:{a.porta}"
    print(f"\n[ok] Abra {url}  (Ctrl+C para encerrar)")
    webbrowser.open(url)
    try:
        ThreadingHTTPServer(("127.0.0.1", a.porta), H).serve_forever()
    except KeyboardInterrupt:
        print("\nEncerrado.")


HTML = """<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>RAG — ciclo de pensamento</title>
<style>
:root{--bg:#fafaf8;--fg:#1c1c1a;--mut:#8a8a84;--card:#fff;--bd:#e2e1dc;--acc:#2f6fed;--ok:#1d9a6c;--bad:#d4493a}
@media(prefers-color-scheme:dark){:root{--bg:#161614;--fg:#ecebe6;--mut:#8f8e87;--card:#1f1f1c;--bd:#33322e;--acc:#6b9bff;--ok:#3cc08f;--bad:#ef6c5d}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif;height:100vh;display:grid;grid-template-columns:minmax(300px,36%) 1fr}
.col{display:flex;flex-direction:column;min-height:0;padding:16px;gap:12px}.col+.col{border-left:1px solid var(--bd);overflow:auto}
h2{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--mut);margin:0 0 6px;font-weight:600}
#chat{flex:1;overflow:auto;display:flex;flex-direction:column;gap:8px}
.m{padding:8px 12px;border-radius:10px;max-width:92%;white-space:pre-wrap}.user{align-self:flex-end;background:var(--acc);color:#fff}.bot{background:var(--card);border:1px solid var(--bd)}
form{display:flex;gap:8px}input[type=text]{flex:1;padding:10px;border-radius:8px;border:1px solid var(--bd);background:var(--card);color:var(--fg)}
button{padding:10px 14px;border-radius:8px;border:0;background:var(--acc);color:#fff;cursor:pointer}
.card{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:12px}
.steps{display:grid;grid-template-columns:repeat(5,1fr);gap:8px}.st{font-size:12px}.st b{display:block;font-size:13px}.st span{color:var(--mut)}
.st.ok b{color:var(--ok)}.st.bad b{color:var(--bad)}
.tk{display:grid;grid-template-columns:70px 1fr 50px;gap:8px;align-items:center;font-size:12px;cursor:pointer;margin:4px 0}
.bar{height:8px;border-radius:4px;background:var(--acc)}
#det{white-space:pre-wrap;font-size:13px;max-height:200px;overflow:auto}
small{color:var(--mut)}
</style></head><body>
<div class="col"><h2>Chat</h2><div id="chat"></div>
<form onsubmit="enviar(event)"><input id="q" type="text" placeholder="Pergunte algo sobre o texto…" autofocus><button>Enviar</button></form>
<small>k = <input id="k" type="number" min="1" max="20" value="5" style="width:50px"> trechos enviados ao LLM · <span id="info"></span></small></div>
<div class="col">
<div class="card"><h2>1 · Ciclo de pensamento</h2><div class="steps" id="steps"><small>Faça uma pergunta para ver as etapas.</small></div></div>
<div class="card"><h2>2 · Similaridade da pergunta com cada trecho do texto</h2><svg id="gr" viewBox="0 0 700 230" width="100%"></svg>
<small>Cada barra é um trecho, na ordem do documento. Azul = enviado ao LLM · linha tracejada = limiar. Clique numa barra para ler o trecho.</small></div>
<div class="card"><h2>3 · Top trechos recuperados</h2><div id="top"></div></div>
<div class="card"><h2>Trecho selecionado</h2><div id="det"><small>Nenhum.</small></div></div>
</div>
<script>
const $=s=>document.querySelector(s),esc=t=>t.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
fetch('/info').then(r=>r.json()).then(i=>{$('#info').textContent=i.trechos+' trechos · '+i.arquivo;$('#k').value=i.k});
function add(c,t){const e=document.createElement('div');e.className='m '+c;e.textContent=t;$('#chat').append(e);e.scrollIntoView();return e}
async function enviar(ev){ev.preventDefault();const p=$('#q').value.trim();if(!p)return;$('#q').value='';add('user',p);const el=add('bot','Pensando…');
try{const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({pergunta:p,k:+$('#k').value})});
const d=await r.json();if(d.erro)throw new Error(d.erro);el.textContent=d.resposta;desenhar(d)}catch(e){el.textContent='Erro: '+e.message}}
function sel(d,i){const t=d.top.find(x=>x.idx===i);$('#det').innerHTML='<small>Trecho '+i+' · similaridade '+d.todos[i].toFixed(3)+'</small>\\n'+esc(t?t.texto:d.textos?d.textos[i]:'(fora do top-k: aumente o k para ler)')}
function desenhar(d){const T=d.tempos,ms=x=>(x*1000).toFixed(0)+' ms',best=d.top[0].score;
$('#steps').innerHTML=[['Pergunta → vetor',ms(T.emb),1],['Busca FAISS',ms(T.busca)+' · '+d.todos.length+' trechos',1],
['Limiar',best.toFixed(2)+(d.passou?' ≥ ':' < ')+d.limiar,d.passou],['Prompt',d.top.length+' trechos',d.passou&&d.prompt!==''],['LLM',T.llm?ms(T.llm):'—',d.passou]]
.map(s=>'<div class="st '+(s[2]?'ok':'bad')+'"><b>'+s[0]+'</b><span>'+s[1]+'</span></div>').join('');
const n=d.todos.length,W=700,H=230,L=34,B=24,Tp=10,min=Math.min(0,...d.todos),max=Math.max(...d.todos,d.limiar)*1.1,y=v=>Tp+(H-Tp-B)*(1-(v-min)/(max-min)),bw=(W-L)/n,tops=new Set(d.top.map(x=>x.idx));
let s='<line x1="'+L+'" x2="'+W+'" y1="'+y(0)+'" y2="'+y(0)+'" stroke="var(--bd)"/>';
d.todos.forEach((v,i)=>{s+='<rect data-i="'+i+'" x="'+(L+i*bw+1)+'" y="'+Math.min(y(v),y(0))+'" width="'+Math.max(bw-2,1)+'" height="'+Math.abs(y(v)-y(0))+'" fill="'+(tops.has(i)?'var(--acc)':'var(--mut)')+'" opacity="'+(tops.has(i)?1:.45)+'" style="cursor:pointer"><title>trecho '+i+' · '+v.toFixed(3)+'</title></rect>'});
s+='<line x1="'+L+'" x2="'+W+'" y1="'+y(d.limiar)+'" y2="'+y(d.limiar)+'" stroke="var(--bad)" stroke-dasharray="5 4"/><text x="'+(W-2)+'" y="'+(y(d.limiar)-4)+'" text-anchor="end" font-size="10" fill="var(--bad)">limiar '+d.limiar+'</text>';
s+='<text x="'+L+'" y="'+(H-6)+'" font-size="10" fill="var(--mut)">início do texto</text><text x="'+W+'" y="'+(H-6)+'" text-anchor="end" font-size="10" fill="var(--mut)">fim</text><text x="2" y="'+(Tp+8)+'" font-size="10" fill="var(--mut)">'+max.toFixed(2)+'</text>';
$('#gr').innerHTML=s;$('#gr').querySelectorAll('rect').forEach(r=>r.onclick=()=>sel(d,+r.dataset.i));
$('#top').innerHTML=d.top.map((t,i)=>'<div class="tk" data-i="'+t.idx+'"><span>#'+(i+1)+' · trecho '+t.idx+'</span><div class="bar" style="width:'+Math.max(t.score/max*100,1)+'%"></div><span>'+t.score.toFixed(3)+'</span></div>').join('');
$('#top').querySelectorAll('.tk').forEach(e=>e.onclick=()=>sel(d,+e.dataset.i));sel(d,d.top[0].idx)}
</script></body></html>
"""

if __name__ == "__main__":
    main()