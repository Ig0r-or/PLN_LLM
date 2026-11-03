"""
RAG local (offline) em CPU: busca semantica (embeddings + FAISS) + LLM pequeno.

Instalacao (uma vez, de preferencia dentro de um venv):
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install sentence-transformers faiss-cpu transformers accelerate

Uso:
    python rag_local.py --arquivo Epilogo2.txt
    python rag_local.py --arquivo Epilogo2.txt --so-busca      # sem LLM (rapido, bom pra testar a busca)
    python rag_local.py --arquivo Epilogo2.txt --janela 6 --passo 4 --k 5

Na 1a execucao os modelos sao baixados (~2,5 GB no total) e ficam em cache.
Depois disso funciona totalmente offline.
"""

import argparse
import os
import sys
import time

try:
    import numpy as np
    import faiss
    import torch
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    sys.exit(f"Dependencia faltando: {e}\nVeja os comandos de instalacao no topo do arquivo.")

# ----------------------------------------------------------------------
# CONFIGURACOES PADRAO (podem ser sobrescritas por argumentos de linha)
# ----------------------------------------------------------------------
MODELO_EMB = "BAAI/bge-m3"                 # leve? troque por "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODELO_LLM = "Qwen/Qwen2.5-0.5B-Instruct"  # 0.5B roda bem na CPU
JANELA = 8        # linhas por trecho
PASSO = 5         # avanco em linhas (JANELA - PASSO = sobreposicao)
K = 5             # trechos recuperados
LIMIAR = 0.30     # similaridade minima do melhor trecho (scores variam por modelo: teste!)
MAX_TOKENS = 350

PROMPT_SISTEMA = (
    "Voce e um assistente de IA que responde perguntas com base em trechos de texto fornecidos. "
    "Responda em portugues usando SOMENTE os trechos fornecidos. "
    "Nao invente nada. Seja claro, curto e fiel aos trechos. "
    "Se os trechos nao trouxerem a resposta, diga apenas: 'Nao sei dizer, nao vi o ocorrido.'"
)


# ----------------------------------------------------------------------
# 1. TEXTO E TRECHOS
# ----------------------------------------------------------------------
def carregar_texto(caminho: str) -> str:
    if not os.path.exists(caminho):
        sys.exit(f"Arquivo nao encontrado: {caminho}")
    # utf-8-sig remove o BOM (caractere invisivel que alguns .txt trazem no inicio)
    with open(caminho, "r", encoding="utf-8-sig") as f:
        texto = f.read().strip()
    if not texto:
        sys.exit("O arquivo esta vazio.")
    print(f"[ok] Texto: {len(texto)} caracteres ({caminho})")
    return texto


def dividir_em_trechos(texto: str, janela: int, passo: int) -> list[str]:
    """Agrupa LINHAS inteiras (nunca corta frase/palavra no meio)."""
    linhas = [l.strip() for l in texto.splitlines() if l.strip()]
    trechos, i = [], 0
    while i < len(linhas):
        trechos.append("\n".join(linhas[i:i + janela]))
        if i + janela >= len(linhas):
            break
        i += passo
    print(f"[ok] {len(trechos)} trechos (janela={janela} linhas, passo={passo})")
    return trechos


# ----------------------------------------------------------------------
# 2. EMBEDDINGS + FAISS
# ----------------------------------------------------------------------
def construir_indice(trechos: list[str], nome_modelo: str):
    print(f"[..] Carregando embedding: {nome_modelo}")
    modelo = SentenceTransformer(nome_modelo, device="cpu")
    emb = modelo.encode(
        trechos, normalize_embeddings=True, batch_size=16, show_progress_bar=True
    ).astype("float32")
    indice = faiss.IndexFlatIP(emb.shape[1])  # produto interno = cosseno (vetores normalizados)
    indice.add(emb)
    print(f"[ok] Indice FAISS: {indice.ntotal} vetores de dimensao {emb.shape[1]}")
    return modelo, indice


def buscar(pergunta, modelo, indice, trechos, k):
    q = modelo.encode([pergunta], normalize_embeddings=True).astype("float32")
    scores, ids = indice.search(q, min(k, len(trechos)))
    return [(float(s), trechos[i]) for s, i in zip(scores[0], ids[0])]


# ----------------------------------------------------------------------
# 3. LLM (opcional)
# ----------------------------------------------------------------------
def carregar_llm(nome_modelo: str):
    from transformers import pipeline

    print(f"[..] Carregando LLM: {nome_modelo} (CPU)")
    gerador = pipeline(
        "text-generation",
        model=nome_modelo,
        model_kwargs={"torch_dtype": torch.float32},  # float32: o mais estavel em CPU
        device=-1,                                    # -1 = CPU
    )
    print("[ok] LLM pronto.")
    return gerador


def gerar_resposta(gerador, pergunta, resultados, max_tokens):
    contexto = "\n\n".join(
        f"[Trecho {n}]\n{t.strip()}" for n, (_, t) in enumerate(resultados, 1)
    )
    mensagens = [
        {"role": "system", "content": PROMPT_SISTEMA},
        {"role": "user", "content": f"{contexto}\n\nPergunta: {pergunta}"},
    ]
    out = gerador(
        mensagens,
        max_new_tokens=max_tokens,
        do_sample=False,              # deterministico: mudancas vem so do contexto
        repetition_penalty=1.1,
        return_full_text=False,
        pad_token_id=gerador.tokenizer.eos_token_id,
    )[0]["generated_text"]
    return (out[-1]["content"] if isinstance(out, list) else out).strip()


# ----------------------------------------------------------------------
# 4. PERGUNTAR
# ----------------------------------------------------------------------
def perguntar(pergunta, modelo, indice, trechos, gerador, args):
    t0 = time.time()
    resultados = buscar(pergunta, modelo, indice, trechos, args.k)
    t_busca = time.time() - t0

    print("\n=== PERGUNTA ===")
    print(pergunta)

    if gerador is None:
        print("\n=== RESPOSTA ===\n(modo --so-busca: LLM desativado)")
    elif resultados[0][0] < args.limiar:
        print("\n=== RESPOSTA ===\nO texto nao informa isso. "
              f"(melhor similaridade {resultados[0][0]:.3f} < limiar {args.limiar})")
    else:
        t1 = time.time()
        resposta = gerar_resposta(gerador, pergunta, resultados, args.max_tokens)
        print(f"\n=== RESPOSTA === ({time.time() - t1:.1f}s)\n{resposta}")

    print(f"\n=== EVIDENCIAS === (busca: {t_busca:.2f}s)")
    for n, (s, t) in enumerate(resultados, 1):
        print(f"\n--- Trecho {n} | similaridade: {s:.3f} ---\n{t.strip()}")
    print()


def main():
    p = argparse.ArgumentParser(description="RAG local offline (CPU)")
    p.add_argument("--arquivo", required=True, help="caminho do .txt")
    p.add_argument("--emb", default=MODELO_EMB, help="modelo de embedding")
    p.add_argument("--llm", default=MODELO_LLM, help="modelo LLM")
    p.add_argument("--janela", type=int, default=JANELA, help="linhas por trecho")
    p.add_argument("--passo", type=int, default=PASSO, help="avanco em linhas")
    p.add_argument("--k", type=int, default=K, help="trechos recuperados")
    p.add_argument("--limiar", type=float, default=LIMIAR, help="similaridade minima")
    p.add_argument("--max-tokens", type=int, default=MAX_TOKENS, dest="max_tokens")
    p.add_argument("--so-busca", action="store_true", help="nao carrega o LLM")
    args = p.parse_args()

    if args.passo < 1 or args.passo > args.janela:
        sys.exit("--passo deve estar entre 1 e --janela.")

    texto = carregar_texto(args.arquivo)
    trechos = dividir_em_trechos(texto, args.janela, args.passo)
    modelo, indice = construir_indice(trechos, args.emb)
    gerador = None if args.so_busca else carregar_llm(args.llm)

    print("\n" + "=" * 55)
    print("Pronto! Digite sua pergunta ('sair' para encerrar).")
    print("=" * 55 + "\n")

    while True:
        try:
            pergunta = input("Voce: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando.")
            break
        if not pergunta:
            continue
        if pergunta.lower() in {"sair", "exit", "quit"}:
            break
        perguntar(pergunta, modelo, indice, trechos, gerador, args)


if __name__ == "__main__":
    main()
