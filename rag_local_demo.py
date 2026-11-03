"""
RAG Local Demonstrador de NLP (Offline em CPU)
==============================================
Busca Semântica Densa (BGE-M3 + FAISS) vs. Busca Esparsa (Palavra-Chave) + RAG (Qwen)

Este script expande o rag_local.py original integrando explicações didáticas em tempo real
no terminal para demonstrar os principais conceitos de Processamento de Linguagem Natural (NLP/PLN).

Instalação de dependências:
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install sentence-transformers faiss-cpu transformers accelerate

Uso:
    python rag_local_demo.py --arquivo Epilogo2.txt
    python rag_local_demo.py --arquivo Epilogo2.txt --so-busca
    python rag_local_demo.py --arquivo Epilogo2.txt --janela 8 --passo 5 --k 5
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
    sys.exit(f"Dependência faltando: {e}\nVeja os comandos de instalação no topo do arquivo.")

# ----------------------------------------------------------------------
# CONFIGURAÇÕES PADRÃO
# ----------------------------------------------------------------------
MODELO_EMB = "BAAI/bge-m3"                 # 1024 dimensões semânticas
MODELO_LLM = "Qwen/Qwen2.5-0.5B-Instruct"  # 0.5B parâmetros (otimizado para CPU)
JANELA = 8        # linhas por trecho
PASSO = 5         # avanço em linhas
K = 5             # trechos recuperados
LIMIAR = 0.30     # similaridade mínima
MAX_TOKENS = 200

PROMPT_SISTEMA = (
    "Você é Eneias relatando fatos da sua perspectiva com base nesse texto. "
    "Responda em português usando SOMENTE os trechos fornecidos. "
    "Não invente nada. Seja claro, curto e fiel aos trechos. "
    "Se os trechos não trouxerem a resposta, diga apenas: 'Não sei dizer, não vi o ocorrido.'"
)

# ----------------------------------------------------------------------
# 1. CARREGAMENTO E FATIAMENTO DO TEXTO
# ----------------------------------------------------------------------
def carregar_texto(caminho: str) -> str:
    if not os.path.exists(caminho):
        sys.exit(f"Arquivo não encontrado: {caminho}")
    with open(caminho, "r", encoding="utf-8-sig") as f:
        texto = f.read().strip()
    if not texto:
        sys.exit("O arquivo está vazio.")
    print(f"  [✓] Arquivo Lido: {len(texto)} caracteres ({caminho})")
    return texto


def dividir_em_trechos(texto: str, janela: int, passo: int) -> list[str]:
    linhas = [l.strip() for l in texto.splitlines() if l.strip()]
    trechos, i = [], 0
    while i < len(linhas):
        trechos.append("\n".join(linhas[i:i + janela]))
        if i + janela >= len(linhas):
            break
        i += passo
    print(f"  [✓] Segmentação (Chunking): {len(trechos)} trechos gerados (janela={janela} linhas, sobreposição={janela-passo} linhas)")
    return trechos

# ----------------------------------------------------------------------
# 2. EMBEDDINGS + FAISS + CONCEITOS DE NLP
# ----------------------------------------------------------------------
def construir_indice(trechos: list[str], nome_modelo: str):
    print("\n" + "=" * 65)
    print(" 🚀 ETAPA 1: VETORIZAÇÃO E CRIAÇÃO DO ESPAÇO SEMÂNTICO (NLP)")
    print("=" * 65)
    print(f"  [..] Carregando modelo de Embedding Denso: {nome_modelo}")
    
    modelo = SentenceTransformer(nome_modelo, device="cpu")
    dimensao = modelo.get_sentence_embedding_dimension()
    
    print(f"  [..] Gerando vetores numéricos para {len(trechos)} trechos...")
    emb = modelo.encode(
        trechos, normalize_embeddings=True, batch_size=16, show_progress_bar=False
    ).astype("float32")
    
    indice = faiss.IndexFlatIP(dimensao)  # Produto interno = similaridade de cosseno (vetores normalizados)
    indice.add(emb)
    
    print(f"  [✓] Vetorização Concluída!")
    print(f"  📌 [CONCEITO NLP 1 - EMBEDDINGS]: Cada frase foi mapeada em um vetor denso de {dimensao} dimensões.")
    print(f"  📌 [CONCEITO NLP 2 - INDEXAÇÃO VETORIAL]: FAISS indexou {indice.ntotal} vetores para busca por produto interno MIPS em milissegundos.")
    return modelo, indice


def busca_esparsa_palavras(pergunta: str, trechos: list[str]) -> list[tuple[int, str]]:
    """Simula busca esparsa por sobreposição exata de palavras (Estilo TF-IDF/BM25 básico)."""
    palavras_q = set(w.lower() for w in pergunta.split() if len(w) > 2)
    resultados = []
    for t in trechos:
        t_words = set(w.lower() for w in t.split())
        score = len(palavras_q.intersection(t_words))
        if score > 0:
            resultados.append((score, t))
    resultados.sort(key=lambda x: x[0], reverse=True)
    return resultados


def buscar_densa(pergunta: str, modelo: SentenceTransformer, indice: faiss.IndexFlatIP, trechos: list[str], k: int):
    q_emb = modelo.encode([pergunta], normalize_embeddings=True).astype("float32")
    scores, ids = indice.search(q_emb, min(k, len(trechos)))
    return [(float(s), trechos[i]) for s, i in zip(scores[0], ids[0])]

# ----------------------------------------------------------------------
# 3. LLM (GERADOR RAG)
# ----------------------------------------------------------------------
def carregar_llm(nome_modelo: str):
    from transformers import pipeline

    print(f"\n  [..] Carregando Gerador LLM (RAG): {nome_modelo} em CPU...")
    gerador = pipeline(
        "text-generation",
        model=nome_modelo,
        model_kwargs={"torch_dtype": torch.float32},
        device=-1,
    )
    print("  [✓] LLM Carregado e Pronto!")
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
        do_sample=False,
        repetition_penalty=1.1,
        return_full_text=False,
        pad_token_id=gerador.tokenizer.eos_token_id,
    )[0]["generated_text"]
    return (out[-1]["content"] if isinstance(out, list) else out).strip()

# ----------------------------------------------------------------------
# 4. EXECUÇÃO DA CONSULTA COM DEMONSTRAÇÃO EXPLICATIVA
# ----------------------------------------------------------------------
def perguntar(pergunta, modelo, indice, trechos, gerador, args):
    print("\n" + "═" * 65)
    print(f" 🔍 PROCESSANDO CONSULTA DE NLP: \"{pergunta}\"")
    print("═" * 65)
    
    # --- DEMONSTRAÇÃO BUSCA ESPARSA VS DENSA ---
    t0 = time.time()
    res_densa = buscar_densa(pergunta, modelo, indice, trechos, args.k)
    t_densa = time.time() - t0
    
    res_esparsa = busca_esparsa_palavras(pergunta, trechos)

    print("\n 📊 1. COMPARATIVO DE RECUPERAÇÃO EM NLP:")
    print(" ---------------------------------------------------------------")
    
    # A) Busca por Palavra-Chave
    print(" A) BUSCA ESPARSA (Palavra-Chave / BM25 Tradicional):")
    if res_esparsa:
        print(f"    • Trechos com palavras exatas em comum: {len(res_esparsa)}")
        print(f"    • Melhor coincidência: {res_esparsa[0][0]} palavras sobrepostas.")
    else:
        print("    ❌ Nenhuma palavra exata coincidiu!")
        print("    📌 [LIMITE DO NLP TRADICIONAL]: Se a pergunta usa sinônimos, a busca por palavra-chave falha.")

    # B) Busca Densa (Embeddings)
    print("\n B) BUSCA DENSA (Embeddings Neurais / DPR / FAISS):")
    print(f"    • Pergunta vetorizada em {modelo.get_sentence_embedding_dimension()}D em {t_densa*1000:.1f}ms.")
    print(f"    • Maior Similaridade de Cosseno encontrada: {res_densa[0][0]:.4f}")
    print("    📌 [PODER DO NLP MODERNO]: O modelo entende o CONTEXTO e SINÔNIMOS, independente das palavras exatas.")

    # Exibição das evidências recuperadas
    print("\n 📚 2. EVIDÊNCIAS SELECIONADAS PELO FAISS (TOP RELEVÂNCIA):")
    print(" ---------------------------------------------------------------")
    for n, (s, t) in enumerate(res_densa[:2], 1):
        trecho_resumo = t.strip().replace("\n", " ")
        if len(trecho_resumo) > 110:
            trecho_resumo = trecho_resumo[:110] + "..."
        print(f"    [Trecho #{n}] (Score de Relevância: {s:.4f})")
        print(f"    └─ \"{trecho_resumo}\"")

    # --- ETAPA DE GERAÇÃO FUNDAMENTADA (RAG) ---
    print("\n 🤖 3. PROCESSAMENTO DO LLM (RAG & PREVENÇÃO DE ALUCINAÇÃO):")
    print(" ---------------------------------------------------------------")
    
    if gerador is None:
        print("    (Modo --so-busca ativado: LLM não carregado)")
    elif res_densa[0][0] < args.limiar:
        print(f"    ⚠️ [ALCANCE DE SEGURANÇA]: Score de similaridade ({res_densa[0][0]:.3f}) < Limiar configurado ({args.limiar}).")
        print("    📌 [CONTROLE DE ALUCINAÇÃO]: O sistema recusa responder pois a informação não está no texto.")
        print("\n 💬 RESPOSTA FINAL DO SISTEMA:")
        print("    \"O texto não informa isso. (Informação ausente no corpus)\"")
    else:
        print(f"    • Injetando os {len(res_densa)} trechos recuperados no Prompt de Sistema.")
        print("    • Executando inferência no Qwen LLM...")
        t1 = time.time()
        resposta = gerar_resposta(gerador, pergunta, res_densa, args.max_tokens)
        t_llm = time.time() - t1
        
        print(f"\n 💬 RESPOSTA GERADA PELO MODELO ({t_llm:.1f}s):")
        print(f"    \"{resposta}\"")

    print("\n" + "═" * 65 + "\n")


def main():
    p = argparse.ArgumentParser(description="Demonstrador Interativo de NLP e RAG Local")
    p.add_argument("--arquivo", required=True, help="Caminho do arquivo .txt de entrada")
    p.add_argument("--emb", default=MODELO_EMB, help="Modelo de embedding semântico")
    p.add_argument("--llm", default=MODELO_LLM, help="Modelo LLM gerador")
    p.add_argument("--janela", type=int, default=JANELA, help="Linhas por trecho (chunking)")
    p.add_argument("--passo", type=int, default=PASSO, help="Avanço em linhas por janela")
    p.add_argument("--k", type=int, default=K, help="Número de trechos recuperados")
    p.add_argument("--limiar", type=float, default=LIMIAR, help="Limiar de similaridade mínima")
    p.add_argument("--max-tokens", type=int, default=MAX_TOKENS, dest="max_tokens")
    p.add_argument("--so-busca", action="store_true", help="Executa apenas a recuperação vetorial")
    args = p.parse_args()

    if args.passo < 1 or args.passo > args.janela:
        sys.exit("--passo deve estar entre 1 e --janela.")

    print("\n" + "=" * 65)
    print(" 📖 INICIALIZANDO O SISTEMA DEMONSTRADOR DE NLP (RAG LOCAL)")
    print("=" * 65)
    
    texto = carregar_texto(args.arquivo)
    trechos = dividir_em_trechos(texto, args.janela, args.passo)
    modelo, indice = construir_indice(trechos, args.emb)
    gerador = None if args.so_busca else carregar_llm(args.llm)

    print("\n Prontinho! O ambiente de NLP está configurado.")
    print(" Digite sua pergunta no terminal (ou 'sair' para finalizar).\n")

    while True:
        try:
            pergunta = input("Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando sessão.")
            break
        if not pergunta:
            continue
        if pergunta.lower() in {"sair", "exit", "quit"}:
            print("Sessão encerrada.")
            break
        perguntar(pergunta, modelo, indice, trechos, gerador, args)


if __name__ == "__main__":
    main()
