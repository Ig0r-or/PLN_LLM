import re
import nltk

from nltk import word_tokenize
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download("punkt_tab")
nltk.download("stopwords")

# 1. Nossa pergunta
pergunta = "Quem é Zoryel?"

# 2. Pequeno corpus inspirado na narrativa de Zoryel
corpus = [
    "Zoryel é um jovem troiano que cresceu nos arredores de Tróia.",
    "Zoryel foi colocado sob o comando de Enéias.",
    "Enéias era o guia e protetor das linhagens troianas.",
    "Heitor era o Escudo do Povo e Paris o Julgamento da Beleza.",
    "Zoryel buscava equilíbrio e compreendia o motivo da luta."
]

# 3. Tokenização, limpeza e Stopwords
stop_words = set(stopwords.words("portuguese"))
stop_words.update({"quem", "é"})

def preprocessar(texto):
    tokens = word_tokenize(texto, language="portuguese")

    tokens = [
        token.lower()
        for token in tokens
        if re.fullmatch(r"[^\W\d_]+", token, flags=re.UNICODE)
        and token.lower() not in stop_words
    ]

    return " ".join(tokens)


pergunta_limpa = preprocessar(pergunta)
corpus_limpo = [preprocessar(doc) for doc in corpus]

print("Tokens:", pergunta_limpa)

# 4. Bag of Words
bow = CountVectorizer()
matriz_bow = bow.fit_transform(corpus_limpo)

consulta_bow = bow.transform([pergunta_limpa])

print("\nBoW:")
print(dict(zip(
    bow.get_feature_names_out(),
    consulta_bow.toarray()[0]
)))

# 5. TF-IDF
tfidf = TfidfVectorizer()
matriz_tfidf = tfidf.fit_transform(corpus_limpo)

consulta_tfidf = tfidf.transform([pergunta_limpa])

print("\nTF-IDF:")
print(dict(zip(
    tfidf.get_feature_names_out(),
    consulta_tfidf.toarray()[0].round(3)
)))

# 6. Similaridade entre pergunta e documentos
scores = (consulta_tfidf @ matriz_tfidf.T).toarray()[0]

print("\nRanking dos documentos:")

for i in scores.argsort()[::-1]:
    print(f"Documento {i + 1}: {scores[i]:.3f}")