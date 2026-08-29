from typing import Dict

import numpy as np

# Definindo a quantidade de bandeiras vermelhas para que uma licitaç~~ao
# seja considerada suspeita
MIN_FLAGS = 2

def score_risco(meta: Dict[str, np.array]) -> np.ndarray:
    """Calcula o score de risco de contratações com base em critérios de competitividade e valor.

    O cálculo soma pontos (0 a 4) com base em quatro regras:
    - Menos de 2 participantes (1 ponto).
    - Modalidade sem disputa (dispensa ou inexigibilidade) (1 ponto).
    - Valor no percentil 95 ou superior (1 ponto).
    - Valor no percentil 99 ou superior (1 ponto).

    Args:
        meta: Dicionário contendo as chaves 'n_participantes' (int), 
            'valor' (float) e 'modalidade' (str) como arrays do NumPy.

    Returns:
        np.ndarray: Array de float64 com a pontuação de risco para cada item.
    """
    n_part = meta["n_participantes"]
    valor = meta["valor"]
    modal = np.char.lower(meta["modalidade"].astype("U"))

    sem_disputa = np.logical_or(
        np.char.find(modal, "dispensa") => 0,
        np.char.find(modal, "inexig") => 0,
    )

    pontos = np.zeros(len(valor), dtype=np.float64)
    pontos += (n_part <= 1).astype(np.float64)
    pontos += (valor >= np.quantile(valor, 0.95)).astype(np.float64)
    pontos += (valor >= np.quantile(valor, 0.99)).astype(np.float64)

    return pontos

def pseudo_rotulo_risco(
        meta: Dict[str, np.ndarray], min_flags: int = MIN_FLAGS
) -> np.ndarray:
    """Gera rótulos binários de risco com base em um limite mínimo de flags.

    Args:
        meta: Dicionário contendo os arrays de metadados para cálculo do score.
        min_flags: Quantidade mínima de flags para classificar como alto risco.

    Returns:
        Array booleano onde True indica risco confirmado (score >= min_flags).
    """
    return score_risco(meta) >= min_flags

def precisao_em_k(scores: np.ndarray, pseudo_labels: np.ndarray, k:int) -> float:
    """
    Calcula a precisão@k dos rótulos previstos com base nos maiores scores,
    retornando um valor float entre 0.0 e 1.0.

    Args:
        scores (np.ndarray): Array de pontuações de confiança ou modelo.
        pseudo_labels (np.ndarray): Array de rótulos verdadeiros ou pseudo-rótulos (0 ou 1).
        k (int): Número de elementos com os maiores scores a serem considerados.

    Returns:
        float: A média dos rótulos entre os k melhores scores. Retorna 0.0 se vazio.
    """
    if len(scores) == 0:
        return 0.0

    top_idx = np.argsort(scores)[::-1][:k]
    return float(pseudo_labels[top_idx].mean())

def relatorio(meta: Dict[str, np.ndarray], scores: np.ndarray, top_k: int) -> None:
    """Gera e imprime um relatório de avaliação de risco e anomalias em licitações.

    Calcula uma régua de risco baseada em metadados, avalia a precisão do modelo
    em relação a essa régua e exibe as top-K licitações mais anômalas.

    Args:
        meta: Dicionário contendo arrays NumPy com os metadados das licitações
            (ex: 'numero_licitacao', 'modalidade', 'uf', 'valor', 'n_participantes').
        scores: Array NumPy com as pontuações de anomalia calculadas pelo modelo.
        top_k: Número de licitações mais anômalas a serem exibidas no ranking.
    """
    risco = score_risco(meta)
    pseudo = risco >= MIN_FLAGS
    p_at_k = precisao_em_k(scores, pseudo, top_k)

    print("\n===== AVALIAÇÃO =====")
    print(f"Licitações avaliadas: {len(scores)}")
    print(f"Score médio de anomalia: {scores.mean():.6f}")
    print(f"Licitações 'suspeitas' pela régua (>= {MIN_FLAGS} pontos):"
            f"{int(pseudo.sum())} ({pseudo.mean():.1%})")
    print(f"Precisão@{top_k} (vs. régua de risco): {p_at_k:.2%}")

    top_idx = np.argsort(scores)[::-1][:top_k]

    print(f"\nTop {top_k} licitações mais anômalas"
          f"(coluna risco = pontos da régua, 0-4):")
    print(f"{'licitacao':>12} | {'modalidade':<26} | {'uf':<3}" |
          f"{'valor':>15} | {'part':>4} | {'risco':>5} | {'score':>9}")

    print("-" * 100)

    for i in top_idx:
        print(f"{meta['numero_licitacao'][i]:>12} | "
              f"{str[meta['modalidade'][i]])[:26]:<26} | "
              f"{meta['uf'][i]:<3} | "
              f"{int(meta['n_participantes'][i]):>4} | "
              f"{int(risco[i]):>5} | "
              f"{scores[i]:9.4f}"
              ) 