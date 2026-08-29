from typing import Dict, List, Tuple

import numpy as np

from src.data.loader import Tabela
from src.preprocessing.transform import parse_dates, parse_valor

# chave que identifica unicamente uma licitação (verificado nos dados reais)
KEY = ["numero_licitacao", "codigo_ug", "codigo_modalidade_compra"]
SEP = "\x1f"  # separador improvável de aparecer nos dados


# --------------------------------------------------------------------------- #
# Helpers de agrupamento/junção em NumPy puro
# --------------------------------------------------------------------------- #
def _composite_key(tabela: Tabela, fields: List[str]) -> np.ndarray:
    """Gera um array de chaves compostas concatenando colunas especificadas de uma tabela."""
    chave = tabela[fields[0]].astype("U")
    for f in fields[1:]:
        chave = np.char.add(np.char.add(chave, SEP), tabela[f].astype("U"))
    return chave


def _nunique_por_grupo(
    grupo_id: np.ndarray, valores: np.ndarray, n_grupos: int
) -> np.ndarray:
    """Calcula a quantidade de valores únicos por grupo em NumPy puro."""
    if grupo_id.size == 0:
        return np.zeros(n_grupos, dtype=np.float64)
    _, codigos = np.unique(valores, return_inverse=True)   # valor -> código inteiro
    base = int(codigos.max()) + 1
    pares = grupo_id.astype(np.int64) * base + codigos      # par (grupo, valor) único
    pares_unicos = np.unique(pares)
    grupos_dos_pares = (pares_unicos // base).astype(np.int64)
    return np.bincount(grupos_dos_pares, minlength=n_grupos).astype(np.float64)


def _soma_por_grupo(grupo_id: np.ndarray, valores: np.ndarray, n_grupos: int) -> np.ndarray:
    """Soma os valores numéricos por grupo."""
    out = np.zeros(n_grupos, dtype=np.float64)

    np.add.at(out, grupo_id, valores)

    return out


def _max_por_grupo(grupo_id: np.ndarray, valores: np.ndarray, n_grupos: int) -> np.ndarray:
    """Encontra o valor máximo numérico por grupo."""
    out = np.zeros(n_grupos, dtype=np.float64)

    if grupo_id.size:
        np.maximum.at(out, grupo_id, valores)

    return out


def _alinhar(
    lic_keys: np.ndarray, grupo_keys: np.ndarray, grupo_valores: np.ndarray
) -> np.ndarray:
    """Alinha métricas agregadas às licitações principais realizando uma busca/junção (join)."""
    ordem = np.argsort(grupo_keys)
    gk, gv = grupo_keys[ordem], grupo_valores[ordem]
    pos = np.searchsorted(gk, lic_keys)
    pos_clip = np.clip(pos, 0, len(gk) - 1) if len(gk) else np.zeros_like(pos)
    out = np.zeros(len(lic_keys), dtype=np.float64)

    if len(gk):
        casou = gk[pos_clip] == lic_keys
        out[casou] = gv[pos_clip][casou]
        
    return out


def _one_hot(valores: np.ndarray, prefixo: str) -> Tuple[np.ndarray, List[str]]:
    """Aplica codificação One-Hot Encoding em uma coluna de categorias."""
    categorias, inv = np.unique(valores.astype("U"), return_inverse=True)
    oh = np.zeros((len(valores), len(categorias)), dtype=np.float64)
    oh[np.arange(len(valores)), inv] = 1.0
    nomes = [f"{prefixo}_{c}" for c in categorias]
    
    return oh, nomes


# --------------------------------------------------------------------------- #
# Agregações por tabela
# --------------------------------------------------------------------------- #
def _agregar_itens(item: Tabela) -> Dict[str, np.ndarray]:
    """Agrega a tabela de itens agrupando por licitação e gerando métricas estatísticas de valores e contagens."""
    chaves = _composite_key(item, KEY)
    grupos, inv = np.unique(chaves, return_inverse=True)
    n = len(grupos)

    valor = np.nan_to_num(parse_valor(item["valor_item"]))
    return {
        "grupos": grupos,
        "n_itens": _nunique_por_grupo(inv, item["codigo_item_compra"], n),
        "valor_total_itens": _soma_por_grupo(inv, valor, n),
        "valor_item_max": _max_por_grupo(inv, valor, n),
        "n_linhas": np.bincount(inv, minlength=n).astype(np.float64),
        "soma_valor": _soma_por_grupo(inv, valor, n),
        "n_vencedores": _nunique_por_grupo(inv, item["codigo_vencedor"], n),
    }


def _agregar_participantes(part: Tabela) -> Dict[str, np.ndarray]:
    """Agrega a tabela de participantes por licitação, contabilizando total de participantes e vencedores."""
    chaves = _composite_key(part, KEY)
    grupos, inv = np.unique(chaves, return_inverse=True)
    n = len(grupos)

    flag = np.char.upper(np.char.strip(part["flag_vencedor"].astype("U")))
    eh_vencedor = np.char.startswith(flag, "S")

    return {
        "grupos": grupos,
        "n_participantes": _nunique_por_grupo(inv, part["codigo_participante"], n),
        "n_vencedores_part": _nunique_por_grupo(
            inv[eh_vencedor], part["codigo_participante"][eh_vencedor], n
        ),
    }


# --------------------------------------------------------------------------- #
# Montagem da matriz de features
# --------------------------------------------------------------------------- #
def build_feature_matrix(
    tables: Dict[str, Tabela],
    categorical_features: List[str],
) -> Tuple[np.ndarray, List[str], Dict[str, np.ndarray]]:
    """Constrói a matriz de features numérica (uma linha por licitação), lista de nomes das colunas e metadados."""
    lic = tables["licitacao"]
    lic_keys = _composite_key(lic, KEY)

    # --- campos numéricos/data da própria licitação ---
    valor_header = np.nan_to_num(parse_valor(lic["valor_licitacao"]))
    abertura = parse_dates(lic["data_abertura"])
    resultado = parse_dates(lic["data_resultado_compra"])
    dias = (resultado - abertura) / np.timedelta64(1, "D")  # NaN quando falta data

    flag_data_ausente = np.isnat(abertura).astype(np.float64)
    dias = np.nan_to_num(dias)

    # --- agregados de itens e participantes, alinhados às licitações ---
    it = _agregar_itens(tables["item"])
    pa = _agregar_participantes(tables["participantes"])

    n_itens = _alinhar(lic_keys, it["grupos"], it["n_itens"])
    valor_total_itens = _alinhar(lic_keys, it["grupos"], it["valor_total_itens"])
    valor_item_max = _alinhar(lic_keys, it["grupos"], it["valor_item_max"])
    n_vencedores = _alinhar(lic_keys, it["grupos"], it["n_vencedores"])
    soma_valor = _alinhar(lic_keys, it["grupos"], it["soma_valor"])
    n_linhas_item = _alinhar(lic_keys, it["grupos"], it["n_linhas"])
    valor_item_medio = np.where(n_linhas_item > 0, soma_valor / np.where(n_linhas_item == 0, 1, n_linhas_item), 0.0)

    n_participantes = _alinhar(lic_keys, pa["grupos"], pa["n_participantes"])
    n_venc_part = _alinhar(lic_keys, pa["grupos"], pa["n_vencedores_part"])
    razao = np.where(n_participantes > 0, n_venc_part / np.where(n_participantes == 0, 1, n_participantes), 0.0)

    valor_efetivo = np.where(valor_header > 0, valor_header, valor_total_itens)
    
    flag_sem_valor = ((valor_header == 0) & (valor_total_itens == 0)).astype(np.float64)

    log_valor_efetivo = np.log1p(np.clip(valor_efetivo, 0, None))
    log_valor_total_itens = np.log1p(np.clip(valor_total_itens, 0, None))

    # --- matriz numérica (ordem fixa e nomeada) ---
    numeric = {
        "log_valor_efetivo": log_valor_efetivo,
        "n_itens": n_itens,
        "log_valor_total_itens": log_valor_total_itens,
        "valor_item_medio": valor_item_medio,
        "valor_item_max": valor_item_max,
        "n_participantes": n_participantes,
        "n_vencedores": n_vencedores,
        "razao_vencedores_participantes": razao,
        "dias_abertura_resultado": dias,
        "flag_sem_valor": flag_sem_valor,
        "flag_data_ausente": flag_data_ausente,
    }
    numeric_names = list(numeric.keys())
    blocos = [np.column_stack(list(numeric.values()))]
    feature_names = list(numeric_names)

    for cat in categorical_features:
        coluna = lic["modalidade_compra"] if cat == "modalidade" else lic[cat]
        oh, nomes = _one_hot(coluna, cat)
        blocos.append(oh)
        feature_names.extend(nomes)

    X = np.column_stack(blocos).astype(np.float64)
    X = np.nan_to_num(X)

    # metadados legíveis (Apenas para interpretar os alertas)
    meta = {
        "numero_licitacao": lic["numero_licitacao"],
        "modalidade": lic["modalidade_compra"],
        "uf": lic["uf"],
        "municipio": lic["municipio"],
        "nome_orgao": lic["nome_orgao"],
        "valor": valor_efetivo,
        "n_participantes": n_participantes,
        "situacao": lic["situacao_licitacao"],
    }
    return X, feature_names, meta
