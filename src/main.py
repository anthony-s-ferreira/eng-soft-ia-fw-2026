from src.data.loader import load_tables
from src.evaluation.evaluate import relatorio
from src.features.feature import build_feature_matrix
from src.preprocessing.transform import standardize, train_test_split_numpy
from src.training.train import (
    compute_anomaly_scores,
    save_artifacts,
    train_autoencoder,
)
from src.utils.config import CONFIG


def main() -> None:
    cfg = CONFIG


    print(f"Carregando dados das 4 tabelas dos ZIPS contidos em: {cfg.data_dir}", end=" ")
    tables = load_tables(cfg.data_dir)
    linhas = {nome: len(table) for nome, table in tables.items()}
    print("\033[32mOK\033[0m")
    print("Linhas por tabela:", linhas)

    print("\nMontando a matriz de features (1 linha por licitação) ", end=" ")
    X_raw, feature_names, meta = build_feature_matrix(
        tables, cfg.categorical_features
    )
    print("\033[32mOK\033[0m")
    print(f"\nMatriz de features: {X_raw.shape[0]} licitações x {X_raw.shape[1]} colunas")


    # 3. PADRONIZAR (NumPy) — guardamos mean/std para a inferência
    print("\nPadronizando com Numpy e guardando mean/std para a inferência ", end=" ")
    X_std, mean, std = standardize(X_raw)
    print("\033[32mOK\033[0m")

    # 4. SPLIT treino/teste 
    print("\nDividindo os dados em treino e teste ", end=" ")  
    X_train, X_test = train_test_split_numpy(
        X_std, test_size=cfg.test_size, seed=cfg.random_seed
    )
    print("\033[32mOK\033[0m")

    # 5. TREINAR (imprime erro de treino e teste a cada época)
    model, _history = train_autoencoder(X_train, X_test, cfg)
    print("\033[32mOK\033[0m")

    # 6. SALVAR modelo + scaler + nomes das colunas
    print("\nSalvando o modelo mais o scaler e o nome das colunas" , end=" ")
    save_artifacts(model, mean, std, feature_names, cfg)
    print("\033[32mOK\033[0m")

    # 7. AVALIAR: score de todas as licitações + top-k anômalas
    print("\nAvaliando o modelo treinado ", end=" ")
    scores = compute_anomaly_scores(model, X_std)
    relatorio(meta, scores, cfg.top_k)
    print("\033[32mOK\033[0m")


if __name__ == "__main__":
    main()
