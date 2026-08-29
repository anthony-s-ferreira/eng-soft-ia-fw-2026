# Detecção de Anomalias em Licitações Públicas - eng-soft-ia-fw-2026

Projeto da disciplina de Engenharia de Software para IA e Frameworks Profundos.

## Descrição

Este repositório contém código, notebooks e testes para detectar anomalias em dados de licitações públicas (Portal da Transparência). O objetivo é explorar métodos como Isolation Forest e Autoencoder para localizar padrões incomuns que possam indicar irregularidades.

## Conteúdo do repositório

Resumo dos principais arquivos e diretórios:

- notebook/
  - deteccao_anomalias_licitacoes.ipynb  — notebook principal de exploração e demonstração
  - entrega_4/exercicios_intermediarios.ipynb
  - entrega_3/entrega_3.ipynb

- docs/
  - Licitacoes-dicionario-dados.md — dicionário de dados / descrição dos campos das bases

- src/ (código principal)
  - src/data/
    - loader.py — carregamento dos CSVs mensais
    - dataset.py — abstração de dataset (Dataset/iteradores)
    - prepare_data.py — preparação e limpeza de dados
  - src/preprocessing/
    - transform.py — transformações / engenharia de features
  - src/features/
    - feature.py — extração / composição de features
  - src/models/
    - isolation_forest.py — wrapper / definição do modelo Isolation Forest
    - autoencoder.py — arquitetura de autoencoder (anomaly detector)
    - model.py — interfaces e helpers para modelos
  - src/training/
    - train.py — pipeline de treinamento genérico
    - train_isolation_forest.py — script de treino específico para Isolation Forest
  - src/inference/
    - inference.py — rotina para inferência / marcar anomalias
  - src/evaluation/
    - evaluate.py — métricas e rotinas de avaliação
  - src/utils/
    - config.py — configuração / constantes
  - src/main.py — ponto de entrada / exemplos de execução

- tests/ (testes automatizados com pytest)
  - test_data.py
  - test_preprocessing.py
  - test_model.py
  - test_training.py

- outros arquivos
  - README.md (este arquivo)
  - docs/Licitacoes-dicionario-dados.md
  - requirements.txt (dependências do projeto)


## Como instalar dependências

Recomenda-se usar um ambiente virtual. Exemplo com venv + pip:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # PowerShell
pip install --upgrade pip
pip install -r requirements.txt
```

## Como executar

- Rodar os notebooks: abrir `notebook/*.ipynb` com Jupyter (Jupyter Lab ou Notebook).
- Treinar modelos:
  - Treino genérico: `python -m src.training.train` (ou `python src\training\train.py`)
  - Treino Isolation Forest: `python src\training\train_isolation_forest.py`
- Fazer inferência: `python -m src.inference.inference` ou `python src\inference\inference.py`
- Executar o módulo principal de exemplo: `python src\main.py`

(Verificar parâmetros e caminhos dentro dos scripts/configs em `src/utils/config.py`.)

## Testes

Executar a suíte com pytest:

```powershell
pytest -q
```

## Dados

Os dados de entrada devem ser obtidos no Portal da Transparência: https://portaldatransparencia.gov.br/download-de-dados/licitacoes
Os arquivos mensais típicos incluem: `*_Licitação.csv`, `*_ItemLicitação.csv`, `*_ParticipantesLicitação.csv`, `*_EmpenhosRelacionados.csv`.
O dicionário de campos está em `docs/Licitacoes-dicionario-dados.md`.

## Estrutura esperada / Observações

- Scripts esperam encontrar dados convertidos/normalizados em `data/` (ver `src/data/loader.py` e `src/data/prepare_data.py`).
- Configurações de caminhos e parâmetros principais ficam em `src/utils/config.py`.

## Contribuição / Autores

Alunos e colaboradores listados originalmente no projeto. Para contribuir, abrir uma issue ou enviar PR com descrições claras das mudanças.

| Arquivo | Conteúdo          |
|---|-------------------|
| `Paulo Brandão` | proba@cin.ufpe.br |
| `Gustavo Bastos` | gcb3@cin.ufpe.br  |
| `Pedro Roncoli Sarmet Moreira` | prsm@cin.ufpe.br  |
| `George Queiroz` | gjq@cin.ufpe.br   |
| `César Calafrioli` | caccm@cin.ufpe.br |
| `Jhonata Lima` | jls3@cin.ufpe.br  |
| `Anthony Ferreira` | asf8@cin.ufpe.com |
| `Bruno Oliveira` | brunooliveirapereir@gmail.com |
