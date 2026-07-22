# Detecção de Anomalias em Licitações Públicas - eng-soft-ia-fw-2026

Projeto da disciplina de Engenharia de Software para IA e Frameworks Profundos.

## Descrição do Problema

Licitações públicas são processos legais que regulam a contratação de serviços e aquisição de bens pelo governo. Irregularidades nesses processos — como sobrepreço, direcionamento ou participação de empresas inidôneas — representam desvios de recursos públicos. Este projeto visa identificar automaticamente padrões suspeitos em dados de licitações federais.

## Base de Dados

Dados abertos do Portal da Transparência do Governo Federal, disponíveis em:
https://portaldatransparencia.gov.br/download-de-dados/licitacoes

Os arquivos mensais contêm quatro tabelas relacionadas:

| Arquivo | Conteúdo |
|---|---|
| `*_Licitação.csv` | Cabeçalho da licitação (modalidade, valor, órgão, situação) |
| `*_ItemLicitação.csv` | Itens contratados e vencedores por item |
| `*_ParticipantesLicitação.csv` | Empresas participantes e flag de vencedor |
| `*_EmpenhosRelacionados.csv` | Empenhos financeiros vinculados |

## Estrutura do Projeto
```text
project/
├── data/
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   └── main.py
├── README.md
└── requirements.txt

```

## Alunos
| Arquivo | Conteúdo |
|---|---|
| `Paulo Brandão` | proba@cin.ufpe.br |
| `Gustavo Bastos` | gcb3@cin.ufpe.br  |
| `Pedro Roncoli Sarmet Moreira` | prsm@cin.ufpe.br |
| `George Queiroz` | gjq@cin.ufpe.br |
| `César Calafrioli` | caccm@cin.ufpe.br |
| `Jhonata Lima` | jls3@cin.ufpe.br |
| `Anthony` | anthony.silv.ferreira@gmail.com |
