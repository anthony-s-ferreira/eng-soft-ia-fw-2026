---
title: "Dicionário de Dados — Licitações"
source: "https://portaldatransparencia.gov.br/dicionario-de-dados/licitacoes"
fonte_dados: "Portal da Transparência / CGU"
created: 2026-06-16
tags:
  - licitacoes
  - dicionario-de-dados
---

# Dicionário de Dados — Licitações

Cada arquivo `<AAAAMM>_Licitacoes.zip` contém quatro tabelas (CSV), descritas abaixo.

> **Nomes das colunas:** os nomes na coluna **Coluna** são exatamente como aparecem no cabeçalho dos CSVs reais (codificação latin-1, separador `;`). Onde isso difere do dicionário oficial do site, há uma observação na seção [Diferenças entre o site e os arquivos](#diferenças-entre-o-site-e-os-arquivos).

---

## EmpenhosRelacionados

| Coluna | Descrição |
| --- | --- |
| Número Licitação | Número que identifica a licitação no SIASG. |
| Código UG | Código da Unidade Gestora (UG) responsável pela licitação. Ver [Definições](#definições). |
| Nome UG | Nome da Unidade Gestora. |
| Código Modalidade Compra | Código da modalidade de compra. Ver [tabela de modalidades](#códigos-de-modalidade-de-compra). |
| Modalidade Compra | Modalidade da compra: Concorrência; Concurso; Convite; Dispensa de Licitação; Inexigibilidade de Licitação; Pregão; Registro de Preço; Tomada de Preços. |
| Número Processo | Número do processo da licitação. |
| Código Empenho | Código do empenho da licitação. |
| Data Emissão Empenho | Data de emissão do empenho. |
| Observação Empenho | Observação do empenho. |
| Valor Empenho (R$) | Valor do empenho, em reais. |

---

## ItemLicitação

| Coluna | Descrição |
| --- | --- |
| Número Licitação | Número que identifica a licitação no SIASG. |
| Código UG | Código da Unidade Gestora (UG) responsável pela licitação. Ver [Definições](#definições). |
| Nome UG | Nome da Unidade Gestora. |
| Código Modalidade Compra | Código da modalidade de compra. Ver [tabela de modalidades](#códigos-de-modalidade-de-compra). |
| Modalidade Compra | Modalidade da compra: Concorrência; Concurso; Convite; Dispensa de Licitação; Inexigibilidade de Licitação; Pregão; Registro de Preço; Tomada de Preços. |
| Número Processo | Número do processo da licitação. |
| Código Órgão | Código do órgão (subordinado) responsável pela licitação. Ver [Definições](#definições). |
| Nome Órgão | Nome do órgão. |
| Código Item Compra | Código do item da compra no SIASG — número de 22 dígitos. Ver [composição](#composição-do-código-item-compra-22-dígitos). |
| Descrição | Descrição do item da compra no SIASG. |
| Quantidade Item | Quantidade do item. |
| Valor Item | Valor total do item. |
| Código Vencedor | CNPJ do licitante vencedor. |
| Nome Vencedor | Nome (razão social) do CNPJ vencedor. |

---

## Licitação

| Coluna | Descrição |
| --- | --- |
| Número Licitação | Número que identifica a licitação no SIASG. |
| Código UG | Código da Unidade Gestora (UG) responsável pela licitação. Ver [Definições](#definições). |
| Nome UG | Nome da Unidade Gestora. |
| Código Modalidade Compra | Código da modalidade de compra. Ver [tabela de modalidades](#códigos-de-modalidade-de-compra). |
| Modalidade Compra | Modalidade da compra: Concorrência; Concurso; Convite; Dispensa de Licitação; Inexigibilidade de Licitação; Pregão; Registro de Preço; Tomada de Preços. |
| Número Processo | Número do processo da licitação. |
| Objeto | Objeto da licitação, ou seja, aquilo que se quer comprar, alienar ou contratar. |
| Situação Licitação | Situação em que se encontra o processo licitatório. |
| Código Órgão Superior | Código do órgão superior responsável pela licitação. Ver [Definições](#definições). |
| Nome Órgão Superior | Nome do órgão superior. |
| Código Órgão | Código do órgão (subordinado) responsável pela licitação. Ver [Definições](#definições). |
| Nome Órgão | Nome do órgão. |
| UF | Unidade Federativa (estado) onde ocorre a licitação. |
| Município | Município onde ocorre a licitação. |
| Data Resultado Compra | Data da publicação da homologação no Diário Oficial da União. |
| Data Abertura | Data de abertura para envio das propostas. |
| Valor Licitação | Valor total licitado. |

---

## ParticipantesLicitação

| Coluna | Descrição |
| --- | --- |
| Número Licitação | Número que identifica a licitação no SIASG. |
| Código UG | Código da Unidade Gestora (UG) responsável pela licitação. Ver [Definições](#definições). |
| Nome UG | Nome da Unidade Gestora. |
| Código Modalidade Compra | Código da modalidade de compra. Ver [tabela de modalidades](#códigos-de-modalidade-de-compra). |
| Modalidade Compra | Modalidade da compra: Concorrência; Concurso; Convite; Dispensa de Licitação; Inexigibilidade de Licitação; Pregão; Registro de Preço; Tomada de Preços. |
| Número Processo | Número do processo da licitação. |
| Código Órgão | Código do órgão (subordinado) responsável pela licitação. Ver [Definições](#definições). |
| Nome Órgão | Nome do órgão. |
| Código Item Compra | Código do item da compra no SIASG. |
| Descrição Item Compra | Descrição do item da compra no SIASG. |
| Código Participante | CNPJ do participante na licitação. |
| Nome Participante | Nome (razão social) do CNPJ participante. |
| Flag Vencedor | Indica se o participante é vencedor: "SIM" ou "NÃO". |

---

## Definições

Termos usados nas descrições acima (fonte: Manual do SIAFI):

- **Unidade Gestora (UG):** unidade orçamentária ou administrativa que realiza atos de gestão orçamentária, financeira e/ou patrimonial, cujo titular está sujeito a tomada de contas anual, conforme os artigos 81 e 82 do Decreto-lei nº 200, de 25 de fevereiro de 1967.
- **Órgão Subordinado:** entidade supervisionada por um órgão da Administração Direta.
- **Órgão Superior:** unidade da Administração Direta que tem entidades por ela supervisionadas.

### Composição do Código Item Compra (22 dígitos)

O código do item é formado por:

6 dígitos da Unidade Gestora + 2 dígitos da modalidade de compra + 5 dígitos do número da licitação no ano + 4 dígitos do ano da licitação + 5 dígitos do sequencial do item dentro da licitação.

### Códigos de modalidade de compra

| Código | Modalidade |
| --- | --- |
| 01 | Convite |
| 02 | Tomada de Preços |
| 03 | Concorrência |
| 04 | Concorrência Internacional |
| 05 | Pregão |
| 06 | Dispensa de Licitação |
| 07 | Inexigibilidade de Licitação |
| 20 | Concurso |
| 22 | Tomada de Preços por Técnica e Preço |
| 33 | Concorrência por Técnica e Preço |
| 44 | Concorrência Internacional por Técnica e Preço |
| -99 | Pregão — Registro de Preços |

---

## Diferenças entre o site e os arquivos

Conferi as quatro tabelas contra o cabeçalho real dos CSVs (arquivo de 2013-11). Pontos onde o dicionário oficial do site **não** bate com os arquivos:

- O site escreve **"Modalidade compra"** e **"Código Modalidade de Compra"**; nos CSVs as colunas são **`Modalidade Compra`** e **`Código Modalidade Compra`** (sem o "de").
- Na tabela **Licitação**, o site lista **"UF/Município"** como um único campo, mas os CSVs trazem **`UF`** e **`Município`** em **duas colunas separadas**.
- Na tabela **ParticipantesLicitação**, o site chama o campo de **"CNPJ Participante"**; no CSV a coluna se chama **`Código Participante`** (e contém o CNPJ).
- O site escreve **"Número do Processo"**; nos CSVs é **`Número Processo`**.

Ao programar, use sempre os nomes exatos das colunas como estão nos arquivos (coluna **Coluna** das tabelas acima).
