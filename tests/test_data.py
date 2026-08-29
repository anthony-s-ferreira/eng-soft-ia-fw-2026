import sys              # só pra falar pro Python onde achar nosso código
import tempfile          # cria uma pasta "de rascunho" temporária
import shutil            # apaga essa pasta de rascunho no final
import zipfile           # cria/le arquivos .zip direto no código
import unittest          # o framework de teste
from pathlib import Path # forma moderna de lidar com "caminho de arquivo"

# Diz ao Python: "quando eu escrever 'from data.loader import ...',
# procure dentro da pasta src/".
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.loader import normalize_field_name, load_data


# ==========================================================================
# TESTE 1: uma função simples, sem arquivo nenhum envolvido.
# Igual ao exemplo test_sum(self) do professor: entrada conhecida,
# saída esperada, comparar com assertEqual.
# ==========================================================================
class TestNormalizeFieldName(unittest.TestCase):
    # normalize_field_name pega um nome de coluna "sujo" do CSV
    # (ex: "Data Abertura") e devolve um nome "limpo" (ex: "data_abertura")
    def test_troca_espaco_por_underline(self):
        resultado = normalize_field_name("Data Abertura")
        self.assertEqual(resultado, "data_abertura")

    def test_remove_acento_e_deixa_minusculo(self):
        resultado = normalize_field_name("Número Licitação")
        self.assertEqual(resultado, "numero_licitacao")


# ==========================================================================
# TESTE 2: uma função que precisa de um arquivo .zip de verdade.
# Por isso aparecem tempfile/zipfile/shutil aqui.
# ==========================================================================
class TestLoadData(unittest.TestCase):

    # setUp roda ANTES de cada teste dessa classe, criando uma pasta de rascunho vazia.
    def setUp(self):
        self.pasta_temporaria = Path(tempfile.mkdtemp())

    # tearDown roda DEPOIS de cada teste. Aqui, apaga a pasta de rascunho, para não deixar lixo espalhado no computador
    def tearDown(self):
        shutil.rmtree(self.pasta_temporaria)

    # Monta um ZIP minúsculo, só com uma linha de dado, do jeito que o loader espera encontrar (nome do arquivo terminando em "_Licitação.csv", separado por ";")
    def test_carrega_uma_licitacao_do_zip(self):
        cabecalho = "Número Licitação;UF;Valor Licitação"
        linha = '1;PE;1000,00'
        conteudo_do_csv = cabecalho + "\r\n" + linha

        caminho_do_zip = self.pasta_temporaria / "202401_Licitacoes.zip"
        with zipfile.ZipFile(caminho_do_zip, "w") as z:
            z.writestr("202401_Licitação.csv", conteudo_do_csv.encode("latin1"))

        # Chama a função de verdade que queremos testar
        dados = load_data(str(self.pasta_temporaria), "_Licitação.csv")

        # Confere se o resultado é o que esperávamos
        self.assertEqual(dados.shape[0], 1)          # carregou 1 linha
        self.assertEqual(dados[0]["uf"], "PE")       # e o valor de UF bateu


if __name__ == "__main__":
    unittest.main()