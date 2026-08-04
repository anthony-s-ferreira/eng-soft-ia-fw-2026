import sys                # só pra falar pro Python onde achar nosso código
import unittest           # o framework de teste (igual aos slides)
from pathlib import Path  # forma moderna de lidar com "caminho de arquivo"

import numpy as np        # o transform.py trabalha com arrays do NumPy,
                           # então precisamos criar arrays de teste também

# Diz ao Python onde achar a pasta src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from preprocessing.transform import clean_data, to_iso_date, split_data


# ==========================================================================
# TESTE 1: clean_data
# Remove linhas que tenham algum campo vazio (ou só com espaço em branco).
# ==========================================================================
class TestCleanData(unittest.TestCase):

    def test_mantem_linha_valida(self):
        # Monta um array de 2 colunas de texto (uf, objeto), com 1 linha,
        # sem nenhum campo vazio.
        dados = np.array(
            [("PE", "Objeto A")],
            dtype=[("uf", "U2"), ("objeto", "U20")],
        )

        resultado = clean_data(dados)

        # A linha era válida, então continua lá: 1 linha no resultado.
        self.assertEqual(resultado.shape[0], 1)

    def test_remove_linha_com_campo_vazio(self):
        # Agora colocamos 2 linhas: uma válida, outra com "uf" vazio.
        dados = np.array(
            [("PE", "Objeto A"), ("", "Objeto B")],
            dtype=[("uf", "U2"), ("objeto", "U20")],
        )

        resultado = clean_data(dados)

        # Só a linha válida deveria sobrar.
        self.assertEqual(resultado.shape[0], 1)
        self.assertEqual(resultado[0]["uf"], "PE")


# ==========================================================================
# TESTE 2: to_iso_date
# Função simples: converte "dd/mm/yyyy" (formato do CSV) para
# "yyyy-mm-dd" (formato que o NumPy entende como data).
# ==========================================================================
class TestToIsoDate(unittest.TestCase):

    def test_converte_formato_de_data(self):
        resultado = to_iso_date("25/12/2023")
        self.assertEqual(resultado, "2023-12-25")


# ==========================================================================
# TESTE 3: split_data
# Divide os dados em treino (80%) e teste (20%).
# ==========================================================================
class TestSplitData(unittest.TestCase):

    def test_divide_em_80_20(self):
        # Um array simples de 0 a 99 (100 "linhas" fictícias)
        dados = np.arange(100)

        treino, teste = split_data(dados)

        # Esperamos 80 linhas de treino e 20 de teste
        self.assertEqual(treino.shape[0], 80)
        self.assertEqual(teste.shape[0], 20)


if __name__ == "__main__":
    unittest.main()