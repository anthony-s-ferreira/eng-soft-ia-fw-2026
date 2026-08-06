import sys              # só pra falar pro Python onde achar nosso código
import tempfile          # cria uma pasta "de rascunho" temporária
import shutil            # apaga essa pasta de rascunho no final
import unittest          # o framework de teste
from pathlib import Path # forma moderna de lidar com "caminho de arquivo"

import torch             # o modelo é feito em PyTorch, então os testes
                         # também precisam criar tensores

# Diz ao Python onde achar a pasta src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.model import AnomalyAutoEncoder


# ==========================================================================
# TESTE 1: o formato (shape) da saída do modelo.
# O AnomalyAutoEncoder é um autoencoder: ele comprime a entrada até 4 números
# (o "gargalo") e depois tenta reconstruir a entrada original. Ou seja,
# a saída tem que ter EXATAMENTE o mesmo formato da entrada.
# ==========================================================================
class TestFormatoDaSaida(unittest.TestCase):

    def test_saida_tem_o_mesmo_formato_da_entrada(self):
        # 5 licitações fictícias, cada uma com 3 características (features)
        entrada = torch.zeros(5, 3)

        modelo = AnomalyAutoEncoder(input_dim=3)
        saida = modelo(entrada)

        # Reconstruiu 5 linhas de 3 colunas, igual à entrada
        self.assertEqual(saida.shape, entrada.shape)

    def test_funciona_com_uma_unica_feature(self):
        # O pipeline atual do projeto gera só 1 feature numérica
        # (diferença de dias entre abertura e resultado), então esse
        # caso precisa funcionar também.
        entrada = torch.zeros(10, 1)

        modelo = AnomalyAutoEncoder(input_dim=1)
        saida = modelo(entrada)

        self.assertEqual(saida.shape, (10, 1))

    def test_saida_e_float32(self):
        # dtype errado quebra o cálculo da perda mais na frente,
        # então vale travar isso num teste.
        entrada = torch.zeros(4, 3)

        modelo = AnomalyAutoEncoder(input_dim=3)
        saida = modelo(entrada)

        self.assertEqual(saida.dtype, torch.float32)


# ==========================================================================
# TESTE 2: o "gargalo" do autoencoder.
# A parte encoder tem que comprimir qualquer entrada para 4 números.
# É isso que força o modelo a aprender só o padrão "normal" das licitações.
# ==========================================================================
class TestGargalo(unittest.TestCase):

    def test_encoder_comprime_para_4_dimensoes(self):
        entrada = torch.zeros(7, 20)

        modelo = AnomalyAutoEncoder(input_dim=20)
        comprimido = modelo.encoder(entrada)

        # 7 linhas continuam 7 linhas, mas 20 colunas viraram 4
        self.assertEqual(comprimido.shape, (7, 4))


# ==========================================================================
# TESTE 3: salvar e carregar o modelo.
# Igual ao test_data.py, usamos uma pasta de rascunho temporária para não
# deixar arquivo .pth espalhado no computador.
# ==========================================================================
class TestSalvarECarregarModelo(unittest.TestCase):

    # setUp roda ANTES de cada teste, criando a pasta de rascunho vazia
    def setUp(self):
        self.pasta_temporaria = Path(tempfile.mkdtemp())
        self.caminho_do_modelo = self.pasta_temporaria / "anomaly_model.pth"

    # tearDown roda DEPOIS de cada teste, apagando a pasta de rascunho
    def tearDown(self):
        shutil.rmtree(self.pasta_temporaria)

    def test_arquivo_do_modelo_e_criado(self):
        modelo = AnomalyAutoEncoder(input_dim=3)

        torch.save(modelo.state_dict(), self.caminho_do_modelo)

        self.assertTrue(self.caminho_do_modelo.exists())

    def test_modelo_carregado_devolve_a_mesma_saida(self):
        # Esse é o teste que realmente importa: não basta o arquivo existir,
        # o modelo recarregado precisa se comportar igual ao original.
        entrada = torch.ones(5, 3)

        modelo_original = AnomalyAutoEncoder(input_dim=3)
        modelo_original.eval()  # desliga comportamentos de treino
        with torch.no_grad():
            saida_original = modelo_original(entrada)

        # Salva os pesos aprendidos
        torch.save(modelo_original.state_dict(), self.caminho_do_modelo)

        # Cria um modelo NOVO (pesos aleatórios) e carrega os pesos salvos
        modelo_carregado = AnomalyAutoEncoder(input_dim=3)
        modelo_carregado.load_state_dict(
            torch.load(self.caminho_do_modelo, weights_only=True)
        )
        modelo_carregado.eval()
        with torch.no_grad():
            saida_carregada = modelo_carregado(entrada)

        # allclose compara tensores permitindo diferença numérica mínima
        self.assertTrue(torch.allclose(saida_original, saida_carregada))


if __name__ == "__main__":
    unittest.main()
