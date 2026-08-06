import io                # para "engolir" os prints do treinamento
import sys               # só pra falar pro Python onde achar nosso código
import shutil            # apaga a pasta de rascunho no final
import tempfile          # cria uma pasta "de rascunho" temporária
import unittest          # o framework de teste
import contextlib        # usado junto com o io, para silenciar os prints
from pathlib import Path # forma moderna de lidar com "caminho de arquivo"

import numpy as np       # o treinamento recebe arrays NumPy
import torch             # ...e devolve um modelo PyTorch

# ATENÇÃO: aqui apontamos para a RAIZ do projeto, e não para src/ como nos
# outros testes. Isso porque o train.py importa com o prefixo "src."
# (ex: "from src.utils.config import Config"), então o Python precisa
# enxergar a pasta que CONTÉM src/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.train import (
    TrainHistory,
    _avg_loss,
    _make_loader,
    compute_anomaly_scores,
    save_artifacts,
    train_autoencoder,
)
from src.models.autoencoder import Autoencoder
from src.utils.config import Config


def dados_de_brincadeira(n_linhas: int = 40, n_features: int = 3) -> np.ndarray:
    """
    Gera uma matriz de features falsa, mas SEMPRE igual (semente fixa),
    para os testes não darem resultado diferente a cada execução.
    """
    rng = np.random.default_rng(42)
    return rng.normal(size=(n_linhas, n_features)).astype(np.float32)


def config_de_teste(pasta: Path) -> Config:
    """
    Um Config pequeno: poucas épocas, para o teste rodar em segundos,
    e artifacts_dir apontando para a pasta de rascunho do teste.
    """
    return Config(artifacts_dir=pasta, epochs=3, batch_size=8)


# ==========================================================================
# TESTE 1: _make_loader
# Transforma o array NumPy em DataLoader do PyTorch — é o passo que prepara
# os dados para o laço de treinamento, em lotes (batches).
# ==========================================================================
class TestMakeLoader(unittest.TestCase):

    def test_divide_os_dados_em_lotes_do_tamanho_pedido(self):
        X = dados_de_brincadeira(n_linhas=40, n_features=3)

        loader = _make_loader(X, batch_size=8, shuffle=False)
        primeiro_lote, _alvo = next(iter(loader))

        # 8 licitações por lote, cada uma com 3 features
        self.assertEqual(primeiro_lote.shape, (8, 3))

    def test_lote_vira_tensor_float32(self):
        # dtype errado quebra o cálculo da perda mais na frente
        X = dados_de_brincadeira()

        loader = _make_loader(X, batch_size=8, shuffle=False)
        primeiro_lote, _alvo = next(iter(loader))

        self.assertEqual(primeiro_lote.dtype, torch.float32)

    def test_entrada_e_alvo_sao_iguais(self):
        # Num autoencoder o "gabarito" é a própria entrada:
        # o modelo tenta reconstruir o que recebeu.
        X = dados_de_brincadeira()

        loader = _make_loader(X, batch_size=8, shuffle=False)
        entrada, alvo = next(iter(loader))

        self.assertTrue(torch.allclose(entrada, alvo))


# ==========================================================================
# TESTE 2: _avg_loss
# Calcula a perda média do modelo sobre um conjunto de dados.
# É a função que produz os números impressos a cada época.
# ==========================================================================
class TestAvgLoss(unittest.TestCase):

    def test_devolve_um_numero_nao_negativo(self):
        X = dados_de_brincadeira()
        modelo = Autoencoder(input_dim=X.shape[1])

        perda = _avg_loss(modelo, X, torch.nn.MSELoss())

        # MSE é uma média de quadrados: nunca pode ser negativa
        self.assertIsInstance(perda, float)
        self.assertGreaterEqual(perda, 0.0)


# ==========================================================================
# TESTE 3: train_autoencoder
# O coração da etapa: o laço de treinamento propriamente dito.
# Os prints das épocas são silenciados para não poluir a saída dos testes.
# ==========================================================================
class TestTrainAutoencoder(unittest.TestCase):

    def setUp(self):
        self.pasta_temporaria = Path(tempfile.mkdtemp())
        self.config = config_de_teste(self.pasta_temporaria)
        self.X_train = dados_de_brincadeira(n_linhas=40, n_features=3)
        self.X_test = dados_de_brincadeira(n_linhas=16, n_features=3)

    def tearDown(self):
        shutil.rmtree(self.pasta_temporaria)

    def _treinar(self, config=None):
        # redirect_stdout joga os prints do treino num "papel de rascunho"
        # em memória, em vez da tela.
        with contextlib.redirect_stdout(io.StringIO()):
            return train_autoencoder(self.X_train, self.X_test, config or self.config)

    def test_devolve_o_modelo_e_o_historico(self):
        modelo, historico = self._treinar()

        self.assertIsInstance(modelo, Autoencoder)
        self.assertIsInstance(historico, TrainHistory)

    def test_registra_uma_perda_por_epoca(self):
        # config_de_teste usa 3 épocas, então esperamos 3 números
        # em cada lista do histórico.
        _modelo, historico = self._treinar()

        self.assertEqual(len(historico.train_loss), 3)
        self.assertEqual(len(historico.test_loss), 3)

    def test_modelo_treinado_reconstroi_com_o_formato_certo(self):
        modelo, _historico = self._treinar()

        entrada = torch.tensor(self.X_test, dtype=torch.float32)
        with torch.no_grad():
            saida = modelo(entrada)

        self.assertEqual(saida.shape, entrada.shape)

    def test_o_erro_de_treino_diminui(self):
        # O teste que prova que o treinamento realmente aprende algo:
        # com mais épocas, a perda no fim tem que ser menor que no começo.
        config = config_de_teste(self.pasta_temporaria)
        config.epochs = 30

        _modelo, historico = self._treinar(config)

        self.assertLess(historico.train_loss[-1], historico.train_loss[0])

    def test_treino_e_reprodutivel(self):
        # Mesma semente (config.random_seed) => mesmo resultado.
        # É o requisito RNF02 (reprodutibilidade) na prática.
        _m1, historico_1 = self._treinar()
        _m2, historico_2 = self._treinar()

        self.assertEqual(historico_1.train_loss, historico_2.train_loss)


# ==========================================================================
# TESTE 4: save_artifacts
# Salvamento do modelo treinado + o scaler (mean/std) + os nomes das colunas.
# Sem esses três, não dá para fazer inferência depois.
# ==========================================================================
class TestSaveArtifacts(unittest.TestCase):

    def setUp(self):
        self.pasta_temporaria = Path(tempfile.mkdtemp())
        self.config = config_de_teste(self.pasta_temporaria)
        self.modelo = Autoencoder(
            input_dim=3,
            hidden1=self.config.hidden1,
            hidden2=self.config.hidden2,
            bottleneck=self.config.bottleneck,
        )
        self.mean = np.zeros(3, dtype=np.float32)
        self.std = np.ones(3, dtype=np.float32)
        self.feature_names = ["dias", "valor", "participantes"]

    def tearDown(self):
        shutil.rmtree(self.pasta_temporaria)

    def _salvar(self):
        with contextlib.redirect_stdout(io.StringIO()):
            save_artifacts(
                self.modelo, self.mean, self.std, self.feature_names, self.config
            )

    def test_cria_os_tres_arquivos(self):
        self._salvar()

        self.assertTrue(self.config.model_path.exists())
        self.assertTrue(self.config.scaler_path.exists())
        self.assertTrue(self.config.feature_names_path.exists())

    def test_modelo_salvo_pode_ser_recarregado(self):
        # Não basta o arquivo existir: o modelo recarregado precisa
        # devolver exatamente a mesma reconstrução do original.
        entrada = torch.ones(5, 3)

        self.modelo.eval()
        with torch.no_grad():
            saida_original = self.modelo(entrada)

        self._salvar()

        modelo_carregado = Autoencoder(
            input_dim=3,
            hidden1=self.config.hidden1,
            hidden2=self.config.hidden2,
            bottleneck=self.config.bottleneck,
        )
        modelo_carregado.load_state_dict(
            torch.load(self.config.model_path, weights_only=True)
        )
        modelo_carregado.eval()
        with torch.no_grad():
            saida_carregada = modelo_carregado(entrada)

        self.assertTrue(torch.allclose(saida_original, saida_carregada))


# ==========================================================================
# TESTE 5: compute_anomaly_scores
# O produto final do sistema: uma nota de anomalia por licitação.
# ==========================================================================
class TestComputeAnomalyScores(unittest.TestCase):

    def test_um_score_para_cada_licitacao(self):
        X = dados_de_brincadeira(n_linhas=25, n_features=3)
        modelo = Autoencoder(input_dim=3)

        scores = compute_anomaly_scores(modelo, X)

        self.assertIsInstance(scores, np.ndarray)
        self.assertEqual(scores.shape, (25,))

    def test_scores_nunca_sao_negativos(self):
        # O score é um erro quadrático médio, então é sempre >= 0
        X = dados_de_brincadeira(n_linhas=25, n_features=3)
        modelo = Autoencoder(input_dim=3)

        scores = compute_anomaly_scores(modelo, X)

        self.assertTrue((scores >= 0).all())


if __name__ == "__main__":
    unittest.main()
