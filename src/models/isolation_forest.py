import numpy as np
import torch
import math
import pickle

# --- 1. FUNÇÕES AUXILIARES DE ANOMALIA ---

def c_factor(n: int) -> float:
    """
    Calcula o comprimento médio de busca mal-sucedida em uma árvore de busca binária (BST).
    Usado como fator de normalização no cálculo do score de anomalia do Isolation Forest.

    Parameters:
        n (int): O número de nós/amostras.

    Returns:
        float: O valor do fator c(n). Retorna 0.0 se n <= 1 e 1.0 se n == 2.
    """
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    # Constante de Euler-Mascheroni aproximadamente 0.5772156649
    return 2.0 * (math.log(n - 1) + 0.5772156649) - (2.0 * (n - 1) / n)


# --- 2. ESTRUTURA DO ISOLATION FOREST ---

class IsolationTreeNode:
    def __init__(self, left=None, right=None, split_feature=None, split_value=None, size=0, is_leaf=False):
        """
        Inicializa o nó da árvore de isolamento.

        Parameters:
            left (IsolationTreeNode, optional): Nó filho à esquerda. Default é None.
            right (IsolationTreeNode, optional): Nó filho à direita. Default é None.
            split_feature (int, optional): Índice da feature utilizada para a divisão neste nó. Default é None.
            split_value (float, optional): Valor de corte (split) aplicado à feature. Default é None.
            size (int, optional): Número de amostras que atingiram este nó. Default é 0.
            is_leaf (bool, optional): Indica se o nó é uma folha. Default é False.
        """
        self.left = left
        self.right = right
        self.split_feature = split_feature
        self.split_value = split_value
        self.size = size
        self.is_leaf = is_leaf


class IsolationTree:
    def __init__(self, max_depth: int):
        """
        Inicializa a estrutura da árvore de isolamento.

        Parameters:
            max_depth (int): A profundidade máxima permitida para a árvore.
        """
        self.max_depth = max_depth
        self.root = None

    def fit(self, X: np.ndarray, current_depth: int = 0) -> IsolationTreeNode:
        """
        Ajusta o modelo aos dados de treinamento construindo a floresta de árvores de isolamento.

        Parameters:
            X (np.ndarray): Conjunto de dados de entrada com formato (n_samples, n_features).

        Returns:
            IsolationForest: Retorna a própria instância treinada.
        """
        n_samples, n_features = X.shape

        if current_depth >= self.max_depth or n_samples <= 1:
            return IsolationTreeNode(size=n_samples, is_leaf=True)

        feature_idx = np.random.randint(0, n_features)
        feat_min, feat_max = X[:, feature_idx].min(), X[:, feature_idx].max()

        if feat_min == feat_max:
            return IsolationTreeNode(size=n_samples, is_leaf=True)

        split_val = np.random.uniform(feat_min, feat_max)
        left_mask = X[:, feature_idx] < split_val

        left_node = self.fit(X[left_mask], current_depth + 1)
        right_node = self.fit(X[~left_mask], current_depth + 1)

        return IsolationTreeNode(
            left=left_node,
            right=right_node,
            split_feature=feature_idx,
            split_value=split_val,
            size=n_samples,
            is_leaf=False
        )


class IsolationForest:
    def __init__(self, n_estimators: int = 100, max_samples: float or int = 256):
        """
        Inicializa o modelo Isolation Forest.

        Parameters:
            n_estimators (int, optional): O número de árvores na floresta. Default é 100.
            max_samples (float ou int, optional): O número de amostras a serem extraídas de X para treinar cada árvore.
                - Se int, extrai `max_samples` amostras.
                - Se float, extrai `max_samples * n_samples` amostras. Default é 256.
        """
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.trees = []
        self.c_subsample = 0.0

    def fit(self, X: np.ndarray):
        """
        Ajusta o modelo aos dados de treinamento construindo a floresta de árvores de isolamento.

        Parameters:
            X (np.ndarray): Conjunto de dados de entrada com formato (n_samples, n_features).

        Returns:
            IsolationForest: Retorna a própria instância treinada.
        """
        n_samples = X.shape[0]
        
        if isinstance(self.max_samples, float):
            subsample_size = int(self.max_samples * n_samples)
        else:
            subsample_size = min(n_samples, self.max_samples)

        self.c_subsample = c_factor(subsample_size)
        max_depth = math.ceil(math.log2(max(subsample_size, 2)))

        self.trees = []
        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, size=subsample_size, replace=False)
            X_sub = X[indices]
            tree = IsolationTree(max_depth=max_depth)
            tree.root = tree.fit(X_sub)
            self.trees.append(tree)

        return self

    def _path_length_torch(self, X_tensor: torch.Tensor, node: IsolationTreeNode, depth: torch.Tensor) -> torch.Tensor:
        """
        Método auxiliar interno. Calcula de forma recursiva e vetorizada (via PyTorch) o comprimento 
        do caminho (path length) de cada amostra em uma árvore específica.

        Parameters:
            X_tensor (torch.Tensor): Tensor de dados de entrada no dispositivo apropriado (CPU/GPU).
            node (IsolationTreeNode): O nó atual durante a travessia na árvore.
            depth (torch.Tensor): Tensor contendo a profundidade atual acumulada para cada amostra.

        Returns:
            torch.Tensor: Tensor contendo os comprimentos dos caminhos acumulados até o nó folha.
        """
        if node.is_leaf:
            return depth + c_factor(node.size)

        mask_left = X_tensor[:, node.split_feature] < node.split_value
        lengths = torch.zeros(X_tensor.shape[0], dtype=torch.float32, device=X_tensor.device)

        if mask_left.any():
            lengths[mask_left] = self._path_length_torch(X_tensor[mask_left], node.left, depth[mask_left] + 1)
        if (~mask_left).any():
            lengths[~mask_left] = self._path_length_torch(X_tensor[~mask_left], node.right, depth[~mask_left] + 1)

        return lengths

    def compute_anomaly_score(self, X: np.ndarray, device: str = "cpu") -> torch.Tensor:
        """
        Calcula o score de anomalia para cada amostra no conjunto de dados X.

        Valores próximos de 1.0 indicam anomalias, enquanto valores significativamente menores que 0.5 
        indicam pontos normais.

        Parameters:
            X (np.ndarray): Array contendo as amostras para avaliação (n_samples, n_features).
            device (str, optional): O dispositivo do PyTorch para realizar os cálculos (ex: "cpu" ou "cuda"). Default é "cpu".

        Returns:
            torch.Tensor: Tensor contendo o score de anomalia calculado para cada amostra.
        """
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        n_samples = X_tensor.shape[0]
        total_paths = torch.zeros(n_samples, dtype=torch.float32, device=device)

        for tree in self.trees:
            depth_init = torch.zeros(n_samples, dtype=torch.float32, device=device)
            total_paths += self._path_length_torch(X_tensor, tree.root, depth_init)

        avg_paths = total_paths / self.n_estimators
        scores = 2.0 ** (- (avg_paths / self.c_subsample))
        return scores