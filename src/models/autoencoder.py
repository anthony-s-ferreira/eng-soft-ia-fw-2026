import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """
    Autoencoder usado para detecção de anomalias em licitações.

    A rede é simétrica: o encoder comprime a entrada até `bottleneck`
    dimensões e o decoder tenta reconstruir a entrada original a partir
    dessa versão comprimida. Como o modelo é treinado com a maioria
    "normal" dos dados, licitações atípicas tendem a ser reconstruídas
    com erro maior — e é esse erro que vira o score de anomalia.

    Os tamanhos das camadas vêm do Config (hidden1, hidden2, bottleneck),
    para que os experimentos da Entrega 7 possam variá-los sem mexer aqui.
    """

    def __init__(
        self,
        input_dim: int,
        hidden1: int = 32,
        hidden2: int = 16,
        bottleneck: int = 8,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, bottleneck),
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed


def reconstruction_error(model: Autoencoder, x: torch.Tensor) -> torch.Tensor:
    """
    Erro de reconstrução (MSE) de cada linha — o score de anomalia.

    Devolve um tensor com um número por licitação, já sem gradiente,
    para poder ser convertido direto com .numpy().
    """
    model.eval()
    with torch.no_grad():
        reconstructed = model(x)
        return torch.mean((x - reconstructed) ** 2, dim=1)
