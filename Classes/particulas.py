from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


class AnaliseParticulas:
    """

    A classe AnaliseParticulas tem como objetivo analisar o comportamento
    das partículas do método MPM. Primeiro, ela recebe os pontos da simulação
    e define uma largura para selecionar uma seção da nuvem de partículas.
    Depois, o método extrair_secao calcula a posição média no eixo Y e seleciona
    os pontos próximos dessa região. Por fim, o método plotar gera dois gráficos:
    um mostrando o perfil horizontal de todas as partículas e outro comparando
    a seção extraída com as duas partes de referência da encosta em perfil
    vertical. Dessa forma, conseguimos visualizar o deslocamento e o
    espalhamento do material.

    """

    '''

    Inicializa a análise armazenando os conjuntos de pontos e definindo
    a largura da seção de extração.

    '''

    def __init__(
            self,

            # < Armazena todas as partículas do método MPM >

            pontos_mpm: np.ndarray,

            # < Armazena os pontos referentes à primeira parte da encosta >

            pontos_parte1: np.ndarray,

            # < Armazena os pontos referentes à segunda parte da encosta >

            pontos_parte2: np.ndarray,

            # < Define a largura da faixa utilizada para extrair a seção >

            largura: float = 20.0,
    ):
        self.pontos_mpm = pontos_mpm
        self.pontos_parte1 = pontos_parte1
        self.pontos_parte2 = pontos_parte2
        self.largura = largura

    # < Extrai uma seção da nuvem de partículas >

    @property
    def extrair_secao(self) -> np.ndarray:
        '''

        Calcula a coordenada Y média da nuvem de partículas,
        utilizada como centro da seção de análise

        '''

        y0 = np.mean(self.pontos_mpm[:, 1])

        '''

        Cria uma máscara lógica que seleciona as partículas
        localizadas a uma distância menor que a largura definida
        em relação a y0

        '''

        mascara = np.abs(self.pontos_mpm[:, 1] - y0) < self.largura

        # < Retorna somente as partículas que passaram pelo filtro >

        return self.pontos_mpm[mascara]

    # < Gera os gráficos para visualizar e comparar as partículas >

    def plotar(self) -> None:
        # < Extrai a seção central da nuvem de partículas >

        secao = self.extrair_secao

        # < Cria uma figura com dois gráficos lado a lado >

        _, ax = plt.subplots(1, 2, figsize=(14, 6))

        # < Primeiro gráfico: mostra todas as partículas no plano X-Y (planta) >

        ax[0].scatter(
            self.pontos_mpm[:, 0],
            self.pontos_mpm[:, 1],
            s=0.2,
        )

        # < Define os nomes dos eixos >

        ax[0].set_xlabel("Coordenada X (metros)")
        ax[0].set_ylabel("Coordenada Y (metros)")

        # < Define o título do primeiro gráfico >

        ax[0].set_title("Perfil Horizontal")

        # < Segundo gráfico: mostra a seção extraída no plano X-Z >

        ax[1].scatter(
            secao[:, 0],
            secao[:, 2],
            s=0.2,
            label="Seção MPM",
        )

        # < Plota os pontos referentes à primeira parte da encosta >

        ax[1].scatter(
            self.pontos_parte1[:, 0],
            self.pontos_parte1[:, 2],
            s=1,
            c="peru",
            label="Parte 1",
        )

        # < Plota os pontos referentes à segunda parte da encosta >

        ax[1].scatter(
            self.pontos_parte2[:, 0],
            self.pontos_parte2[:, 2],
            s=1,
            c="gray",
            label="Parte 2",
        )

        # < Define os nomes dos eixos do segundo gráfico >

        ax[1].set_xlabel("Coordenada X (metros)")
        ax[1].set_ylabel("Coordenada Z (metros)")

        # < Define o título do segundo gráfico >

        ax[1].set_title("Perfil Vertical")

        # < Exibe a legenda para identificar cada conjunto de pontos >

        ax[1].legend()

        # < Ajusta automaticamente os espaçamentos dos gráficos >

        plt.tight_layout()

        # < Exibe os gráficos na tela >

        plt.show()
