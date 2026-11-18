from __future__ import annotations

from typing import Optional

import numpy as np

from .Base import ParametrosHidrologicos


class ModeloHidrologico:

    """

    A classe ModeloHidrologico tem como objetivo calcular como a água da
    chuva e a recarga do solo se movimentam e se acumulam na malha. Primeiro,
    ela recebe o grafo e os parâmetros e calcula o estado hidrológico estático
    com a recarga de base. Depois, o método simular_chuva aplica a chuva hora
    a hora e registra a saturação máxima de cada ponto ao longo do período.
    Por fim, essa saturação crítica fica no grafo, determinando o nível de
    saturação e a pressão da água que alimentam a análise de estabilidade.

    """

    '''

    Inicializa o modelo armazenando o grafo e os parâmetros, lê a espessura
    do solo da malha e calcula o estado hidrológico estático.

    '''

    def __init__(
        self,

        # < Armazena o grafo com a malha, a inclinação e a área específica >

        grafo,

        # < Define os parâmetros hidrológicos, usando os padrões se não informado >

        parametros: Optional[ParametrosHidrologicos] = None,
    ):
        self.grafo = grafo
        self.parametros = parametros or ParametrosHidrologicos()

        # < Lê da malha a espessura do solo em cada ponto >

        self.espessura_solo = np.asarray(
            grafo.mesh.point_data["ProfundidadeSolo"],
            dtype=float,
        )

        # < Guarda a saturação máxima atingida, preenchida na simulação da chuva >

        self.saturacao_critica = None

        # < Calcula o estado hidrológico com a recarga de base >

        self._calcular_hidrologia_estatica()

    # < Calcula o índice de saturação a partir de uma vazão específica >

    def _indice_saturacao(self, vazao: np.ndarray) -> np.ndarray:

        '''

        Aplica a relação do SHALSTAB, em que a saturação é a vazão específica
        dividida pela capacidade de drenagem do solo, que é a transmissividade
        multiplicada pelo seno da inclinação. Em pontos planos, onde o seno é
        zero, qualquer vazão satura o solo

        '''

        g = self.grafo

        capacidade = g.transmissividade * g.sin_theta

        saturacao = np.zeros_like(vazao, dtype=float)
        mascara = capacidade > 0
        saturacao[mascara] = vazao[mascara] / capacidade[mascara]
        saturacao[~mascara & (vazao > 0)] = 1.0

        # < Limita a saturação entre 0 (solo seco) e 1 (solo saturado) >

        return np.clip(saturacao, 0.0, 1.0)

    # < Calcula o estado hidrológico com a recarga de base >

    def _calcular_hidrologia_estatica(self) -> None:
        p = self.parametros
        g = self.grafo

        # < Calcula a vazão específica a partir da recarga e da área específica >

        g.q = p.recarga_base * g.area_especifica

        # < Calcula a transmissividade do solo >

        g.transmissividade = (
            p.condutividade_saturada
            * self.espessura_solo
            * g.cos_theta
        )

        # < Calcula a saturação e atualiza o estado do grafo >

        indice_saturacao = self._indice_saturacao(g.q)

        self._atualizar_estado(indice_saturacao)

        # < Salva os resultados na malha para visualização >

        g.mesh.point_data["VazaoEspecifica"] = g.q
        g.mesh.point_data["Transmissividade"] = g.transmissividade
        g.mesh.point_data["IndiceSaturacao"] = indice_saturacao
        g.mesh.point_data["EspessuraSaturada"] = g.h
        g.mesh.point_data["PressaoNeutra"] = g.u

    # < Atualiza a saturação, a espessura saturada e a pressão neutra no grafo >

    def _atualizar_estado(self, saturacao: np.ndarray) -> None:
        p = self.parametros
        g = self.grafo

        # < Armazena a saturação e calcula a espessura saturada >

        g.w = saturacao
        g.h = saturacao * self.espessura_solo

        # < Calcula a pressão neutra (poropressão) na base do solo >

        g.u = p.peso_especifico_agua * g.h * g.cos_theta**2

    # < Simula a chuva hora a hora e registra a saturação crítica >

    def simular_chuva(self, chuva_si: np.ndarray) -> dict[str, np.ndarray]:

        # < Troca valores inválidos por zero para não contaminar o cálculo >

        chuva_si = np.nan_to_num(np.asarray(chuva_si, dtype=float), nan=0.0)
        quantidade_passos = len(chuva_si)

        # < Cria os arrays com a saturação média, máxima e mínima de cada passo >

        media = np.zeros(quantidade_passos)
        maxima = np.zeros(quantidade_passos)
        minima = np.zeros(quantidade_passos)

        # < Cria o array que guarda a maior saturação de cada ponto no período >

        saturacao_critica = np.zeros_like(self.grafo.area_especifica, dtype=float)

        for passo, chuva in enumerate(chuva_si):

            # < Calcula a vazão específica gerada pela chuva do passo >

            vazao = chuva * self.grafo.area_especifica

            # < Calcula a saturação de cada ponto nesse passo >

            saturacao = self._indice_saturacao(vazao)

            # < Registra o resumo do passo e atualiza a maior saturação de cada ponto >

            media[passo] = saturacao.mean()
            maxima[passo] = saturacao.max()
            minima[passo] = saturacao.min()
            saturacao_critica = np.maximum(saturacao_critica, saturacao)

        '''

        Deixa no grafo o estado crítico, isto é, a maior saturação de cada
        ponto durante o período. Assim o SHALSTAB analisa a pior condição
        da chuva e não apenas a saturação da última hora

        '''

        if quantidade_passos > 0:
            self.saturacao_critica = saturacao_critica
            self._atualizar_estado(saturacao_critica)

            g = self.grafo
            g.mesh.point_data["IndiceSaturacao"] = g.w
            g.mesh.point_data["EspessuraSaturada"] = g.h
            g.mesh.point_data["PressaoNeutra"] = g.u

        return {"media": media, "maxima": maxima, "minima": minima}