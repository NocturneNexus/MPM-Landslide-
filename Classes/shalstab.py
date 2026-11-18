from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from .Base import ParametrosSHALSTAB


class SHALSTAB:

    """

    A classe SHALSTAB tem como objetivo classificar a estabilidade da encosta
    a partir do fator de segurança e da saturação do solo. Primeiro, ela recebe
    o grafo da malha, a hidrologia e os parâmetros do solo, e organiza os dados
    de área específica, inclinação, saturação e espessura. Depois, o método
    _calcular compara a resistência ao cisalhamento com a força de mobilização
    gravitacional em cada ponto da malha e marca as regiões com fator de
    segurança menor que 1. Por fim, o método plotar gera um histograma com a
    distribuição do fator de segurança. Dessa forma, conseguimos identificar
    as regiões instáveis que servem como base para a mobilização do MPM.

    """

    '''

    Inicializa a análise armazenando o grafo, a hidrologia e os parâmetros,
    organiza as variáveis de entrada em arrays e executa o cálculo do
    fator de segurança.

    '''

    def __init__(
        self,

        # < Armazena o grafo com a malha, a área específica e a inclinação >

        grafo,

        # < Armazena as geometrias da encosta (opcional) >

        geometrias=None,

        # < Armazena a hidrologia com a espessura do solo (opcional) >

        hidrologia=None,

        # < Define os parâmetros do solo, usando os padrões se não informado >

        parametros: Optional[ParametrosSHALSTAB] = None,
    ):

        # < Armazena as referências recebidas na inicialização >

        self.grafo = grafo
        self.geometrias = geometrias
        self.hidrologia = hidrologia

        # < Usa os parâmetros padrão caso nenhum seja informado >

        self.parametros = parametros or ParametrosSHALSTAB()

        # < Converte a área específica e a inclinação para arrays numéricos >

        self.area_especifica = np.asarray(grafo.area_especifica, dtype=float)
        self.theta = np.asarray(grafo.theta, dtype=float)

        # < Obtém a saturação do grafo, assumindo solo seco se não existir >

        self.saturacao = np.asarray(
            getattr(grafo, "w", None)
            if getattr(grafo, "w", None) is not None
            else np.zeros_like(self.area_especifica),
            dtype=float,
        )

        # < Obtém a espessura do solo >

        self.espessura_solo = self._obter_espessura()

        # < Inicializa os resultados que serão preenchidos pelo cálculo >

        self.fator_seguranca = None
        self.regioes_instaveis = None

        # < Executa o cálculo do fator de segurança >

        self._calcular()

    # < Obtém a espessura do solo da hidrologia, da malha ou assume 1 metro >

    def _obter_espessura(self) -> np.ndarray:

        # < Prioridade 1: a espessura já lida pela hidrologia >

        if self.hidrologia is not None and hasattr(self.hidrologia, "espessura_solo"):
            return np.asarray(self.hidrologia.espessura_solo, dtype=float)

        # < Prioridade 2: a profundidade do solo gravada na malha >

        mesh = getattr(self.grafo, "mesh", None)
        if mesh is not None and "ProfundidadeSolo" in mesh.point_data:
            return np.asarray(mesh.point_data["ProfundidadeSolo"], dtype=float)

        # < Último caso: espessura de 1 metro em todos os pontos >

        return np.ones_like(self.area_especifica, dtype=float)

    # < Calcula o fator de segurança e identifica as regiões instáveis >

    def _calcular(self) -> np.ndarray:

        # < Atalho para os parâmetros do solo >

        p = self.parametros

        # < Calcula o seno e o cosseno da inclinação do terreno >

        theta = np.asarray(self.theta, dtype=float)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # < Obtém a espessura do solo >

        espessura = np.asarray(self.espessura_solo, dtype=float)

        # < Limita a saturação entre 0 (solo seco) e 1 (solo saturado) >

        saturacao = np.clip(np.asarray(self.saturacao, dtype=float), 0.0, 1.0)

        '''

        Normaliza a área específica entre 0 e 1. Se todos os valores forem
        iguais, a normalização fica zerada para evitar divisão por zero

        '''

        area = np.asarray(self.area_especifica, dtype=float)
        area_min = area.min()
        area_max = area.max()
        area_norm = np.zeros_like(area, dtype=float)
        denom = area_max - area_min
        if denom > 0:
            area_norm = (area - area_min) / denom

        '''

        Calcula o peso específico do solo e da água e, a partir da saturação,
        o peso específico efetivo do solo

        '''

        gamma_solo = p.densidade_solo * p.gravidade
        gamma_agua = p.densidade_agua * p.gravidade
        gamma_efetivo = gamma_solo - gamma_agua * saturacao

        '''

        Calcula a resistência ao cisalhamento, somando a coesão com o atrito
        gerado pelo peso efetivo do solo

        '''

        resistencia = (
            p.coesao
            + gamma_efetivo * espessura * cos_theta**2 * np.tan(p.angulo_atrito)
        )

        '''

        Calcula a tensão de mobilização causada pela gravidade, ampliada
        pela contribuição da área específica

        '''

        mobilizacao = (
            gamma_solo
            * espessura
            * sin_theta
            * cos_theta
            * (1.0 + p.alfa_area * area_norm)
        )

        '''

        Calcula o fator de segurança dividindo a resistência pela mobilização.
        Onde não há mobilização (terreno plano), o ponto é muito estável e
        recebe o fator de segurança máximo

        '''

        fs = np.divide(
            resistencia,
            mobilizacao,
            out=np.full_like(resistencia, p.fs_maximo, dtype=float),
            where=mobilizacao > 0,
        )

        # < Mantém o fator de segurança entre um mínimo positivo e o máximo definido >

        fs = np.clip(fs, 1e-6, p.fs_maximo)

        # < Armazena o fator de segurança e marca as regiões com FS menor que 1 >

        self.fator_seguranca = fs
        self.regioes_instaveis = fs < 1.0

        # < Salva os resultados na malha para visualização >

        mesh = getattr(self.grafo, "mesh", None)
        if mesh is not None:
            mesh.point_data["FatorSeguranca"] = fs
            mesh.point_data["RegiaoInstavel"] = self.regioes_instaveis.astype(int)

        # < Salva os resultados no grafo para uso na mobilização do MPM >

        self.grafo.fator_seguranca = fs
        self.grafo.regioes_instaveis = self.regioes_instaveis

        # < Retorna o fator de segurança calculado >

        return fs

    # < Gera o histograma da distribuição do fator de segurança >

    def plotar(self) -> None:

        # < Calcula o fator de segurança caso ainda não tenha sido calculado >

        if self.fator_seguranca is None:
            self._calcular()

        # < Cria a figura do histograma >

        _, ax = plt.subplots(figsize=(12, 6))

        # < Plota a distribuição do fator de segurança em 30 intervalos >

        ax.hist(self.fator_seguranca, bins=30, edgecolor="black")

        # < Marca a linha de FS = 1, que separa regiões estáveis e instáveis >

        ax.axvline(1.0, color="red", linestyle="--", label="FS = 1")

        # < Define o título e os nomes dos eixos >

        ax.set_title("Distribuição do Fator de Segurança (SHALSTAB)")
        ax.set_xlabel("Fator de Segurança")
        ax.set_ylabel("Frequência")

        # < Exibe a legenda da linha de referência >

        ax.legend()

        # < Ajusta automaticamente os espaçamentos do gráfico >

        plt.tight_layout()

        # < Exibe o gráfico na tela >

        plt.show()
