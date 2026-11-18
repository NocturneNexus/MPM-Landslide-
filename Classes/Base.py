from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PropriedadesTriangulo:

    """

    A classe PropriedadesTriangulo tem como objetivo representar o estado
    geométrico e hidráulico de um triângulo da malha TIN. Ela armazena os
    atributos necessários para calcular a declividade, a direção de fluxo,
    a área e a topologia de vizinhança de cada elemento.

    """

    # < Identificador do triângulo dentro da malha >

    id: int

    # < Índices dos três vértices que formam o triângulo >

    vertices: tuple[int, int, int]

    # < Área do triângulo em metros quadrados >

    area: float

    # < Coordenadas (x, y, z) do centroide do triângulo >

    centroide: np.ndarray

    # < Vetor normal unitário da face, sempre apontando para cima >

    normal: np.ndarray

    # < Declividade do triângulo em radianos >

    declividade: float

    # < Direção horizontal de maior declive, no sentido da descida >

    direcao_fluxo: np.ndarray

    # < Comprimento das três arestas do triângulo >

    arestas: np.ndarray


@dataclass
class ParametrosHidrologicos:

    """

    A classe ParametrosHidrologicos agrupa os parâmetros físicos do modelo
    hidrológico aplicados à malha TIN. Ela controla variáveis como recarga,
    condutividade hidráulica, porosidade e peso específico da água, que são
    essenciais para a estimativa da saturação e da pressão neutra usadas na
    análise de mobilização.

    """

    # < Recarga de base do solo em metros por segundo >

    recarga_base: float = 5e-8

    # < Condutividade hidráulica saturada em metros por segundo >

    condutividade_saturada: float = 1e-5

    # < Porosidade do solo (reservada, ainda não utilizada no cálculo) >

    porosidade: float = 0.45

    # < Parâmetro do modelo (reservado, ainda não utilizado no cálculo) >

    lambda_modelo: float = 1.0

    # < Aceleração da gravidade em metros por segundo ao quadrado >

    gravidade: float = 9.81

    # < Peso específico da água em newtons por metro cúbico >

    peso_especifico_agua: float = 9810.0


@dataclass
class ParametrosSHALSTAB:

    """

    A classe ParametrosSHALSTAB agrupa os parâmetros do modelo calibrados
    para solos saprolíticos e residuais de rochas gnáissicas e migmatíticas
    na região de Juiz de Fora. Os valores de coesão e ângulo de atrito foram
    ajustados para refletir com maior realismo a resposta crítica à saturação
    sob a formulação padrão do modelo.

    """

    # < Coesão efetiva do solo em pascal >

    coesao: float = 5.0e3

    # < Ângulo de atrito interno do solo em radianos >

    angulo_atrito: float = np.radians(25.0)

    # < Densidade do solo em quilogramas por metro cúbico >

    densidade_solo: float = 1700.0

    # < Densidade da água em quilogramas por metro cúbico >

    densidade_agua: float = 1000.0

    # < Aceleração da gravidade em metros por segundo ao quadrado >

    gravidade: float = 9.81

    # < Peso da área específica na mobilização (valor pequeno, efeito mínimo) >

    alfa_area: float = 1.0e-4

    # < Fator de segurança máximo, usado também em terrenos planos >

    fs_maximo: float = 10.0


@dataclass
class ParticulaMPM:

    """

    A classe ParticulaMPM representa uma partícula discreta do sistema MPM.
    Ela armazena as propriedades cinemáticas e dinâmicas, como posição,
    velocidade, aceleração e massa, além dos dados auxiliares de mobilização,
    saturação e estado de atividade.

    """

    # < Identificador da partícula >

    id: int

    # < Posição (x, y, z) da partícula em metros >

    posicao: np.ndarray

    # < Massa da partícula em quilogramas >

    massa: float

    # < Volume da partícula em metros cúbicos >

    volume: float

    # < Densidade do material da partícula >

    densidade: float

    # < Velocidade (x, y, z) da partícula em metros por segundo >

    velocidade: np.ndarray

    # < Aceleração (x, y, z) da partícula em metros por segundo ao quadrado >

    aceleracao: np.ndarray

    # < Dados auxiliares: fator de segurança, saturação e estado de atividade >

    propriedades: dict[str, float]