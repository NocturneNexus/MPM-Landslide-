"""

O arquivo __init__ define a interface pública do pacote Classes. Ele importa
as classes de cada módulo (parâmetros, clima, grafo, hidrologia, SHALSTAB,
MPM, simulação e visualização) e usa '__all__' para restringir quais nomes
são exportados ao utilizar a importação 'from Classes import *'.

"""

from .Base import (
    ParametrosHidrologicos,
    ParametrosSHALSTAB,
    ParticulaMPM,
    PropriedadesTriangulo,
)
from .Clima import ClimaNASA
from .Grafo import GrafoTIN
from .hidrologia import ModeloHidrologico
from .mpm import GeradorParticulasMPM, MassaInstavel, RegiaoInstavel
from .particulas import AnaliseParticulas
from .shalstab import SHALSTAB
from .simulacao import SimulacaoTIN
from .tin import GeometriaTIN
from .visualizacao import Animacao2D, Animacao3D, Visualizador

# < Lista dos nomes exportados pelo pacote >

__all__ = [
    "AnaliseParticulas",
    "Animacao3D",
    "Animacao2D",
    "ClimaNASA",
    "GeometriaTIN",
    "GeradorParticulasMPM",
    "GrafoTIN",
    "MassaInstavel",
    "ModeloHidrologico",
    "ParametrosHidrologicos",
    "ParametrosSHALSTAB",
    "ParticulaMPM",
    "PropriedadesTriangulo",
    "RegiaoInstavel",
    "SHALSTAB",
    "SimulacaoTIN",
    "Visualizador",
]