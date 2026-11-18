from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import requests
import pyvista as pv

matplotlib.use("TkAgg")

import numpy as np

from Classes import Animacao2D, SimulacaoTIN
from Classes.Base import ParametrosSHALSTAB

URL_MALHA = "https://drive.google.com/uc?export=download&id=1kgnBGem9NknmpD0bRXc1rl0YlSi1t0KU"
BASE_DIR = Path(__file__).resolve().parent
PASTA_DADOS = BASE_DIR / "Dados"
PASTA_MALHAS = BASE_DIR / "Malhas"
CAMINHO_MALHA_PADRAO = PASTA_MALHAS / "nuvem_pontos-1.vtp"
JUÍZ_DE_FORA = {
    "latitude": -21.75,
    "longitude": -43.35,
}


def parametros_shalstab_juiz_de_fora() -> ParametrosSHALSTAB:
    return ParametrosSHALSTAB(
        coesao=5.0e3,
        angulo_atrito=np.radians(25.0),
        densidade_solo=1700.0,
        densidade_agua=1000.0,
        gravidade=9.81,
        alfa_area=1.0e-4,
    )


def localizar_malha() -> Path:
    candidatos = [
        PASTA_DADOS / "mesh.npz",
        PASTA_DADOS / "malha.vtp",
        PASTA_DADOS / "malha.vtu",
        PASTA_DADOS / "mesh.vtp",
        PASTA_DADOS / "mesh.vtu",
        PASTA_MALHAS / "mesh.npz",
        PASTA_MALHAS / "malha.vtp",
        PASTA_MALHAS / "malha.vtu",
        PASTA_MALHAS / "nuvem_pontos-1.vtp",
    ]

    for caminho in candidatos:
        if caminho.exists():
            return caminho

    if not CAMINHO_MALHA_PADRAO.exists():
        print(
            "Aviso: nenhum dataset interno do projeto foi encontrado em 'Dados/' ou 'Malhas/'. "
            "Será usado o arquivo de fallback baixado automaticamente."
        )
        return baixar_malha()

    return CAMINHO_MALHA_PADRAO


def baixar_malha() -> Path:
    PASTA_MALHAS.mkdir(parents=True, exist_ok=True)
    if not CAMINHO_MALHA_PADRAO.exists():
        resposta = requests.get(URL_MALHA, timeout=60)
        resposta.raise_for_status()
        CAMINHO_MALHA_PADRAO.write_bytes(resposta.content)
    return CAMINHO_MALHA_PADRAO


def triangulizar_malha(malha: pv.PolyData) -> tuple[pv.PolyData, np.ndarray]:
    if malha.n_points < 3:
        raise ValueError("A malha precisa ter pelo menos 3 pontos para gerar um TIN.")

    faces = None
    if malha.n_faces > 0 and malha.faces is not None and len(malha.faces) > 0:
        faces = np.asarray(malha.faces).reshape(-1, 4)[:, 1:]

    if faces is None or len(faces) == 0:
        malha_triangulada = malha.delaunay_2d()
        faces = np.asarray(malha_triangulada.faces).reshape(-1, 4)[:, 1:]

    mesh_tin = pv.PolyData()
    mesh_tin.points = malha.points
    mesh_tin.faces = np.hstack(
        [np.full((len(faces), 1), 3, dtype=int), faces]
    ).ravel()
    mesh_tin.point_data["ProfundidadeSolo"] = np.full(mesh_tin.n_points, 1.0, dtype=float)
    return mesh_tin, faces.astype(int)


def calcular_vizinhos(faces: np.ndarray) -> list[list[int]]:
    mapa_arestas: dict[tuple[int, int], list[int]] = defaultdict(list)
    for indice, triangulo in enumerate(faces):
        arestas = [
            tuple(sorted((triangulo[0], triangulo[1]))),
            tuple(sorted((triangulo[1], triangulo[2]))),
            tuple(sorted((triangulo[2], triangulo[0]))),
        ]
        for aresta in arestas:
            mapa_arestas[aresta].append(indice)

    vizinhos = [[] for _ in range(len(faces))]
    for triangulos in mapa_arestas.values():
        if len(triangulos) != 2:
            continue
        i, j = triangulos
        vizinhos[i].append(j)
        vizinhos[j].append(i)

    return vizinhos


def calcular_theta(malha: pv.PolyData) -> np.ndarray:
    pontos = malha.points
    n_pontos = len(pontos)
    if n_pontos < 3:
        return np.zeros(n_pontos, dtype=float)

    distancias = np.linalg.norm(pontos[:, None, :] - pontos[None, :, :], axis=2)
    np.fill_diagonal(distancias, np.inf)

    theta = np.zeros(n_pontos, dtype=float)
    for indice in range(n_pontos):
        vizinhos = np.argsort(distancias[indice])[:6]
        if len(vizinhos) < 3:
            continue

        x = pontos[vizinhos, 0] - pontos[indice, 0]
        y = pontos[vizinhos, 1] - pontos[indice, 1]
        z = pontos[vizinhos, 2] - pontos[indice, 2]

        matriz = np.column_stack((x, y, np.ones_like(x)))
        coeficientes, *_ = np.linalg.lstsq(matriz, z, rcond=None)
        inclinacao = np.hypot(coeficientes[0], coeficientes[1])
        theta[indice] = np.arctan(inclinacao)

    return theta


def calcular_area_vertices(pontos: np.ndarray, faces: np.ndarray) -> np.ndarray:
    area_vertices = np.zeros(len(pontos), dtype=float)
    contador = np.zeros(len(pontos), dtype=float)

    for triangulo in faces:
        p1, p2, p3 = pontos[triangulo]
        area = 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))
        for indice in triangulo:
            area_vertices[indice] += area / 3.0
            contador[indice] += 1.0

    mascara = contador > 0
    area_vertices[mascara] /= contador[mascara]
    return area_vertices


def calcular_largura_hidraulica(malha: pv.PolyData) -> np.ndarray:
    n_pontos = malha.n_points
    largura = np.ones(n_pontos, dtype=float)
    if n_pontos < 2:
        return largura

    pontos = malha.points
    distancias = np.linalg.norm(pontos[:, None, :] - pontos[None, :, :], axis=2)
    np.fill_diagonal(distancias, np.nan)
    menor = np.nanmin(distancias, axis=1)
    menor = np.nan_to_num(menor, nan=1.0)
    return np.clip(menor, 1e-6, None)


def preparar_simulacao(malha: pv.PolyData) -> SimulacaoTIN:
    mesh_tin, faces = triangulizar_malha(malha)
    vizinhos = calcular_vizinhos(faces)
    theta = calcular_theta(malha)
    area_vertices = calcular_area_vertices(malha.points, faces)
    largura_hidraulica = calcular_largura_hidraulica(malha)

    return SimulacaoTIN(
        mesh=mesh_tin,
        faces=faces,
        vizinhos=vizinhos,
        theta=theta,
        area_vertices=area_vertices,
        largura_hidraulica=largura_hidraulica,
    )


def criar_simulacao() -> SimulacaoTIN:
    caminho_malha = localizar_malha()
    return preparar_simulacao(pv.read(caminho_malha))


def salvar_graficos(simulacao: SimulacaoTIN, pasta_saida: str | Path | None = None) -> Path:
    import matplotlib.pyplot as plt

    pasta = (
        Path(pasta_saida)
        if pasta_saida is not None
        else Path(__file__).resolve().parent / "Visualizador"
    )
    pasta.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 7))
    ax = fig.add_subplot(111)
    ax.triplot(
        simulacao.mesh.points[:, 0],
        simulacao.mesh.points[:, 1],
        simulacao.faces,
        color="lightgray",
        linewidth=0.5,
    )
    ax.quiver(
        simulacao.geometria.centroides[:, 0],
        simulacao.geometria.centroides[:, 1],
        simulacao.geometria.direcoes_fluxo[:, 0],
        simulacao.geometria.direcoes_fluxo[:, 1],
        color="red",
        scale=100,
        width=0.0009,
    )
    ax.set_title("Campo Vetorial de Maior Declive")
    ax.set_xlabel("Coordenada X (metros)")
    ax.set_ylabel("Coordenada Y (metros)")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(pasta / "campo_fluxo.png", dpi=200)
    plt.close(fig)

    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(111)
    ax.plot(
        simulacao.clima.dataframe["Data_Hora"],
        simulacao.resultados_saturacao["media"],
        color="royalblue",
        linewidth=2,
        label="Saturação média",
    )
    ax.set_xlabel("Data")
    ax.set_ylabel("Índice de Saturação")
    ax.set_title("Evolução Temporal do Índice Médio de Saturação")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(pasta / "saturacao.png", dpi=200)
    plt.close(fig)

    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(111)
    ax.bar(
        simulacao.clima.dataframe["Data_Hora"],
        simulacao.clima.chuva,
        width=0.035,
        color="royalblue",
        edgecolor="black",
        linewidth=0.2,
        label="Chuva",
    )
    ax.set_xlabel("Data")
    ax.set_ylabel("Chuva (mm/h)")
    ax.set_title("Intensidade Horária de Chuva")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(pasta / "chuva.png", dpi=200)
    plt.close(fig)

    return pasta


def executar_simulacao(
    latitude: float = JUÍZ_DE_FORA["latitude"],
    longitude: float = JUÍZ_DE_FORA["longitude"],
    inicio: str = "20240101",
    fim: str = "20240131",
    mostrar_graficos: bool = False,
    salvar_graficos_flag: bool = False,
    pasta_saida: str | Path | None = None,
    parametros_shalstab: ParametrosSHALSTAB | None = None,
) -> SimulacaoTIN:
    simulacao = criar_simulacao()
    simulacao.preparar_rede()
    simulacao.carregar_clima(
        latitude=latitude,
        longitude=longitude,
        inicio=inicio,
        fim=fim,
    )
    simulacao.executar_hidrologia_temporal()
    simulacao.calcular_shalstab(
        parametros=parametros_shalstab or parametros_shalstab_juiz_de_fora()
    )
    simulacao.imprimir_resumo()

    if salvar_graficos_flag:
        pasta = salvar_graficos(simulacao, pasta_saida)
        print(f"Gráficos salvos em: {pasta}")

    if mostrar_graficos:
        simulacao.mostrar_resultados()

    return simulacao


def executar_visualizacao_mpm(
    latitude: float = JUÍZ_DE_FORA["latitude"],
    longitude: float = JUÍZ_DE_FORA["longitude"],
    inicio: str = "20240101",
    fim: str = "20240131",
    dt: float = 0.05,
    passos: int = 10,
    save_path: str | Path | None = None,
    parametros_shalstab: ParametrosSHALSTAB | None = None,
    final: bool = True,
):
    simulacao = executar_simulacao(
        latitude=latitude,
        longitude=longitude,
        inicio=inicio,
        fim=fim,
        mostrar_graficos=False,
        salvar_graficos_flag=False,
        parametros_shalstab=parametros_shalstab,
    )
    historico = simulacao.visualizar_mpm(
        dt=dt,
        passos=passos,
        save_path=str(save_path) if save_path is not None else None,
        final=final,
    )
    return simulacao, historico


def executar_visualizacao_3d(
    latitude: float = JUÍZ_DE_FORA["latitude"],
    longitude: float = JUÍZ_DE_FORA["longitude"],
    inicio: str = "20240101",
    fim: str = "20240131",
    dt: float = 0.05,
    passos: int = 5,
    save_path_inicial: str | Path | None = None,
    save_path_mpm: str | Path | None = None,
    parametros_shalstab: ParametrosSHALSTAB | None = None,
):
    simulacao = executar_simulacao(
        latitude=latitude,
        longitude=longitude,
        inicio=inicio,
        fim=fim,
        mostrar_graficos=False,
        salvar_graficos_flag=False,
        parametros_shalstab=parametros_shalstab,
    )

    simulacao.visualizar_malha_3d(save_path=str(save_path_inicial) if save_path_inicial is not None else None)
    historico = simulacao.visualizar_mpm_3d(
        dt=dt,
        passos=passos,
        save_path=str(save_path_mpm) if save_path_mpm is not None else None,
    )
    return simulacao, historico


def executar_animacao_2d(
    latitude: float = JUÍZ_DE_FORA["latitude"],
    longitude: float = JUÍZ_DE_FORA["longitude"],
    inicio: str = "20240101",
    fim: str = "20240131",
    dt: float = 0.05,
    passos: int = 8,
    save_path: str | Path | None = None,
    mostrar: bool = True,
    parametros_shalstab: ParametrosSHALSTAB | None = None,
):
    simulacao = executar_simulacao(
        latitude=latitude,
        longitude=longitude,
        inicio=inicio,
        fim=fim,
        mostrar_graficos=False,
        salvar_graficos_flag=False,
        parametros_shalstab=parametros_shalstab,
    )

    if simulacao.mpm is None:
        simulacao.gerar_particulas_mpm()

    simulacao.mpm.atualizar(dt=dt, passos=passos)

    animacao = Animacao2D(
        simulacao=simulacao,
        fps=10,
        save_path=str(save_path) if save_path is not None else None,
        mostrar=mostrar,
    )
    animacao.executar()
    return simulacao, animacao


def executar_animacao_2d_local(
    latitude: float = JUÍZ_DE_FORA["latitude"],
    longitude: float = JUÍZ_DE_FORA["longitude"],
    inicio: str = "20240101",
    fim: str = "20240131",
    dt: float = 0.05,
    passos: int = 200,
    pasta_saida: str | Path | None = None,
    mostrar_local: bool = True,
    parametros_shalstab: ParametrosSHALSTAB | None = None,
):
    destino = Path(pasta_saida) if pasta_saida is not None else BASE_DIR / "Visualizador"
    destino.mkdir(parents=True, exist_ok=True)
    caminho_gif = destino / "animacao_2d.gif"

    return executar_animacao_2d(
        latitude=latitude,
        longitude=longitude,
        inicio=inicio,
        fim=fim,
        dt=dt,
        passos=passos,
        save_path=caminho_gif,
        mostrar=mostrar_local,
        parametros_shalstab=parametros_shalstab,
    )


if __name__ == "__main__":
    executar_animacao_2d_local(
        mostrar_local=True,
        pasta_saida=BASE_DIR / "Visualizador",
    )
