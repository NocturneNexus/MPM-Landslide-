"""

Visualização do pipeline Land. Este módulo reúne as funções de renderização
da malha TIN, da região instável, das partículas do MPM e dos diagnósticos
hidrológicos. Ele centraliza a conversão de faces da malha para estruturas
compatíveis com PyVista e a geração de animações 2D/3D do histórico do movimento.

"""

from __future__ import annotations

from pathlib import Path

import imageio
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
from matplotlib.animation import FuncAnimation


class Visualizador:

    """

    A classe Visualizador reúne os utilitários compartilhados para conversão
    de malhas e exibição gráfica. Os métodos de gráficos estáticos ficam
    definidos na classe Animacao2D e são ligados a esta classe no final do
    arquivo, mantendo a compatibilidade com o restante do pipeline.

    """

    # < Converte as faces triangulares da malha para o formato aceito pelo PyVista >

    @staticmethod
    def _faces_para_polydata(mesh) -> np.ndarray:
        faces = np.asarray(mesh.faces, dtype=int)

        # < Faces em formato plano viram uma matriz com 4 colunas (3, a, b, c) >

        if faces.ndim == 1:
            faces = faces.reshape(-1, 4)[:, 1:]

        # < Remove a coluna que indica a quantidade de vértices >

        if faces.shape[-1] == 4:
            faces = faces[:, 1:]

        faces = np.asarray(faces, dtype=int)
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"Formato de faces incompatível para visualização 3D: {faces.shape}")

        # < Devolve as faces no formato plano do PyVista, com o 3 antes de cada triângulo >

        return np.hstack([np.full((len(faces), 1), 3, dtype=int), faces]).ravel()


class Animacao3D:

    """

    A classe Animacao3D tem como objetivo renderizar a simulação em 3D para
    inspecionar a malha, a região instável e o MPM. Primeiro, ela coleta o
    histórico de posições das partículas. Depois, o método configurar_figura
    monta a cena com o terreno, a região instável e as partículas. Por fim, o
    método executar percorre os passos do histórico, exibindo a animação em
    tela ou salvando um GIF com os quadros renderizados.

    """

    '''

    Inicializa a animação armazenando a simulação e as opções de exibição
    e coleta o histórico de movimento do MPM.

    '''

    def __init__(
        self,

        # < Armazena a simulação com a malha, a região instável e o MPM >

        simulacao,

        # < Define a quantidade de quadros por segundo >

        fps: int = 10,

        # < Define o caminho do GIF a ser salvo (opcional) >

        save_path: str | Path | None = None,

        # < Define se a animação é exibida em tela >

        mostrar: bool = True,

        # < Define a escala que amplia o deslocamento das partículas no GIF >

        escala_deslocamento: float = 20.0,
    ):
        self.simulacao = simulacao
        self.fps = fps
        self.save_path = Path(save_path) if save_path is not None else None
        self.mostrar = mostrar
        self.escala_deslocamento = escala_deslocamento

        # < Cria a janela 3D, em modo offscreen quando for apenas salvar o GIF >

        self.plotter = pv.Plotter(off_screen=self.save_path is not None or not mostrar)

        # < Inicializa os elementos da cena e coleta o histórico >

        self.terreno = None
        self.region_instavel = None
        self.particulas = None
        self.historico = self._coletar_historico()

    # < Recupera o histórico do MPM ou gera um curto se ainda não existir >

    def _coletar_historico(self) -> list[np.ndarray]:
        if self.simulacao.regiao_instavel is None:
            self.simulacao.selecionar_regiao_instavel()

        # < Usa o histórico já existente, se houver >

        if self.simulacao.mpm is not None and getattr(self.simulacao.mpm, "historico_posicoes", None):
            return [np.asarray(pos, dtype=float) for pos in self.simulacao.mpm.historico_posicoes]

        if self.simulacao.mpm is None:
            self.simulacao.gerar_particulas_mpm()

        # < Se ainda não existe histórico, simula 5 passos >

        if self.simulacao.mpm is not None:
            historico = self.simulacao.mpm.atualizar(dt=0.05, passos=5)
            return [np.asarray(pos, dtype=float) for pos in historico]

        return []

    # < Monta a cena 3D inicial com o terreno e a massa instável >

    def configurar_figura(self) -> None:
        if self.simulacao.mesh is None:
            raise ValueError("A malha do terreno não está disponível para a animação 3D.")

        # < Adiciona o terreno à cena >

        terreno = pv.PolyData(self.simulacao.mesh.points)
        terreno.faces = Visualizador._faces_para_polydata(self.simulacao.mesh)
        self.terreno = self.plotter.add_mesh(
            terreno,
            color="lightgray",
            opacity=0.75,
            show_edges=True,
            edge_color="gray",
            smooth_shading=True,
        )

        # < Destaca em vermelho os pontos da região instável >

        if self.simulacao.regiao_instavel is not None and len(self.simulacao.regiao_instavel.pontos) > 0:
            self.region_instavel = self.plotter.add_points(
                pv.PolyData(np.asarray(self.simulacao.regiao_instavel.pontos, dtype=float)),
                color="red",
                render_points_as_spheres=True,
                point_size=10,
            )

        # < Adiciona as partículas na posição inicial >

        if self.historico:
            pontos_iniciais = pv.PolyData(np.asarray(self.historico[0], dtype=float))
            self.particulas_data = pontos_iniciais
            self.particulas = self.plotter.add_points(
                self.particulas_data,
                color="tomato",
                render_points_as_spheres=True,
                point_size=8,
            )

        # < Configura os eixos, o fundo e a câmera >

        self.plotter.add_axes()
        self.plotter.set_background("white")
        self.plotter.camera_position = "iso"

    # < Atualiza a posição das partículas para o quadro solicitado >

    def atualizar(self, frame: int) -> None:
        if not self.historico:
            return

        posicoes = np.asarray(self.historico[min(frame, len(self.historico) - 1)], dtype=float)
        if posicoes.size == 0:
            return

        self.particulas_data = pv.PolyData(posicoes)
        if self.particulas is not None:
            try:
                self.particulas.GetMapper().SetInputData(self.particulas_data)
                self.particulas.GetMapper().Update()
            except Exception:
                pass

    # < Abre a cena de visualização e monta o cenário base >

    def inicializar(self) -> None:
        self.configurar_figura()

    # < Gera uma imagem do quadro, fora da tela, para exportar o GIF >

    def _renderizar_frame(self, posicoes: np.ndarray) -> np.ndarray:
        plotter = pv.Plotter(off_screen=True)

        # < Adiciona o terreno à cena >

        terreno = pv.PolyData(self.simulacao.mesh.points)
        terreno.faces = Visualizador._faces_para_polydata(self.simulacao.mesh)
        plotter.add_mesh(
            terreno,
            color="lightgray",
            opacity=0.75,
            show_edges=True,
            edge_color="gray",
            smooth_shading=True,
        )

        # < Destaca os pontos da região instável >

        if self.simulacao.regiao_instavel is not None and len(self.simulacao.regiao_instavel.pontos) > 0:
            plotter.add_points(
                pv.PolyData(np.asarray(self.simulacao.regiao_instavel.pontos, dtype=float)),
                color="red",
                render_points_as_spheres=True,
                point_size=10,
            )

        '''

        Amplia o deslocamento das partículas em relação à posição inicial,
        para que o movimento fique visível no GIF

        '''

        posicoes_exibicao = np.asarray(posicoes, dtype=float).copy()
        if len(self.historico) > 1:
            posicao_inicial = np.asarray(self.historico[0], dtype=float)
            deslocamento = posicoes_exibicao - posicao_inicial
            posicoes_exibicao = posicao_inicial + self.escala_deslocamento * deslocamento

        # < Adiciona as partículas na posição ampliada >

        particulas = pv.PolyData(posicoes_exibicao)
        plotter.add_points(
            particulas,
            color="tomato",
            render_points_as_spheres=True,
            point_size=8,
        )

        # < Configura os eixos, o fundo e a câmera >

        plotter.add_axes()
        plotter.set_background("white")
        plotter.camera_position = "iso"

        # < Renderiza, captura a imagem e fecha a janela >

        plotter.render()
        imagem = plotter.screenshot(transparent_background=False, return_img=True)
        plotter.close()
        return imagem

    # < Cria a lista de imagens, uma para cada passo temporal >

    def _capturar_frames(self) -> list[np.ndarray]:
        if not self.historico:
            return []

        frames: list[np.ndarray] = []
        for frame in range(len(self.historico)):
            posicoes = np.asarray(self.historico[frame], dtype=float)
            frames.append(self._renderizar_frame(posicoes))
        return frames

    # < Executa a animação em tela ou salva em arquivo a partir do histórico >

    def executar(self) -> None:
        if not self.historico:
            raise ValueError("Não há histórico de posições do MPM para animar.")

        self.inicializar()

        # < Se há caminho de saída, renderiza os quadros e salva o GIF >

        if self.save_path is not None:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            frames = self._capturar_frames()
            imageio.mimsave(str(self.save_path), frames, fps=self.fps, loop=0)
            self.plotter.close()
            return

        # < Caso contrário, percorre os quadros e exibe a cena >

        if self.mostrar:
            for frame in range(len(self.historico)):
                self.atualizar(frame)
                self.plotter.render()
            self.plotter.show(auto_close=True)


class Animacao2D:

    """

    A classe Animacao2D tem como objetivo animar em 2D a projeção horizontal
    do TIN e o histórico da massa em movimento. Primeiro, ela coleta o
    histórico de posições do MPM. Depois, o método configurar_figura desenha o
    terreno, a região instável e as partículas iniciais. Por fim, o método
    executar anima os passos, exibindo em tela ou salvando em arquivo. A
    classe também reúne os gráficos estáticos de fluxo, rede hidráulica,
    chuva, saturação e MPM.

    """

    '''

    Inicializa a animação armazenando a simulação e as opções de exibição
    e coleta o histórico de movimento do MPM.

    '''

    def __init__(
        self,

        # < Armazena a simulação com a malha, a região instável e o MPM >

        simulacao,

        # < Define a quantidade de quadros por segundo >

        fps: int = 10,

        # < Define o caminho do arquivo a ser salvo (opcional) >

        save_path: str | Path | None = None,

        # < Define se a animação é exibida em tela >

        mostrar: bool = True,
    ):
        self.simulacao = simulacao
        self.fps = fps
        self.save_path = Path(save_path) if save_path is not None else None
        self.mostrar = mostrar
        self.historico = self._coletar_historico()
        self.fig = None
        self.ax = None
        self.particulas = None

    # < Obtém a trajetória do MPM ou simula uma sequência curta se não existir >

    def _coletar_historico(self) -> list[np.ndarray]:
        if self.simulacao.regiao_instavel is None:
            self.simulacao.selecionar_regiao_instavel()
        if self.simulacao.mpm is None:
            self.simulacao.gerar_particulas_mpm()

        historico = getattr(self.simulacao.mpm, "historico_posicoes", [])
        if not historico:
            historico = self.simulacao.mpm.atualizar(dt=0.05, passos=5)

        # < Garante que cada quadro tenha formato (N, 3), mesmo sem partículas >

        return [np.asarray(pos, dtype=float).reshape(-1, 3) for pos in historico]

    # < Desenha o terreno, a zona instável e a posição inicial das partículas >

    def configurar_figura(self) -> None:
        mesh = self.simulacao.mesh
        self.fig, self.ax = plt.subplots(figsize=(12, 8))

        # < Desenha as arestas da malha em cinza >

        self.ax.triplot(
            mesh.points[:, 0],
            mesh.points[:, 1],
            self.simulacao.faces,
            color="lightgray",
            linewidth=0.5,
            alpha=0.8,
        )

        # < Destaca em vermelho os pontos da região instável >

        regiao = self.simulacao.regiao_instavel
        if regiao is not None and len(regiao.pontos):
            self.ax.scatter(
                regiao.pontos[:, 0],
                regiao.pontos[:, 1],
                s=25,
                c="red",
                alpha=0.45,
                label="Região instável",
            )

        # < Desenha as partículas na posição inicial >

        if self.historico and self.historico[0].size:
            posicoes = self.historico[0]
            self.particulas = self.ax.scatter(
                posicoes[:, 0],
                posicoes[:, 1],
                s=22,
                c="tomato",
                edgecolors="black",
                linewidths=0.4,
                label="Partículas MPM",
                zorder=3,
            )

        # < Define o título, os eixos, a grade e os limites do gráfico >

        self.ax.set_title("Animação 2D do escorregamento")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_aspect("equal")
        self.ax.grid(True, alpha=0.3)

        # < Só exibe a legenda se houver elementos com rótulo >

        if self.ax.get_legend_handles_labels()[0]:
            self.ax.legend(loc="upper right")

        self.ax.set_xlim(np.min(mesh.points[:, 0]), np.max(mesh.points[:, 0]))
        self.ax.set_ylim(np.min(mesh.points[:, 1]), np.max(mesh.points[:, 1]))

    # < Move o scatter 2D para a posição do quadro atual >

    def atualizar(self, frame: int):
        if self.particulas is None or not self.historico:
            return ()

        posicoes = self.historico[min(frame, len(self.historico) - 1)]
        self.particulas.set_offsets(posicoes[:, :2] if posicoes.size else np.empty((0, 2)))
        self.ax.set_title(f"Animação 2D do escorregamento — passo {frame + 1}/{len(self.historico)}")
        return (self.particulas,)

    # < Cria a animação 2D e a mostra em janela ou grava em arquivo >

    def executar(self) -> None:
        if not self.historico:
            raise ValueError("Não há histórico de posições do MPM para animar.")

        self.configurar_figura()

        animacao = FuncAnimation(
            self.fig,
            self.atualizar,
            frames=len(self.historico),
            interval=1000 // max(1, self.fps),
            blit=False,
        )

        # < Salva a animação em GIF, se houver caminho de saída >

        if self.save_path is not None:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            animacao.save(self.save_path, writer="pillow", fps=self.fps)

        # < Exibe em janela ou apenas fecha a figura >

        if self.mostrar:
            plt.show()
        else:
            plt.close(self.fig)

    # < Plota o vetor de maior declive da superfície em projeção horizontal >

    @staticmethod
    def campo_fluxo(mesh, faces, geometria) -> None:
        _, ax = plt.subplots(figsize=(16, 7))

        # < Desenha as arestas da malha >

        ax.triplot(
            mesh.points[:, 0],
            mesh.points[:, 1],
            faces,
            color="lightgray",
            linewidth=0.5,
        )

        # < Desenha uma seta em cada centroide, no sentido da descida >

        ax.quiver(
            geometria.centroides[:, 0],
            geometria.centroides[:, 1],
            geometria.direcoes_fluxo[:, 0],
            geometria.direcoes_fluxo[:, 1],
            color="red",
            scale=100,
            width=0.0009,
        )

        # < Define o título, os eixos e a grade >

        ax.set_title("Campo Vetorial de Maior Declive da Superfície")
        ax.set_xlabel("Coordenada X (metros)")
        ax.set_ylabel("Coordenada Y (metros)")
        ax.set_aspect("equal")
        ax.grid(True)

        plt.tight_layout()
        plt.show()

    # < Mostra as ligações hidráulicas entre triângulos adjacentes da malha >

    @staticmethod
    def rede_hidraulica(mesh, faces, geometria, grafo) -> None:
        _, ax = plt.subplots(figsize=(16, 7))

        # < Desenha as arestas da malha >

        ax.triplot(
            mesh.points[:, 0],
            mesh.points[:, 1],
            faces,
            color="lightgray",
            linewidth=0.5,
        )

        # < Liga com uma linha vermelha os centroides de triângulos vizinhos >

        for ligacao in grafo.ligacoes:
            c1 = geometria.centroides[ligacao["triangulo1"]]
            c2 = geometria.centroides[ligacao["triangulo2"]]
            ax.plot([c1[0], c2[0]], [c1[1], c2[1]], color="red", linewidth=0.6)

        # < Marca cada centroide com um ponto preto >

        ax.scatter(
            geometria.centroides[:, 0],
            geometria.centroides[:, 1],
            s=5,
            color="black",
            zorder=3,
        )

        # < Define o título, os eixos e a grade >

        ax.set_title("Rede de Conectividade Hidráulica")
        ax.set_xlabel("Coordenada X (metros)")
        ax.set_ylabel("Coordenada Y (metros)")
        ax.set_aspect("equal")
        ax.grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()
        plt.show()

    # < Gera o gráfico de precipitação horária a partir do DataFrame climático >

    @staticmethod
    def chuva(clima) -> None:
        dataframe = clima.dataframe

        plt.figure(figsize=(15, 5))

        # < Plota a chuva de cada hora em barras >

        plt.bar(
            dataframe["Data_Hora"],
            clima.chuva,
            width=0.035,
            color="royalblue",
            edgecolor="black",
            linewidth=0.2,
            label="Chuva",
        )

        # < Sombreia um dia sim, um dia não, para facilitar a leitura >

        dias = dataframe["Data_Hora"].dt.normalize().drop_duplicates()

        for indice, dia in enumerate(dias):
            if indice % 2 == 0:
                plt.axvspan(dia, dia + pd.Timedelta(days=1), color="lightgray", alpha=0.15)

        # < Marca o eixo X com um rótulo por dia >

        eixo = plt.gca()
        eixo.xaxis.set_major_locator(mdates.DayLocator())
        eixo.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))

        # < Define os rótulos, o título e a grade >

        plt.xticks(rotation=45)
        plt.xlabel("Data")
        plt.ylabel("Chuva (mm/h)")
        plt.title("Intensidade Horária de Chuva")
        plt.grid(axis="y", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # < Traça a evolução temporal da saturação média, destacando os dias >

    @staticmethod
    def saturacao(clima, resultados: dict[str, object]) -> None:
        dataframe = clima.dataframe

        plt.figure(figsize=(15, 5))

        # < Plota a saturação média de cada hora >

        plt.plot(
            dataframe["Data_Hora"],
            resultados["media"],
            color="royalblue",
            linewidth=2,
            label="Saturação média",
        )

        # < Sombreia um dia sim, um dia não, para facilitar a leitura >

        dias = dataframe["Data_Hora"].dt.normalize().drop_duplicates()

        for indice, dia in enumerate(dias):
            if indice % 2 == 0:
                plt.axvspan(dia, dia + pd.Timedelta(days=1), color="lightgray", alpha=0.15)

        # < Marca o eixo X com um rótulo por dia >

        eixo = plt.gca()
        eixo.xaxis.set_major_locator(mdates.DayLocator())
        eixo.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))

        # < Define os rótulos, o título, o limite do eixo Y e a grade >

        plt.xticks(rotation=45)
        plt.xlabel("Data")
        plt.ylabel("Índice de Saturação")
        plt.title("Evolução Temporal do Índice Médio de Saturação")
        plt.ylim(0, 1)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # < Cria a animação 2D de evolução do MPM a partir do histórico de posições >

    @staticmethod
    def mpm_animacao(
        mesh,
        faces,
        historico: list[np.ndarray],
        fps: int = 10,
        save_path: str | Path | None = None,
        titulo: str = "MPM em evolução",
        regiao_instavel=None,
    ) -> FuncAnimation | None:
        if not historico:
            raise ValueError("É necessário fornecer pelo menos um passo de evolução do MPM.")

        # < Garante que cada quadro tenha formato (N, 3), mesmo sem partículas >

        historico = [np.asarray(pos, dtype=float).reshape(-1, 3) for pos in historico]

        _, ax = plt.subplots(figsize=(12, 8))
        fig = ax.figure

        # < Desenha as arestas da malha >

        ax.triplot(
            mesh.points[:, 0],
            mesh.points[:, 1],
            faces,
            color="lightgray",
            linewidth=0.5,
            alpha=0.8,
        )

        # < Destaca em vermelho os pontos da região instável >

        if regiao_instavel is not None and len(regiao_instavel.pontos) > 0:
            ax.scatter(
                regiao_instavel.pontos[:, 0],
                regiao_instavel.pontos[:, 1],
                s=30,
                c="red",
                alpha=0.5,
                edgecolors="black",
                linewidths=0.2,
                label="Região instável",
            )

        # < Desenha as partículas na posição inicial >

        posicoes = historico[0]
        scatter = ax.scatter(
            posicoes[:, 0],
            posicoes[:, 1],
            s=18,
            c="tomato",
            edgecolors="black",
            linewidths=0.5,
            label="MPM",
        )

        # < Define o título, os eixos e a grade >

        ax.set_title(titulo)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # < Ajusta os limites do gráfico à nuvem de partículas, com uma margem >

        if posicoes.size > 0:
            x_min, x_max = np.min(posicoes[:, 0]), np.max(posicoes[:, 0])
            y_min, y_max = np.min(posicoes[:, 1]), np.max(posicoes[:, 1])
            pad_x = max((x_max - x_min) * 0.15, 1.0)
            pad_y = max((y_max - y_min) * 0.15, 1.0)
            ax.set_xlim(x_min - pad_x, x_max + pad_x)
            ax.set_ylim(y_min - pad_y, y_max + pad_y)

        # < Exibe a legenda da região instável e das partículas >

        if regiao_instavel is not None and len(regiao_instavel.pontos) > 0:
            ax.legend(loc="upper right")

        # < Atualiza a posição do scatter em cada quadro da animação >

        def atualizar(frame: int):
            pos = historico[min(frame, len(historico) - 1)]
            if pos.size == 0:
                scatter.set_offsets(np.empty((0, 2)))
                return scatter

            scatter.set_offsets(pos[:, :2])
            return scatter

        anim = FuncAnimation(
            fig,
            atualizar,
            frames=len(historico),
            interval=1000 // max(1, fps),
            blit=False,
        )

        # < Salva a animação em GIF, se houver caminho de saída >

        if save_path is not None:
            caminho = Path(save_path)
            caminho.parent.mkdir(parents=True, exist_ok=True)
            anim.save(caminho, writer="pillow", fps=fps)

        plt.tight_layout()
        plt.show()
        return anim

    # < Exporta ou exibe a malha TIN em 3D, com pontos opcionais destacados >

    @staticmethod
    def malha_3d(
        mesh,
        titulo: str = "Malha TIN em 3D",
        save_path: str | Path | None = None,
        pontos_destacados: np.ndarray | None = None,
        cor_malha: str = "lightblue",
        cor_pontos: str = "tomato",
    ) -> None:
        plotter = pv.Plotter(off_screen=save_path is not None)

        # < Adiciona a malha à cena >

        malha = pv.PolyData(mesh.points)
        malha.faces = Visualizador._faces_para_polydata(mesh)
        plotter.add_mesh(malha, color=cor_malha, show_edges=True, opacity=0.85, smooth_shading=True)

        # < Destaca os pontos informados >

        if pontos_destacados is not None and len(pontos_destacados) > 0:
            pontos = pv.PolyData(np.asarray(pontos_destacados, dtype=float))
            plotter.add_points(pontos, color=cor_pontos, render_points_as_spheres=True, point_size=8)

        # < Configura os eixos, a câmera e o fundo >

        plotter.add_axes()
        plotter.camera_position = "iso"
        plotter.set_background("white")

        '''

        Passa o caminho da imagem direto para o show, pois a captura precisa
        acontecer antes de a janela ser fechada

        '''

        if save_path is not None:
            caminho = Path(save_path)
            caminho.parent.mkdir(parents=True, exist_ok=True)
            plotter.show(title=titulo, auto_close=True, screenshot=str(caminho))
        else:
            plotter.show(title=titulo, auto_close=True)

    # < Exibe a malha com a região instável e as partículas após a mobilização >

    @staticmethod
    def mpm_3d(
        mesh,
        historico: list[np.ndarray],
        regiao_instavel=None,
        save_path: str | Path | None = None,
        titulo: str = "MPM após a mobilização",
    ) -> None:
        if not historico:
            raise ValueError("É necessário fornecer pelo menos um passo de evolução do MPM.")

        # < Usa a última posição do histórico, que é a do fim da simulação >

        ultimo = np.asarray(historico[-1], dtype=float)
        plotter = pv.Plotter(off_screen=save_path is not None)

        # < Adiciona a malha à cena >

        malha = pv.PolyData(mesh.points)
        malha.faces = Visualizador._faces_para_polydata(mesh)
        plotter.add_mesh(malha, color="lightgray", show_edges=True, opacity=0.55, smooth_shading=True)

        # < Destaca em vermelho os pontos da região instável >

        if regiao_instavel is not None and len(regiao_instavel.pontos) > 0:
            pontos_instaveis = pv.PolyData(np.asarray(regiao_instavel.pontos, dtype=float))
            plotter.add_points(pontos_instaveis, color="red", render_points_as_spheres=True, point_size=10)

        # < Adiciona as partículas na posição final >

        if ultimo.size > 0:
            pontos_mpm = pv.PolyData(ultimo)
            plotter.add_points(pontos_mpm, color="tomato", render_points_as_spheres=True, point_size=9)

        # < Configura os eixos, a câmera e o fundo >

        plotter.add_axes()
        plotter.camera_position = "iso"
        plotter.set_background("white")

        # < Passa o caminho da imagem direto para o show, antes de a janela fechar >

        if save_path is not None:
            caminho = Path(save_path)
            caminho.parent.mkdir(parents=True, exist_ok=True)
            plotter.show(title=titulo, auto_close=True, screenshot=str(caminho))
        else:
            plotter.show(title=titulo, auto_close=True)

    # < Mostra a animação do MPM com o título da visualização final >

    @staticmethod
    def mpm_final(
        mesh,
        faces,
        historico: list[np.ndarray],
        regiao_instavel=None,
        fps: int = 10,
        save_path: str | Path | None = None,
    ) -> FuncAnimation | None:
        return Visualizador.mpm_animacao(
            mesh=mesh,
            faces=faces,
            historico=historico,
            fps=fps,
            save_path=save_path,
            titulo="Visualização final do MPM",
            regiao_instavel=regiao_instavel,
        )


'''

Compatibilidade: os visualizadores estáticos históricos continuam disponíveis
em Visualizador, embora a animação principal agora seja 2D

'''

for _nome_metodo in (
    "campo_fluxo",
    "rede_hidraulica",
    "chuva",
    "saturacao",
    "mpm_animacao",
    "malha_3d",
    "mpm_3d",
    "mpm_final",
):
    setattr(Visualizador, _nome_metodo, getattr(Animacao2D, _nome_metodo))