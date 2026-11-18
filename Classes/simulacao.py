from __future__ import annotations

import numpy as np

from .Clima import ClimaNASA
from .Grafo import GrafoTIN
from .hidrologia import ModeloHidrologico
from .mpm import GeradorParticulasMPM, RegiaoInstavel
from .particulas import AnaliseParticulas
from .shalstab import SHALSTAB
from .tin import GeometriaTIN
from .visualizacao import Visualizador


class SimulacaoTIN:

    """

    A classe SimulacaoTIN tem como objetivo orquestrar o pipeline completo
    da malha TIN para hidrologia, SHALSTAB e MPM. Primeiro, o método
    preparar_rede calcula a geometria, a conectividade, o fluxo e a área de
    contribuição. Depois, os métodos de clima e hidrologia simulam a chuva e a
    saturação, e o método calcular_shalstab classifica a estabilidade. Por
    fim, a região instável é selecionada e transformada em partículas, cujo
    movimento é simulado e visualizado em 2D ou 3D.

    """

    '''

    Inicializa a simulação armazenando a malha, cria o grafo e a geometria
    e deixa vazias as etapas que serão preenchidas ao longo do pipeline.

    '''

    def __init__(
        self,

        # < Armazena a malha triangular com os pontos e os dados >

        mesh,

        # < Armazena os índices dos três vértices de cada triângulo >

        faces,

        # < Armazena a lista de triângulos vizinhos de cada triângulo >

        vizinhos,

        # < Armazena a inclinação do terreno em cada ponto, em radianos >

        theta,

        # < Armazena a área associada a cada vértice >

        area_vertices,

        # < Armazena a largura hidráulica de cada ponto >

        largura_hidraulica,

        # < Armazena os pontos do MPM para a análise de seção (opcional) >

        pontos_mpm=None,

        # < Armazena os pontos da primeira parte da encosta (opcional) >

        pontos_parte1=None,

        # < Armazena os pontos da segunda parte da encosta (opcional) >

        pontos_parte2=None,
    ):
        self.mesh = mesh
        self.faces = faces

        # < Cria o grafo com a conectividade e o fluxo da malha >

        self.grafo = GrafoTIN(
            mesh=mesh,
            faces=faces,
            vizinhos=vizinhos,
            theta=theta,
            area_vertices=area_vertices,
            largura_hidraulica=largura_hidraulica,
        )

        # < Cria a geometria dos triângulos >

        self.geometria = GeometriaTIN(mesh, faces)

        # < Inicializa as etapas que serão preenchidas ao longo do pipeline >

        self.hidrologia = None
        self.clima = None
        self.resultados_saturacao = None
        self.shalstab = None
        self.regiao_instavel = None
        self.mpm = None
        self.particulas = None

        # < Cria a análise de seção, se os pontos de referência foram informados >

        if pontos_mpm is not None:
            self.particulas = AnaliseParticulas(
                pontos_mpm=pontos_mpm,
                pontos_parte1=pontos_parte1,
                pontos_parte2=pontos_parte2,
            )

    # < Calcula a geometria, a conectividade, o fluxo e a hidrologia estática >

    def preparar_rede(self) -> None:
        self.geometria.calcular()
        self.grafo.construir_conectividade(centroides=self.geometria.centroides)
        self.grafo.calcular_mfd(geometrias=self.geometria, expoente=1.1)
        self.grafo.calcular_area_contribuicao(geometrias=self.geometria)
        self.hidrologia = ModeloHidrologico(self.grafo)

    # < Baixa os dados de chuva e temperatura do período informado >

    def carregar_clima(self, latitude: float, longitude: float, inicio: str, fim: str) -> None:
        self.clima = ClimaNASA(
            latitude=latitude,
            longitude=longitude,
            inicio=inicio,
            fim=fim,
        )
        self.clima.download()

    # < Simula a saturação do solo ao longo da série de chuva >

    def executar_hidrologia_temporal(self) -> None:
        if self.clima is None:
            raise RuntimeError("O clima precisa ser carregado antes da simulação.")
        if self.hidrologia is None:
            raise RuntimeError("A rede TIN precisa ser preparada antes da simulação.")

        self.resultados_saturacao = self.hidrologia.simular_chuva(self.clima.chuva_si)

    # < Calcula o fator de segurança com a saturação crítica da chuva >

    def calcular_shalstab(self, **kwargs):
        if self.hidrologia is None:
            raise RuntimeError("A hidrologia precisa ser calculada antes do SHALSTAB.")

        self.shalstab = SHALSTAB(
            grafo=self.grafo,
            geometrias=self.geometria,
            hidrologia=self.hidrologia,
            **kwargs,
        )
        return self.shalstab

    # < Seleciona os pontos com fator de segurança menor que o critério >

    def selecionar_regiao_instavel(self, criterio_fs: float = 1.0):
        if self.shalstab is None:
            self.calcular_shalstab()

        self.regiao_instavel = RegiaoInstavel(
            grafo=self.grafo,
            geometrias=self.geometria,
            shalstab=self.shalstab,
            criterio_fs=criterio_fs,
        )
        return self.regiao_instavel

    # < Define a densidade do solo, usando a do SHALSTAB quando não informada >

    def _densidade_solo(self, densidade_solo: float | None) -> float:
        if densidade_solo is not None:
            return densidade_solo
        if self.shalstab is not None:
            return self.shalstab.parametros.densidade_solo
        return 1700.0

    # < Garante que a região instável e o gerador de partículas existam >

    def _garantir_mpm(self, densidade_solo: float | None = None, volume_base: float = 1.0):
        if self.regiao_instavel is None:
            self.selecionar_regiao_instavel()

        if self.mpm is None:
            self.mpm = GeradorParticulasMPM(
                regiao_instavel=self.regiao_instavel,
                densidade_solo=self._densidade_solo(densidade_solo),
                volume_base=volume_base,
            )

        return self.mpm

    # < Gera as partículas do MPM a partir da região instável >

    def gerar_particulas_mpm(self, densidade_solo: float | None = None, volume_base: float = 1.0):
        if self.regiao_instavel is None:
            self.selecionar_regiao_instavel()

        self.mpm = GeradorParticulasMPM(
            regiao_instavel=self.regiao_instavel,
            densidade_solo=self._densidade_solo(densidade_solo),
            volume_base=volume_base,
        )
        return self.mpm

    # < Executa a simulação do MPM e retorna o histórico de posições >

    def executar_mpm(
        self,
        dt: float = 0.05,
        passos: int = 1,
        gravidade: float = 9.81,
        coeficiente_resistencia: float = 0.15,
        espacamento: float = 2.0,
        densidade_solo: float | None = None,
        volume_base: float = 1.0,
    ):
        self._garantir_mpm(densidade_solo, volume_base)

        historico = self.mpm.atualizar(
            dt=dt,
            passos=passos,
            gravidade=gravidade,
            coeficiente_resistencia=coeficiente_resistencia,
            espacamento=espacamento,
        )
        return historico

    # < Simula o MPM e mostra a animação 2D, ou apenas o resultado final >

    def visualizar_mpm(
        self,
        dt: float = 0.05,
        passos: int = 10,
        gravidade: float = 9.81,
        coeficiente_resistencia: float = 0.15,
        espacamento: float = 2.0,
        save_path: str | None = None,
        final: bool = False,
    ) -> list[np.ndarray]:
        historico = self.executar_mpm(
            dt=dt,
            passos=passos,
            gravidade=gravidade,
            coeficiente_resistencia=coeficiente_resistencia,
            espacamento=espacamento,
        )

        # < Sem partículas não há o que animar, então avisa e encerra >

        if len(self.mpm.particulas) == 0:
            print("Aviso: nenhuma região instável foi encontrada, não há partículas para animar.")
            return historico

        if final:
            Visualizador.mpm_final(
                self.mesh,
                self.faces,
                historico,
                regiao_instavel=self.regiao_instavel,
                fps=10,
                save_path=save_path,
            )
        else:
            Visualizador.mpm_animacao(
                self.mesh,
                self.faces,
                historico,
                fps=10,
                save_path=save_path,
                titulo="Simulação MPM da região instável",
                regiao_instavel=self.regiao_instavel,
            )

        return historico

    # < Exibe a malha TIN inicial em 3D, destacando a região instável >

    def visualizar_malha_3d(self, save_path: str | None = None) -> None:
        if self.regiao_instavel is None:
            self.selecionar_regiao_instavel()

        Visualizador.malha_3d(
            self.mesh,
            titulo="Malha TIN inicial em 3D",
            save_path=save_path,
            pontos_destacados=self.regiao_instavel.pontos if self.regiao_instavel is not None else None,
        )

    # < Simula o MPM e exibe a malha com as partículas em 3D >

    def visualizar_mpm_3d(
        self,
        dt: float = 0.05,
        passos: int = 10,
        gravidade: float = 9.81,
        coeficiente_resistencia: float = 0.15,
        espacamento: float = 2.0,
        save_path: str | None = None,
    ) -> list[np.ndarray]:
        historico = self.executar_mpm(
            dt=dt,
            passos=passos,
            gravidade=gravidade,
            coeficiente_resistencia=coeficiente_resistencia,
            espacamento=espacamento,
        )

        Visualizador.mpm_3d(
            self.mesh,
            historico,
            regiao_instavel=self.regiao_instavel,
            save_path=save_path,
            titulo="Malha e MPM em 3D",
        )

        return historico

    # < Mostra todos os gráficos disponíveis dos resultados da simulação >

    def mostrar_resultados(self) -> None:

        # < Mostra a análise de seção, que só existe se os pontos de referência foram informados >

        if isinstance(self.particulas, AnaliseParticulas):
            self.particulas.plotar()

        # < Mostra o campo de fluxo e a rede hidráulica >

        Visualizador.campo_fluxo(self.mesh, self.faces, self.geometria)
        Visualizador.rede_hidraulica(self.mesh, self.faces, self.geometria, self.grafo)

        # < Mostra a chuva e a saturação ao longo do tempo >

        if self.clima is not None:
            Visualizador.chuva(self.clima)

        if self.resultados_saturacao is not None:
            Visualizador.saturacao(self.clima, self.resultados_saturacao)

        # < Mostra o fator de segurança e a região instável >

        if self.shalstab is not None:
            self.shalstab.plotar()

        if self.regiao_instavel is not None:
            self.regiao_instavel.plotar()

        # < Mostra a evolução do MPM, se houver partículas >

        if self.mpm is not None and len(self.mpm.particulas) > 0:
            historico = self.mpm.atualizar(dt=0.05, passos=6)
            Visualizador.mpm_animacao(self.mesh, self.faces, historico, fps=8, titulo="MPM em evolução")

    # < Imprime no terminal um resumo da rede, da hidrologia e do SHALSTAB >

    def imprimir_resumo(self) -> None:

        # < Calcula as estatísticas da rede de triângulos >

        quantidade_triangulos = len(self.faces)
        quantidade_ligacoes = len(self.grafo.ligacoes)
        isolados = sum(len(vizinhos) == 0 for vizinhos in self.grafo.vizinhos_triangulo)
        media_vizinhos = np.mean([len(vizinhos) for vizinhos in self.grafo.vizinhos_triangulo])
        quantidade_divisoes = sum(len(fluxo) > 1 for fluxo in self.grafo.fluxo_triangular)
        quantidade_fluxo_unico = sum(len(fluxo) == 1 for fluxo in self.grafo.fluxo_triangular)

        print("\nRede Hidráulica:")
        print(f"Triângulos: {quantidade_triangulos:,}")
        print(f"Ligações: {quantidade_ligacoes:,}")
        print(f"Triângulos isolados: {isolados}")
        print(f"Média de vizinhos: {media_vizinhos:.2f}")

        print("\nMFD-TIN:")
        print(f"Fluxo único: {quantidade_fluxo_unico:,}")
        print(f"Fluxo múltiplo: {quantidade_divisoes:,}")
        print(f"Percentual MFD: {100 * quantidade_divisoes / quantidade_triangulos:.1f}%")

        # < A saturação impressa é a crítica, isto é, a máxima do período de chuva >

        print("\nHidrologia:")
        print(f"Recarga: {self.hidrologia.parametros.recarga_base:.2e} m/s")
        print(f"Ks: {self.hidrologia.parametros.condutividade_saturada:.2e} m/s")
        print(f"a/b máximo: {self.grafo.area_especifica.max():.2f}")
        print(f"Índice médio de saturação (crítico): {self.grafo.w.mean():.3f}")
        print(f"Máxima saturação (crítica): {self.grafo.w.max():.3f}")
        print(f"Espessura média saturada: {self.grafo.h.mean():.2f} m")

        if self.shalstab is not None:
            print("\nSHALSTAB:")
            print(f"FS médio: {self.shalstab.fator_seguranca.mean():.3f}")
            print(f"Regiões instáveis: {self.shalstab.regioes_instaveis.sum():,}")