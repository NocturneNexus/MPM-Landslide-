from __future__ import annotations

from collections import defaultdict

import numpy as np


class GrafoTIN:

    """

    A classe GrafoTIN tem como objetivo estruturar a conectividade da malha
    triangular e do fluxo hidráulico. Primeiro, ela recebe a malha, as faces
    e os dados de cada ponto. Depois, o método construir_conectividade mapeia
    os vizinhos e monta a rede entre os triângulos, o método calcular_mfd
    distribui o fluxo entre os vizinhos mais baixos e o método
    calcular_area_contribuicao acumula a área que drena para cada ponto.
    Dessa forma, conseguimos representar a rede de drenagem e a propagação
    da mobilização entre os triângulos.

    """

    '''

    Inicializa o grafo armazenando a malha e os dados de cada ponto, calcula
    as razões trigonométricas da inclinação e deixa vazios os resultados que
    serão preenchidos pelas etapas seguintes.

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

        # < Armazena a largura hidráulica de cada ponto, em metros >

        largura_hidraulica,
    ):
        self.mesh = mesh
        self.faces = np.asarray(faces)
        self.vizinhos = vizinhos
        self.theta = np.asarray(theta, dtype=float)
        self.area_vertices = np.asarray(area_vertices, dtype=float)

        # < Garante que a largura tenha o mesmo formato da área dos vértices >

        self.largura_hidraulica = np.broadcast_to(
            np.asarray(largura_hidraulica, dtype=float),
            self.area_vertices.shape,
        ).copy()

        # < Separa as coordenadas dos pontos da malha >

        self.x = mesh.points[:, 0]
        self.y = mesh.points[:, 1]
        self.z = mesh.points[:, 2]

        # < Calcula o seno, o cosseno e a tangente da inclinação >

        self.sin_theta = np.sin(self.theta)
        self.cos_theta = np.cos(self.theta)
        self.tan_theta = np.tan(self.theta)

        # < Ordena os vértices do mais alto para o mais baixo >

        self.order_vertices = np.argsort(self.z)[::-1]
        self.order_triangulos = None

        # < Inicializa a topologia e o fluxo, preenchidos mais adiante >

        self.rede_triangular = None
        self.vizinhos_triangulo = None
        self.ligacoes = None
        self.fluxo_triangular = None

        # < Inicializa as áreas de contribuição >

        self.area_triangular_contribuicao = None
        self.area_contribuicao = np.copy(self.area_vertices)
        self.area_especifica = None

        # < Inicializa as variáveis hidrológicas, preenchidas pelo modelo >

        self.q = None
        self.transmissividade = None
        self.w = None
        self.h = None
        self.u = None

    # < Constrói a topologia da malha e a rede de conexões entre triângulos >

    def construir_conectividade(self, centroides: np.ndarray) -> None:

        '''

        Mapeia as arestas compartilhadas, calcula as propriedades
        geométricas de vizinhança e estrutura a rede de conexões
        e fluxos entre os elementos

        '''

        quantidade = len(self.faces)
        mapa_arestas = defaultdict(list)

        # < Registra, para cada aresta, quais triângulos a compartilham >

        for indice, triangulo in enumerate(self.faces):
            a, b, c = triangulo
            arestas = [
                tuple(sorted((a, b))),
                tuple(sorted((b, c))),
                tuple(sorted((c, a))),
            ]
            for aresta in arestas:
                mapa_arestas[aresta].append(indice)

        vizinhos_triangulo = [[] for _ in range(quantidade)]
        ligacoes = []

        for aresta, triangulos in mapa_arestas.items():

            # < Só interessam as arestas internas, compartilhadas por dois triângulos >

            if len(triangulos) != 2:
                continue

            t1, t2 = triangulos
            vizinhos_triangulo[t1].append(t2)
            vizinhos_triangulo[t2].append(t1)

            # < Obtém os dois pontos que formam a aresta compartilhada >

            p1 = self.mesh.points[aresta[0]]
            p2 = self.mesh.points[aresta[1]]

            # < Calcula o comprimento da aresta, a distância entre centroides e o desnível >

            comprimento = np.linalg.norm(p2 - p1)
            distancia = np.linalg.norm(centroides[t2] - centroides[t1])
            dz = centroides[t1][2] - centroides[t2][2]

            ligacoes.append(
                {
                    "triangulo1": t1,
                    "triangulo2": t2,
                    "aresta": aresta,
                    "comprimento": comprimento,
                    "distancia": distancia,
                    "dz": dz,
                }
            )

        # < Monta a rede, em que cada triângulo lista seus vizinhos >

        rede = [[] for _ in range(quantidade)]

        for ligacao in ligacoes:
            t1 = ligacao["triangulo1"]
            t2 = ligacao["triangulo2"]

            rede[t1].append(
                {
                    "vizinho": t2,
                    "comprimento": ligacao["comprimento"],
                    "distancia": ligacao["distancia"],
                    "dz": ligacao["dz"],
                }
            )

            # < Na direção oposta o desnível troca de sinal >

            rede[t2].append(
                {
                    "vizinho": t1,
                    "comprimento": ligacao["comprimento"],
                    "distancia": ligacao["distancia"],
                    "dz": -ligacao["dz"],
                }
            )

        # < Armazena os resultados e ordena os triângulos do mais alto para o mais baixo >

        self.rede_triangular = rede
        self.vizinhos_triangulo = vizinhos_triangulo
        self.ligacoes = ligacoes
        self.order_triangulos = np.argsort(centroides[:, 2])[::-1]

    # < Distribui o fluxo de cada triângulo entre seus vizinhos mais baixos >

    def calcular_mfd(self, geometrias, expoente: float = 1.1) -> None:

        '''

        Calcula a distribuição de fluxo multifluxo utilizando a Direção de
        Fluxo Múltiplo (MFD), que pondera as direções de escoamento e os
        declives entre os triângulos da malha

        '''

        quantidade = len(self.faces)
        fluxo = [[] for _ in range(quantidade)]

        # < Percorre os triângulos do mais alto para o mais baixo >

        for triangulo in self.order_triangulos:
            centro = geometrias.centroides[triangulo]
            direcao = geometrias.direcoes_fluxo[triangulo]

            destinos = []
            pesos = []

            for vizinho in self.rede_triangular[triangulo]:
                destino = vizinho["vizinho"]
                centro_destino = geometrias.centroides[destino]

                # < Calcula o vetor unitário que aponta do triângulo para o vizinho >

                vetor = centro_destino - centro
                distancia = np.linalg.norm(vetor)

                if distancia == 0:
                    continue

                vetor /= distancia

                # < Só recebe fluxo o vizinho alinhado com a direção de descida >

                alinhamento = np.dot(direcao, vetor)
                if alinhamento <= 0:
                    continue

                # < O vizinho também precisa estar mais baixo que o triângulo >

                diferenca_altura = centro[2] - centro_destino[2]
                if diferenca_altura <= 0:
                    continue

                # < O peso depende do alinhamento, do declive e do comprimento da aresta >

                declive = diferenca_altura / distancia
                peso = alinhamento * declive**expoente * vizinho["comprimento"]

                destinos.append(destino)
                pesos.append(peso)

            # < Normaliza os pesos para que a soma da fração distribuída seja 1 >

            if pesos:
                pesos = np.asarray(pesos, dtype=float)
                soma = pesos.sum()
                if soma > 0:
                    pesos /= soma
                    fluxo[triangulo] = list(zip(destinos, pesos))

        self.fluxo_triangular = fluxo

    # < Acumula a área que drena para cada triângulo e cada ponto >

    def calcular_area_contribuicao(self, geometrias) -> None:

        # < Começa com a área do próprio triângulo >

        areas = np.copy(geometrias.areas)

        '''

        Propaga a área de cima para baixo. Como os triângulos estão
        ordenados do mais alto para o mais baixo, a área de um triângulo
        já está completa quando ele repassa sua parte aos vizinhos

        '''

        for triangulo in self.order_triangulos:
            for destino, peso in self.fluxo_triangular[triangulo]:
                areas[destino] += areas[triangulo] * peso

        # < Converte a área dos triângulos em área dos vértices pela média >

        area_vertices = np.zeros(self.mesh.n_points)
        contador = np.zeros(self.mesh.n_points)

        for indice, triangulo in enumerate(self.faces):
            for vertice in triangulo:
                area_vertices[vertice] += areas[indice]
                contador[vertice] += 1

        mascara = contador > 0
        area_vertices[mascara] /= contador[mascara]

        # < Calcula a área específica dividindo pela largura hidráulica >

        area_especifica = np.zeros_like(area_vertices)
        mascara = self.largura_hidraulica > 0
        area_especifica[mascara] = (
            area_vertices[mascara] / self.largura_hidraulica[mascara]
        )

        # < Armazena os resultados no grafo >

        self.area_triangular_contribuicao = areas
        self.area_contribuicao = area_vertices
        self.area_especifica = area_especifica

        # < Salva os resultados na malha para visualização >

        self.mesh.cell_data["AreaContribuicaoTriangulo"] = areas
        self.mesh.point_data["AreaContribuicao"] = area_vertices
        self.mesh.point_data["AreaEspecifica"] = area_especifica
        self.mesh.point_data["AreaEspecificaLog"] = np.log10(area_especifica + 1)