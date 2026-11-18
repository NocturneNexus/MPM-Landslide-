from __future__ import annotations

import numpy as np

from .Base import PropriedadesTriangulo


class GeometriaTIN:

    """

    A classe GeometriaTIN tem como objetivo calcular as grandezas geométricas
    e hidráulicas dos triângulos da malha. Primeiro, ela recebe a malha e as
    faces. Depois, o método calcular deriva a normal, o centroide, a área, a
    declividade e a direção de fluxo de cada elemento. Por fim, o método
    _salvar_no_mesh grava essas informações no próprio objeto mesh, para que
    possam ser visualizadas e usadas pelas outras etapas do pipeline.

    """

    '''

    Inicializa a geometria armazenando a malha e as faces, e deixa vazios
    os arrays que serão preenchidos pelo cálculo.

    '''

    def __init__(
        self,

        # < Armazena a malha triangular com os pontos e os dados >

        mesh,

        # < Armazena os índices dos três vértices de cada triângulo >

        faces,
    ):
        self.mesh = mesh
        self.faces = np.asarray(faces)

        # < Inicializa os resultados que serão preenchidos pelo cálculo >

        self.normais = None
        self.centroides = None
        self.areas = None
        self.declividades = None
        self.direcoes_fluxo = None
        self.comprimentos_arestas = None
        self.triangulos: list[PropriedadesTriangulo] = []

    # < Calcula as grandezas geométricas de todos os triângulos >

    def calcular(self) -> None:

        # < Define a quantidade de triângulos da malha >

        quantidade = len(self.faces)

        # < Cria os arrays que receberão os resultados >

        self.normais = np.zeros((quantidade, 3))
        self.centroides = np.zeros((quantidade, 3))
        self.areas = np.zeros(quantidade)
        self.declividades = np.zeros(quantidade)
        self.direcoes_fluxo = np.zeros((quantidade, 3))
        self.comprimentos_arestas = np.zeros((quantidade, 3))

        # < Reinicia a lista de triângulos para não duplicar em um novo cálculo >

        self.triangulos = []

        for indice, face in enumerate(self.faces):

            # < Obtém as coordenadas dos três vértices do triângulo >

            v1 = self.mesh.points[face[0]]
            v2 = self.mesh.points[face[1]]
            v3 = self.mesh.points[face[2]]

            # < Calcula dois vetores que partem do primeiro vértice >

            e1 = v2 - v1
            e2 = v3 - v1

            # < O produto vetorial dá a direção da normal e o dobro da área >

            produto_vetorial = np.cross(e1, e2)
            norma = np.linalg.norm(produto_vetorial)

            # < Ignora triângulos degenerados, que têm área zero >

            if norma == 0:
                continue

            # < Normaliza o vetor e garante que a normal aponte para cima >

            normal = produto_vetorial / norma

            if normal[2] < 0:
                normal *= -1

            # < Calcula a área e o centroide do triângulo >

            area = 0.5 * norma
            centroide = (v1 + v2 + v3) / 3.0

            # < Calcula a declividade como o ângulo entre a normal e a vertical >

            declividade = np.arctan2(
                np.sqrt(normal[0] ** 2 + normal[1] ** 2),
                abs(normal[2]),
            )

            '''

            Calcula a direção de maior declive. Com a normal apontando para
            cima, a componente horizontal da normal aponta no sentido da
            descida da superfície, por isso ela é usada sem inverter o sinal

            '''

            direcao = np.array([normal[0], normal[1], 0.0])
            norma_direcao = np.linalg.norm(direcao)

            # < Normaliza a direção, exceto em triângulos horizontais >

            if norma_direcao > 0:
                direcao /= norma_direcao

            # < Calcula o comprimento das três arestas >

            arestas = np.array(
                [
                    np.linalg.norm(v2 - v1),
                    np.linalg.norm(v3 - v2),
                    np.linalg.norm(v1 - v3),
                ]
            )

            # < Armazena os resultados do triângulo nos arrays >

            self.normais[indice] = normal
            self.centroides[indice] = centroide
            self.areas[indice] = area
            self.declividades[indice] = declividade
            self.direcoes_fluxo[indice] = direcao
            self.comprimentos_arestas[indice] = arestas

            # < Guarda também um objeto com as propriedades do triângulo >

            self.triangulos.append(
                PropriedadesTriangulo(
                    id=indice,
                    vertices=tuple(face),
                    area=area,
                    centroide=centroide,
                    normal=normal,
                    declividade=declividade,
                    direcao_fluxo=direcao,
                    arestas=arestas,
                )
            )

        # < Salva os resultados no objeto mesh >

        self._salvar_no_mesh()

    # < Grava as grandezas calculadas nos dados de célula da malha >

    def _salvar_no_mesh(self) -> None:
        self.mesh.cell_data["AreaTriangulo"] = self.areas

        # < A declividade é salva em graus para facilitar a leitura >

        self.mesh.cell_data["DeclividadeTriangulo"] = np.degrees(self.declividades)

        # < Salva as três componentes da normal >

        self.mesh.cell_data["NormalX"] = self.normais[:, 0]
        self.mesh.cell_data["NormalY"] = self.normais[:, 1]
        self.mesh.cell_data["NormalZ"] = self.normais[:, 2]

        # < Salva as três componentes da direção de fluxo >

        self.mesh.cell_data["FluxoX"] = self.direcoes_fluxo[:, 0]
        self.mesh.cell_data["FluxoY"] = self.direcoes_fluxo[:, 1]
        self.mesh.cell_data["FluxoZ"] = self.direcoes_fluxo[:, 2]

        # < Salva o comprimento de cada aresta >

        self.mesh.cell_data["ComprimentoAresta1"] = self.comprimentos_arestas[:, 0]
        self.mesh.cell_data["ComprimentoAresta2"] = self.comprimentos_arestas[:, 1]
        self.mesh.cell_data["ComprimentoAresta3"] = self.comprimentos_arestas[:, 2]