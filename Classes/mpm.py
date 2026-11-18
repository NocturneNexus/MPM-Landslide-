"""

Módulo do Método do Ponto Material (MPM). Este módulo implementa a camada de
dinâmica do material instável identificado pelo SHALSTAB, em uma versão
simplificada de sistema de partículas com contato, sem a etapa de grade. Ele
transforma a região instável em partículas discretas, calcula a aceleração da
gravidade projetada sobre a encosta, integra a velocidade e a posição, aplica
contato com o terreno, com os contornos da malha e entre as partículas, e
registra o histórico temporal usado pela animação.

Fluxo principal:
    1. RegiaoInstavel identifica os pontos instáveis.
    2. GeradorParticulasMPM cria uma partícula por ponto instável.
    3. calcular_forcas() define a aceleração de cada partícula.
    4. atualizar() integra o sistema em passos temporais.
    5. historico_posicoes armazena a trajetória de cada partícula.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from .Base import ParticulaMPM


@dataclass
class RegiaoInstavel:

    """

    A classe RegiaoInstavel tem como objetivo representar a região do terreno
    classificada como instável pelo SHALSTAB. Primeiro, ela seleciona os
    vértices com fator de segurança menor que o critério. Depois, converte
    esses índices em pontos físicos da malha e calcula o centroide da
    vizinhança de cada vértice. Por fim, resume a massa da região, que alimenta
    a geração de partículas do MPM.

    """

    # < Armazena o grafo com a malha e as faces >

    grafo: object

    # < Armazena as geometrias dos triângulos (opcional) >

    geometrias: Optional[object] = None

    # < Armazena o SHALSTAB com o fator de segurança (opcional) >

    shalstab: Optional[object] = None

    # < Define o fator de segurança abaixo do qual o ponto é instável >

    criterio_fs: float = 1.0

    # < Armazena o resumo da massa instável, calculado na inicialização >

    massa_instavel: Optional[object] = None

    # < Inicializa a busca por pontos instáveis e resume a massa da região >

    def __post_init__(self) -> None:
        self.indice = self._selecionar_indices()
        self.pontos = self._obter_pontos()
        self.centroides = self._obter_centroides()
        self.quantidade = len(self.indice)
        self.massa_instavel = self.calcular_massa_instavel()

    # < Retorna a quantidade de pontos da malha >

    def _numero_pontos(self) -> int:
        mesh = getattr(self.grafo, "mesh", None)
        return 0 if mesh is None else len(mesh.points)

    # < Retorna os índices dos vértices com fator de segurança menor que o critério >

    def _selecionar_indices(self) -> np.ndarray:
        if self.shalstab is None:
            return np.empty(0, dtype=int)

        fator = np.asarray(
            getattr(self.shalstab, "fator_seguranca", np.empty(0)),
            dtype=float,
        )

        if fator.size:
            return np.flatnonzero(fator < self.criterio_fs)

        # < Sem fator de segurança, usa a máscara de regiões instáveis >

        regioes = np.asarray(
            getattr(self.shalstab, "regioes_instaveis", np.empty(0, dtype=bool)),
            dtype=bool,
        )
        return np.flatnonzero(regioes)

    # < Extrai da malha as coordenadas 3D dos pontos instáveis >

    def _obter_pontos(self) -> np.ndarray:
        mesh = getattr(self.grafo, "mesh", None)
        if mesh is None:
            return np.empty((0, 3), dtype=float)

        pontos = np.asarray(mesh.points, dtype=float)
        if pontos.ndim != 2 or pontos.shape[1] != 3:
            raise ValueError("A malha precisa fornecer pontos com três coordenadas.")

        return pontos[self.indice]

    # < Calcula o centroide da vizinhança de cada vértice instável >

    def _obter_centroides(self) -> np.ndarray:

        '''

        Os índices instáveis são de vértices, mas os centroides da geometria
        são de triângulos. Por isso, cada vértice recebe a média dos
        centroides dos triângulos que o contêm, mantendo o mesmo índice
        dos pontos instáveis

        '''

        centroides_triangulos = getattr(self.geometrias, "centroides", None)
        faces = getattr(self.grafo, "faces", None)

        # < Sem geometria ou faces, usa os próprios pontos como centroides >

        if centroides_triangulos is None or faces is None or self._numero_pontos() == 0:
            return self.pontos.copy()

        faces = np.asarray(faces)
        centroides_triangulos = np.asarray(centroides_triangulos, dtype=float)

        # < Soma os centroides dos triângulos que tocam cada vértice >

        soma = np.zeros((self._numero_pontos(), 3))
        contador = np.zeros(self._numero_pontos())

        for canto in range(3):
            np.add.at(soma, faces[:, canto], centroides_triangulos)
            np.add.at(contador, faces[:, canto], 1.0)

        # < Calcula a média e seleciona somente os vértices instáveis >

        media = soma / np.maximum(contador, 1.0)[:, None]
        return media[self.indice]

    # < Calcula a área associada a cada vértice da malha >

    def _area_por_vertice(self) -> np.ndarray:
        n_pontos = self._numero_pontos()
        areas = getattr(self.geometrias, "areas", None)
        faces = getattr(self.grafo, "faces", None)

        # < Sem geometria, usa a área dos vértices guardada no grafo >

        if areas is None or faces is None:
            area_grafo = getattr(self.grafo, "area_vertices", np.zeros(n_pontos))
            return np.asarray(area_grafo, dtype=float)

        # < Cada triângulo reparte sua área igualmente entre seus três vértices >

        faces = np.asarray(faces)
        area_vertices = np.zeros(n_pontos)

        for canto in range(3):
            np.add.at(area_vertices, faces[:, canto], np.asarray(areas, dtype=float) / 3.0)

        return area_vertices

    # < Agrega a massa da região instável para diagnóstico e rastreio >

    def calcular_massa_instavel(self) -> MassaInstavel:

        # < Sem pontos instáveis, retorna uma massa vazia >

        if self.indice.size == 0:
            return MassaInstavel(np.empty(0, dtype=int), np.empty((0, 3)), np.empty(0), 0.0)

        n_pontos = self._numero_pontos()

        # < Obtém o fator de segurança dos pontos instáveis >

        fator = np.asarray(
            getattr(self.shalstab, "fator_seguranca", np.ones(n_pontos)),
            dtype=float,
        )
        fs = fator[self.indice]

        # < Obtém a espessura e a densidade do solo usadas no SHALSTAB >

        espessura = np.asarray(
            getattr(self.shalstab, "espessura_solo", np.ones(n_pontos)),
            dtype=float,
        )
        parametros = getattr(self.shalstab, "parametros", None)
        densidade = getattr(parametros, "densidade_solo", 1700.0)

        '''

        Calcula a massa total como a soma de área, espessura e densidade
        de cada vértice instável, ou seja, a massa de solo em quilogramas

        '''

        volume = self._area_por_vertice()[self.indice] * espessura[self.indice]
        massa_total = float(np.sum(volume * densidade))

        return MassaInstavel(self.indice.copy(), self.centroides.copy(), fs, massa_total)

    # < Exibe a região instável em projeção 2D para inspeção visual >

    def plotar(self) -> None:
        if not len(self.pontos):
            return

        _, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(self.pontos[:, 0], self.pontos[:, 1], c="red", s=20)
        ax.set_title("Região instável identificada por SHALSTAB")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.tight_layout()
        plt.show()


@dataclass
class MassaInstavel:

    """

    A classe MassaInstavel resume a massa instável após a classificação do
    SHALSTAB. Ela guarda os índices da malha em risco, os centroides
    associados e o fator de segurança de cada elemento, além da massa total
    em quilogramas. Isso permite mapear a região instável para a geração do
    sistema de partículas.

    """

    # < Índices dos vértices da malha em risco >

    indices: np.ndarray

    # < Centroides associados a cada vértice em risco >

    centroides: np.ndarray

    # < Fator de segurança de cada vértice em risco >

    fs: np.ndarray

    # < Massa total da região instável em quilogramas >

    massa_total: float

    # < Número de elementos instáveis presentes na região >

    @property
    def quantidade(self) -> int:
        return len(self.indices)

    # < Centro geométrico da região instável em coordenadas do terreno >

    def centro_geom(self) -> np.ndarray:
        return np.mean(self.centroides, axis=0) if self.quantidade else np.zeros(3)


class GeradorParticulasMPM:

    """

    A classe GeradorParticulasMPM tem como objetivo simular o movimento da
    massa instável com uma partícula material por ponto instável. Primeiro,
    ela cria as partículas a partir da região instável. Depois, o método
    calcular_forcas projeta a gravidade sobre a superfície local do terreno e
    o método atualizar integra velocidade e posição, aplicando resistência,
    contato com o terreno, contornos da malha e colisões entre partículas. Por
    fim, o histórico de posições guarda a trajetória usada na animação. As
    etapas seguem a ordem: partículas, aceleração, integração, contato e
    histórico.

    """

    '''

    Inicializa a massa instável como sistema de partículas discretas,
    prepara a busca de vizinhos na malha e cria os históricos vazios.

    '''

    def __init__(
        self,

        # < Armazena a região instável que origina as partículas >

        regiao_instavel: RegiaoInstavel,

        # < Define a densidade do solo em quilogramas por metro cúbico >

        densidade_solo: float = 1800.0,

        # < Define o volume de cada partícula em metros cúbicos >

        volume_base: float = 1.0,
    ):
        if densidade_solo <= 0 or volume_base <= 0:
            raise ValueError("densidade_solo e volume_base devem ser positivos.")

        self.regiao_instavel = regiao_instavel
        self.densidade_solo = float(densidade_solo)
        self.volume_base = float(volume_base)
        self.massa_instavel = regiao_instavel.massa_instavel

        # < Obtém a malha do terreno, necessária para o contato das partículas >

        mesh = getattr(regiao_instavel.grafo, "mesh", None)
        if mesh is None:
            raise ValueError("A região instável precisa de uma malha de terreno.")

        '''

        Guarda os pontos da malha e uma árvore de busca no plano XY, que
        encontra os vértices mais próximos de cada partícula sem varrer
        a malha inteira

        '''

        self.pontos_malha = np.asarray(mesh.points, dtype=float)
        self.arvore_malha = cKDTree(self.pontos_malha[:, :2])

        # < Cria as partículas, os vizinhos e o estado de mobilização >

        self.particulas = self.gerar()
        self.grid = None
        self.vizinhos = self._construir_vizinhos()
        self.mobilizacao = np.array(
            [p.propriedades["mobilizada"] for p in self.particulas], dtype=bool
        )

        # < Inicializa os históricos e o diagnóstico >

        self.historico_posicoes: list[np.ndarray] = []
        self.historico_velocidades: list[np.ndarray] = []
        self.historico_tempos: list[float] = []
        self.debug_log: list[dict[str, float]] = []

    # < Cria uma partícula para cada ponto instável com propriedades físicas simples >

    def gerar(self) -> list[ParticulaMPM]:

        # < Obtém o fator de segurança e a saturação de cada ponto da malha >

        fator = np.asarray(
            getattr(self.regiao_instavel.shalstab, "fator_seguranca", np.ones(0)),
            dtype=float,
        )
        saturacao = np.asarray(
            getattr(getattr(self.regiao_instavel, "grafo", None), "w", np.zeros(0)),
            dtype=float,
        )

        particulas = []

        for local, (indice, ponto) in enumerate(zip(self.regiao_instavel.indice, self.regiao_instavel.pontos)):

            # < Lê o fator de segurança e a saturação do ponto, com valores padrão de segurança >

            fs = float(fator[indice]) if indice < len(fator) else 1.0
            sat = float(np.clip(saturacao[indice], 0.0, 1.0)) if indice < len(saturacao) else 0.0

            particulas.append(
                ParticulaMPM(
                    id=local,
                    posicao=np.asarray(ponto, dtype=float).copy(),
                    massa=self.densidade_solo * self.volume_base,
                    volume=self.volume_base,
                    densidade=self.densidade_solo,
                    velocidade=np.zeros(3),
                    aceleracao=np.zeros(3),
                    propriedades={
                        "fs": fs,
                        "indice_ponto": int(indice),
                        "mobilizada": bool(fs < 1.0 or sat > 0.55),
                        "saturacao": sat,
                        "poropressao": 0.0,
                        "ativa": True,
                        "depositada": False,
                    },
                )
            )

        return particulas

    # < Associa cada partícula às k partículas mais próximas >

    def _construir_vizinhos(self, k: int = 6) -> list[list[int]]:
        quantidade = len(self.particulas)

        if quantidade < 2:
            return [[] for _ in self.particulas]

        # < Consulta o vizinho mais próximo com uma árvore de busca >

        posicoes = self.posicoes
        k_efetivo = min(k, quantidade - 1)
        _, indices = cKDTree(posicoes).query(posicoes, k=k_efetivo + 1)
        indices = indices.reshape(quantidade, -1)

        # < Remove a própria partícula da lista de vizinhos >

        return [
            [int(j) for j in linha if j != i][:k_efetivo]
            for i, linha in enumerate(indices)
        ]

    # < Retorna os limites da malha em X, Y e Z para aplicar contornos do domínio >

    def _limites(self) -> tuple[float, float, float, float, float, float]:
        return tuple(
            float(v)
            for v in (*self.pontos_malha.min(axis=0), *self.pontos_malha.max(axis=0))
        )

    # < Retorna um array booleano indicando quais partículas estão ativas >

    def _ativas(self) -> np.ndarray:
        return np.array([bool(p.propriedades["ativa"]) for p in self.particulas], dtype=bool)

    # < Ajusta um plano ao terreno ao redor de cada partícula >

    def _ajuste_local(self, posicoes: np.ndarray) -> np.ndarray:

        '''

        Busca os vértices mais próximos de cada partícula no plano XY e
        ajusta, por mínimos quadrados, o plano dz = a*dx + b*dy + c0 em
        coordenadas relativas à partícula. Os coeficientes a e b são o
        gradiente do terreno e c0 é a diferença de altura entre o terreno
        e a partícula, o que interpola suavemente a elevação

        '''

        quantidade = len(posicoes)
        if quantidade == 0:
            return np.empty((0, 3))

        k = min(12, len(self.pontos_malha))
        _, indices = self.arvore_malha.query(posicoes[:, :2], k=k)
        indices = indices.reshape(quantidade, k)

        # < Monta as coordenadas dos vizinhos relativas a cada partícula >

        relativo = self.pontos_malha[indices] - posicoes[:, None, :]
        matriz = np.concatenate(
            [relativo[:, :, :2], np.ones((quantidade, k, 1))],
            axis=2,
        )

        # < Resolve os mínimos quadrados de todas as partículas de uma vez >

        return np.einsum("nij,nj->ni", np.linalg.pinv(matriz), relativo[:, :, 2])

    # < Calcula a elevação interpolada do terreno sob cada partícula >

    def _alturas_terreno(self, posicoes: np.ndarray) -> np.ndarray:
        if len(posicoes) == 0:
            return np.empty(0)

        return posicoes[:, 2] + self._ajuste_local(posicoes)[:, 2]

    # < Calcula a aceleração da gravidade projetada sobre a superfície do terreno >

    def _aceleracoes(self, posicoes: np.ndarray, gravidade: float) -> np.ndarray:

        '''

        Obtém a normal da superfície a partir do gradiente local e retira da
        gravidade a componente perpendicular ao terreno. Sobra a componente
        tangente, que aponta no sentido da descida e tem módulo g*sen(theta),
        isto é, a aceleração de um corpo deslizando sem atrito

        '''

        coeficientes = self._ajuste_local(posicoes)

        normal = np.column_stack(
            (-coeficientes[:, 0], -coeficientes[:, 1], np.ones(len(posicoes)))
        )
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)

        vetor_gravidade = np.array([0.0, 0.0, -gravidade])
        componente_normal = normal @ vetor_gravidade

        return vetor_gravidade[None, :] - componente_normal[:, None] * normal

    # < Obtém o vetor unitário de maior declive em torno da posição da partícula >

    def _direcao_descida(self, posicao: np.ndarray) -> np.ndarray:
        aceleracao = self._aceleracoes(np.asarray(posicao, dtype=float)[None, :], 1.0)[0]
        norma = np.linalg.norm(aceleracao)

        return aceleracao / norma if norma > 1e-12 else np.zeros(3)

    # < Impede que as partículas penetrem no terreno e amortece o movimento >

    def _contato_terreno(
        self,
        posicoes: np.ndarray,
        velocidades: np.ndarray,
        ativas: np.ndarray,
        folga: float = 0.15,
    ) -> None:
        alturas = self._alturas_terreno(posicoes)

        # < Seleciona as partículas ativas que ficaram abaixo da superfície >

        abaixo = ativas & (posicoes[:, 2] < alturas + folga)

        # < Devolve a partícula para cima do terreno e anula a velocidade vertical de descida >

        posicoes[abaixo, 2] = alturas[abaixo] + folga
        descendo = abaixo & (velocidades[:, 2] < 0)
        velocidades[descendo, 2] = 0.0

        # < Reduz em 10% a velocidade horizontal pelo atrito do contato >

        velocidades[abaixo, :2] = velocidades[abaixo, :2] * 0.9

    # < Mantém a massa dentro da janela da malha e evita vazamento do domínio >

    def _contato_limites(
        self,
        posicoes: np.ndarray,
        velocidades: np.ndarray,
        ativas: np.ndarray,
        margem: float = 0.5,
    ) -> None:
        x0, y0, z0, x1, y1, z1 = self._limites()

        for eixo, minimo, maximo in ((0, x0, x1), (1, y0, y1)):

            # < Parede inferior: reposiciona e impede o movimento para fora >

            baixo = ativas & (posicoes[:, eixo] < minimo + margem)
            posicoes[baixo, eixo] = minimo + margem
            velocidades[baixo, eixo] = np.maximum(0.0, velocidades[baixo, eixo])

            # < Parede superior: reposiciona e impede o movimento para fora >

            alto = ativas & (posicoes[:, eixo] > maximo - margem)
            posicoes[alto, eixo] = maximo - margem
            velocidades[alto, eixo] = np.minimum(0.0, velocidades[alto, eixo])

        # < Impede que as partículas fiquem abaixo da cota mínima da malha >

        posicoes[ativas, 2] = np.maximum(posicoes[ativas, 2], z0 + 0.05)

    # < Calcula o raio efetivo de contato derivado do volume da partícula >

    def _raio_contato(self, particula: ParticulaMPM) -> float:
        return max((3.0 * particula.volume / (4.0 * np.pi)) ** (1.0 / 3.0), 1e-6)

    # < Impede sobreposição e aplica impulso normal entre partículas ativas >

    def _resolver_colisoes_particulas(
        self,
        posicoes: np.ndarray,
        velocidades: np.ndarray,
        ativas: np.ndarray,
        restitucao: float = 0.05,
    ) -> int:
        indices_ativos = np.flatnonzero(ativas)
        if len(indices_ativos) < 2:
            return 0

        massas = self.massas
        raios = np.array([self._raio_contato(p) for p in self.particulas])

        '''

        Usa uma árvore de busca para encontrar só os pares de partículas
        próximos o bastante para colidir, em vez de testar todos os pares.
        Os pares são ordenados para que o resultado seja reproduzível

        '''

        arvore = cKDTree(posicoes[indices_ativos])
        pares = arvore.query_pairs(r=2.0 * raios.max(), output_type="ndarray")
        if len(pares) == 0:
            return 0

        pares = pares[np.lexsort((pares[:, 1], pares[:, 0]))]

        colisoes = 0

        for local_a, local_b in pares:
            a = indices_ativos[local_a]
            b = indices_ativos[local_b]

            # < Calcula a distância entre as partículas e a distância de contato >

            delta = posicoes[b] - posicoes[a]
            distancia = float(np.linalg.norm(delta))
            distancia_contato = raios[a] + raios[b]

            if distancia >= distancia_contato:
                continue

            # < Define a normal do contato, tratando partículas na mesma posição >

            if distancia <= 1e-12:
                normal = np.array([1.0, 0.0, 0.0], dtype=float)
                distancia = 1e-12
            else:
                normal = delta / distancia

            # < Separa as partículas, deslocando mais a de menor massa >

            inversa_massa_a = 1.0 / massas[a]
            inversa_massa_b = 1.0 / massas[b]
            soma_inversas = inversa_massa_a + inversa_massa_b
            profundidade = distancia_contato - distancia

            posicoes[a] -= normal * profundidade * inversa_massa_a / soma_inversas
            posicoes[b] += normal * profundidade * inversa_massa_b / soma_inversas

            # < Aplica o impulso normal somente se as partículas estão se aproximando >

            velocidade_relativa = velocidades[b] - velocidades[a]
            componente_normal = float(np.dot(velocidade_relativa, normal))

            if componente_normal < 0.0:
                impulso = -(1.0 + restitucao) * componente_normal / soma_inversas
                vetor_impulso = impulso * normal
                velocidades[a] -= vetor_impulso * inversa_massa_a
                velocidades[b] += vetor_impulso * inversa_massa_b

            colisoes += 1

        return colisoes

    # < Atribui a cada partícula a aceleração da gravidade projetada na encosta >

    def calcular_forcas(self, gravidade: float = 9.81, **_) -> list[np.ndarray]:
        quantidade = len(self.particulas)
        ativas = self._ativas()

        # < Partículas inativas ficam com aceleração zero >

        aceleracoes = np.zeros((quantidade, 3))
        if ativas.any():
            aceleracoes[ativas] = self._aceleracoes(self.posicoes[ativas], gravidade)

        for particula, aceleracao in zip(self.particulas, aceleracoes):
            particula.aceleracao = aceleracao.copy()

        # < Retorna a força de cada partícula, massa vezes aceleração >

        return [p.massa * p.aceleracao.copy() for p in self.particulas]

    # < Agrega massa e momento por célula de grade para diagnóstico numérico >

    def particula_para_grade(self, espacamento: float = 2.0):
        self.grid = {}

        for particula in self.particulas:
            chave = tuple(np.floor(particula.posicao / espacamento).astype(int))
            dados = self.grid.setdefault(chave, {"massa": 0.0, "momento": np.zeros(3)})
            dados["massa"] += particula.massa
            dados["momento"] += particula.massa * particula.velocidade

        return self.grid

    # < Grava nas partículas as posições, velocidades e acelerações calculadas >

    def _gravar_estado(
        self,
        posicoes: np.ndarray,
        velocidades: np.ndarray,
        aceleracoes: np.ndarray,
    ) -> None:
        for indice, particula in enumerate(self.particulas):
            particula.posicao[:] = posicoes[indice]
            particula.velocidade[:] = velocidades[indice]
            particula.aceleracao[:] = aceleracoes[indice]

    # < Integra a dinâmica do sistema para um número de passos temporais >

    def atualizar(
        self,
        dt: float = 0.05,
        passos: int = 1,
        gravidade: float = 9.81,
        coeficiente_resistencia: float = 0.15,
        espacamento: float = 2.0,
        debug: bool = False,
    ) -> list[np.ndarray]:

        # < Valida o passo de tempo e o número de passos >

        if dt <= 0 or passos < 1:
            raise ValueError("dt deve ser positivo e passos deve ser maior que zero.")

        # < Reinicia os históricos antes de uma nova simulação >

        self.historico_posicoes = []
        self.historico_velocidades = []
        self.historico_tempos = []
        self.debug_log = []

        massa_inicial = float(np.sum(self.massas))

        for passo in range(passos):

            # < Calcula a aceleração da gravidade projetada na encosta >

            self.calcular_forcas(gravidade)

            ativas = self._ativas()
            posicoes = self.posicoes
            velocidades = self.velocidades
            aceleracoes = np.array(
                [p.aceleracao for p in self.particulas], dtype=float
            ).reshape(-1, 3)

            '''

            Aplica a resistência proporcional à velocidade e integra o
            movimento: primeiro atualiza a velocidade com a aceleração e
            depois atualiza a posição com a nova velocidade

            '''

            aceleracoes[ativas] -= coeficiente_resistencia * velocidades[ativas]
            velocidades[ativas] += aceleracoes[ativas] * dt
            posicoes[ativas] += velocidades[ativas] * dt

            # < Aplica o contato com o terreno e com os contornos da malha >

            self._contato_terreno(posicoes, velocidades, ativas)
            self._contato_limites(posicoes, velocidades, ativas)

            # < Resolve as colisões entre partículas e reaplica os contatos >

            colisoes = self._resolver_colisoes_particulas(posicoes, velocidades, ativas)
            self._contato_terreno(posicoes, velocidades, ativas)
            self._contato_limites(posicoes, velocidades, ativas)

            # < Grava o novo estado nas partículas >

            self._gravar_estado(posicoes, velocidades, aceleracoes)

            # < Registra o diagnóstico e o histórico do passo >

            estado = self._diagnostico(massa_inicial, colisoes)
            self.debug_log.append(estado)
            self.historico_posicoes.append(self.posicoes.copy())
            self.historico_velocidades.append(self.velocidades.copy())
            self.historico_tempos.append((passo + 1) * dt)

            if debug:
                print(f"passo={passo + 1:03d} v_max={estado['max_speed']:.3f} massa={estado['mass_final']:.3f}")

        return self.historico_posicoes

    # < Calcula métricas resumidas para avaliar estabilidade e vazamento físico >

    def _diagnostico(self, massa_inicial: float, colisoes: int = 0) -> dict[str, float]:
        posicoes = self.posicoes

        # < Calcula o módulo da velocidade e a distância das partículas ao terreno >

        velocidades = np.linalg.norm(self.velocidades, axis=1)
        folgas = posicoes[:, 2] - self._alturas_terreno(posicoes)

        return {
            "max_speed": float(velocidades.max()) if len(velocidades) else 0.0,
            "mean_speed": float(velocidades.mean()) if len(velocidades) else 0.0,
            "max_ground_gap": float(folgas.max()) if len(folgas) else 0.0,
            "mass_initial": massa_inicial,
            "mass_final": float(np.sum(self.massas)),
            "active_particles": float(sum(p.propriedades["ativa"] for p in self.particulas)),
            "particle_collisions": float(colisoes),
        }

    # < Posições 3D de todas as partículas do sistema >

    @property
    def posicoes(self) -> np.ndarray:
        return np.array([p.posicao for p in self.particulas], dtype=float).reshape(-1, 3)

    # < Vetores de velocidade de todas as partículas >

    @property
    def velocidades(self) -> np.ndarray:
        return np.array([p.velocidade for p in self.particulas], dtype=float).reshape(-1, 3)

    # < Massa de cada partícula, usada em diagnósticos e integração >

    @property
    def massas(self) -> np.ndarray:
        return np.asarray([p.massa for p in self.particulas], dtype=float)

    # < Volume de cada partícula, usado em contato e escala de colisões >

    @property
    def volumes(self) -> np.ndarray:
        return np.asarray([p.volume for p in self.particulas], dtype=float)