from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import requests


class ClimaNASA:

    """

    A classe ClimaNASA tem como objetivo coletar e estruturar dados horários
    de precipitação e temperatura pela API NASA POWER. Primeiro, ela recebe a
    posição geográfica e o período de análise. Depois, o método download busca
    os dados, trata os valores ausentes e organiza tudo em um DataFrame com
    data e hora no fuso de São Paulo. Por fim, converte as séries temporais em
    arrays numéricos e padroniza a intensidade da chuva para o sistema
    internacional, preparando os dados de entrada para o modelo hidrológico.

    """

    # < Valor que a API NASA POWER usa para indicar dado ausente >

    VALOR_AUSENTE = -999.0

    '''

    Inicializa o coletor armazenando a posição geográfica e o período
    de análise, e deixa vazios os resultados que serão preenchidos
    pelo download.

    '''

    def __init__(
        self,

        # < Latitude do ponto de coleta em graus >

        latitude: float,

        # < Longitude do ponto de coleta em graus >

        longitude: float,

        # < Data inicial no formato AAAAMMDD >

        inicio: str,

        # < Data final no formato AAAAMMDD >

        fim: str,
    ):
        self.latitude = latitude
        self.longitude = longitude
        self.inicio = inicio
        self.fim = fim

        # < Inicializa os resultados que serão preenchidos pelo download >

        self.dataframe: Optional[pd.DataFrame] = None
        self.chuva = None
        self.temperatura = None
        self.chuva_si = None
        self.acumulado = None

    # < Baixa os dados horários da API NASA POWER e os estrutura >

    def download(self) -> pd.DataFrame:

        # < Monta o endereço da requisição com os parâmetros da coleta >

        url = (
            "https://power.larc.nasa.gov/api/temporal/hourly/point"
            "?parameters=T2M,PRECTOTCORR"
            f"&community=AG"
            f"&longitude={self.longitude}"
            f"&latitude={self.latitude}"
            f"&start={self.inicio}"
            f"&end={self.fim}"
            "&format=JSON"
            "&time-standard=UTC"
        )

        # < Faz a requisição e interrompe caso a API retorne erro >

        retorna = requests.get(url, timeout=60)
        retorna.raise_for_status()

        # < Extrai da resposta a parte com as séries de dados >

        dados = retorna.json()["properties"]["parameter"]

        # < Estrutura os dados em um DataFrame >

        dataframe = pd.DataFrame(
            {
                "Temperatura_C": dados["T2M"],
                "Chuva_mm_hora": dados["PRECTOTCORR"],
            }
        )

        '''

        Troca o valor -999 (dado ausente da NASA) por NaN. A chuva ausente
        é tratada como zero e a temperatura ausente é interpolada, para que
        valores sem sentido físico não entrem no modelo

        '''

        dataframe = dataframe.replace(self.VALOR_AUSENTE, np.nan)
        dataframe["Chuva_mm_hora"] = dataframe["Chuva_mm_hora"].fillna(0.0)
        dataframe["Temperatura_C"] = dataframe["Temperatura_C"].interpolate(
            limit_direction="both"
        )

        '''

        Converte o índice para data e hora, informa que está em UTC e
        converte para o fuso de São Paulo

        '''

        dataframe.index = pd.to_datetime(dataframe.index, format="%Y%m%d%H")
        dataframe.index = (
            dataframe.index.tz_localize("UTC").tz_convert("America/Sao_Paulo")
        )

        # < Transforma o índice em uma coluna chamada Data_Hora >

        dataframe.index.name = "Data_Hora"
        dataframe = dataframe.reset_index()

        # < Armazena o DataFrame e as séries em arrays numéricos >

        self.dataframe = dataframe
        self.chuva = dataframe["Chuva_mm_hora"].to_numpy(dtype=float)
        self.temperatura = dataframe["Temperatura_C"].to_numpy(dtype=float)

        # < Converte a chuva de milímetros por hora para metros por segundo >

        self.chuva_si = self.chuva / 1000.0 / 3600.0

        # < Calcula a chuva acumulada ao longo do período >

        self.acumulado = np.cumsum(self.chuva)

        # < Retorna o DataFrame com os dados >

        return dataframe