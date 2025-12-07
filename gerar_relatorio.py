import os
import csv
import re


def extrair_chave_solver(nome):
    """
    Extrai (numero, tipo) de nomes como: 12_hes_resultado.txt
    Retorna uma tupla (numero, tipo)
    """
    padrao = r"(\d+)_(hes|wee|ton|ros)_resultado"
    m = re.search(padrao, nome)
    if m:
        return m.group(1), m.group(2)
    return None


def extrair_chave_heuristica(nome):
    """
    Extrai (numero, tipo) de nomes como: saida_12_hes.txt
    """
    padrao = r"saida_(\d+)_(hes|wee|ton|ros)"
    m = re.search(padrao, nome)
    if m:
        return m.group(1), m.group(2)
    return None

def extrair_float(texto):
    import re
    m = re.search(r"[-+]?\d*\.?\d+", texto)
    if m:
        return float(m.group(0))
    raise ValueError(f"Não foi possível extrair número de: {texto}")


def ler_solver(caminho):
    with open(caminho, "r") as f:
        for linha in f:

            # Tempo de ciclo
            if "Tempo de ciclo" in linha:
                m = re.search(r"Tempo de ciclo C:\s*([\d.]+)", linha)
                if m:
                    ciclo = float(m.group(1))
                    continue

            # Tempo de execução → assim que encontrar, retorna e PARA
            if "Tempo de execução" in linha:
                m = re.search(r"([\d]+(?:\.\d+)?)", linha)
                if m:
                    tempo = float(m.group(1))
                else:
                    tempo = None

                return ciclo, tempo

    # Caso algo dê errado
    return None, None


def ler_heuristica(path):
    with open(path, "r") as f:
        linhas = [l.strip() for l in f if l.strip()]

    melhor_sf = float("inf")
    melhor_si = None
    melhor_tempo = None
    i = 0

    while i < len(linhas):

        # SEED
        seed = linhas[i]

        # CASO: semente não conseguiu achar solução
        if "não conseguiu achar" in linhas[i+1]:
            # Pular bloco até o próximo seed
            i += 2
            while i < len(linhas) and "Estacao" in linhas[i]:
                i += 1
            continue

        # Caso normal
        si = extrair_float(linhas[i+1])
        sf = extrair_float(linhas[i+2])
        tempo = extrair_float(linhas[i+3])

        # CRITÉRIO DE ESCOLHA:
        # 1. SF menor sempre vence
        # 2. Se SF for igual, tempo menor vence
        if (sf < melhor_sf) or (sf == melhor_sf and tempo < melhor_tempo):
            melhor_sf = sf
            melhor_si = si
            melhor_tempo = tempo

        # pular bloco de estações
        i += 4
        while i < len(linhas) and "Estacao" in linhas[i]:
            i += 1

    return melhor_si, melhor_sf, melhor_tempo

def carregar_upperbounds(caminho):
    upper = {}
    with open(caminho, "r") as f:
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            chave, valor = linha.split(",")
            chave = chave.strip()
            valor = int(valor.strip())
            upper[chave] = valor
    return upper


def extrair_chave_instancia(nome_arquivo):
    m = re.search(r"(\d+)_(hes|ros|wee|ton)", nome_arquivo)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    return None


def gerar_relatorio(pasta_solver, pasta_heuristica, output_csv, upper):

    arquivos_solver = os.listdir(pasta_solver)
    arquivos_heur = os.listdir(pasta_heuristica)

    # indexar solver
    dict_solver = {}
    for arq in arquivos_solver:
        chave = extrair_chave_solver(arq)
        if chave:
            dict_solver[chave] = arq

    # indexar heuristica
    dict_heur = {}
    for arq in arquivos_heur:
        chave = extrair_chave_heuristica(arq)
        if chave:
            dict_heur[chave] = arq

    resultados = []

    # cruzar dados
    for chave_solver in dict_solver:

        if chave_solver not in dict_heur:
            print(f"[AVISO] Não existe arquivo da heurística para {chave_solver}")
            continue

        arq_solver = os.path.join(pasta_solver, dict_solver[chave_solver])
        arq_heur = os.path.join(pasta_heuristica, dict_heur[chave_solver])

        # -----------------------------------------
        # Extrair chave da instância: ex. "41_hes"
        # -----------------------------------------
        chave_inst = extrair_chave_instancia(arq_solver)

        if not chave_inst:
            print(f"[ERRO] Não foi possível extrair chave para {arq_solver}")
            continue

        # -----------------------------------------
        # Buscar valor ótimo via upperbound
        # -----------------------------------------
        if chave_inst not in upper:
            print(f"[AVISO] Chave não encontrada no upperbound para {arq_solver}")
            sol_otima = None
        else:
            sol_otima = upper[chave_inst]

        # -----------------------------------------
        # Ler somente o tempo do solver
        # -----------------------------------------
        _, tempo_solver = ler_solver(arq_solver)

        # -----------------------------------------
        # Ler heurística (SI, SF, tempo)
        # -----------------------------------------
        si, sf, tempo_heur = ler_heuristica(arq_heur)

        # Heurística pode falhar -> "ALNS não conseguiu achar solução válida"
        if sf is None or si is None:
            # Guarda valores nulos, sem cálculo de desvios
            resultados.append([
                chave_inst,
                si,
                sf,
                None,
                None,
                tempo_heur,
                tempo_solver
            ])
            continue

        # -----------------------------------------
        # Cálculo dos desvios
        # -----------------------------------------
        desvio_si_sf = 100 * (si - sf) / si if si != 0 else 0
        desvio_sf_opt = 100 * (sf - sol_otima) / sol_otima if sol_otima else None

        resultados.append([
            chave_inst,
            si,
            sf,
            round(desvio_si_sf, 4),
            round(desvio_sf_opt, 4) if desvio_sf_opt is not None else None,
            tempo_heur,
            tempo_solver
        ])

    # -----------------------------------------
    # Ordenação alfabética das instâncias
    # -----------------------------------------
    resultados.sort(key=lambda x: str(x[0]).lower())

    # -----------------------------------------
    # Escrita do CSV final
    # -----------------------------------------
    with open(output_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "Instancia", "SI", "SF",
            "Desvio_SI_SF", "Desvio_SF_Opt",
            "Tempo_Heuristica", "Tempo_Solver"
        ])
        w.writerows(resultados)

    print(f"Relatório gerado em: {output_csv}")


upper = carregar_upperbounds("instancias_upperbound.txt")

gerar_relatorio(
    "resultados_solver",
    "resultados_heuristica",
    "relatorio_final_mesmo.csv",
    upper
)