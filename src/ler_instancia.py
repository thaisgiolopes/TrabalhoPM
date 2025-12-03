def ler_instancia(caminho_arquivo):
    """
    Formato:
    n
    t_{11} t_{12} ... t_{1k}
    ...
    t_{n1} ... t_{nk}
    i j
    ...
    -1 -1
    Retorna: (n, tempos, precedencias)
    """
    with open(caminho_arquivo, "r") as f:
        linhas = [linha.strip() for linha in f if linha.strip()]

    n = int(linhas[0])
    tempos = []
    idx = 1
    for _ in range(n):
        valores = []
        for v in linhas[idx].split():
            if v.lower() == "inf":
                valores.append(float("inf"))
            else:
                valores.append(float(v))
        tempos.append(valores)
        idx += 1

    precedencias = []
    while idx < len(linhas):
        i, j = linhas[idx].split()
        i, j = int(i), int(j)
        if i == -1 and j == -1:
            break
        precedencias.append((i, j))
        idx += 1

    return n, tempos, precedencias
