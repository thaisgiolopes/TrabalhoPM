# 👷‍♂️ Projeto ALWABP

Este projeto apresenta uma implementação para solucionar o **Assembly Line Worker Assignment and Balancing Problem (ALWABP)**. Ele utiliza um solver exato **(Gurobi)** e a meta-heurística **Adaptive Large Neighborhood Search (ALNS)**.

## ✨ Funcionalidades Principais

* **Leitura de Dados:** Processamento de instâncias ALWABP a partir de arquivos `.txt`.
* **Construção de Soluções:** Implementação de **heurísticas** utilizando grafos.
* **Otimização:** Uso de técnicas exatadas do **Solver - Gurobipy** e da **Meta-Heurística (ALNS)**.
* **Modos de Execução:** Suporte para execução em **uma instância específica** ou em **lote** para todas as instâncias em um diretório.
* **Exportação:** Salvamento detalhado dos resultados em arquivos.

---

## 📁 Estrutura do Projeto

Abaixo está a organização dos principais arquivos e diretórios:

```
/TrabalhoPM
├── /algoritmo
|    ├── /alns
|    |    ├── /operators
|    |    |    ├── workers.py
|    |    |    ├── removals.py
|    |    |    ├── insertions.py
|    |    ├── /heuristics
|    |    |    ├── construction.py
|    |    ├── /evaluation
|    |    |    ├── feasibility.py
|    |    |    ├── evaluate.py
|    |    ├── alns.py
|    ├── main_unica_instancia.py
|    ├── main_todas_instancias.py
|    ├── solution.py
|    ├── ler_instancia.py
|    ├── alwabpData.py
├── /resultados_solver
├── /resultados_heuristica
├── /instancias
├── solver_unica_instancia.py
├── solver_todas_instancias.py
├── relatorio_final.csv
├── instancias_upperbound.txt
├── gerar_relatorio.py
├── README.md
```

## 🚀 Como Executar

📦 Requisitos
* **Python:** Versão 3.8 ou superior.
* **Gurobi:** `gurobipy` e uma licença de uso instalada na máquina (necessário apenas para os modos de execução do **Solver**).

📚 O projeto possui **quatro modos de execução**, permitindo rodar uma única instância ou todas as instâncias automaticamente, tanto para o solver, quanto para a meta-heurística.

Antes de tudo, certifique-se de que você está dentro da pasta raiz do projeto:
```
cd TrabalhoPM
```

E execute com Python 3:

▶️ 1. Rodar uma única instância (resolver por solver)
Executa o solver completo para uma instância específica.
```
python3 solver_unica_instancia.py <caminhodo_da_instancia> <caminho_para_salvar_resultado>
```

▶️ 2. Rodar todas as instâncias (resolver por solver)
Percorre toda a pasta /instancias e executa o solver para cada arquivo .txt encontrado.
``` 
python3 solver_todas_instancias.py
```
Os resultados serão salvos no diretório:
\resultados_solver

▶️ 3. Rodar uma única instânca (resulver por metaeurística)
```
python3 algoritmo/main_instancia_unica.py <caminhodo_da_instancia> <caminho_para_salvar_resultado>
```

▶️ 4. Rodar todas as instâncas (resulver por metaeurística)
Percorre toda a pasta /instancias e executa o solver para cada arquivo .txt encontrado.
```
python3 algoritmo/main_todas_instancias.py 
```
Os resultados serão salvos no diretório:
\resultados_heuristica

## 📊 Resultados
O programa gera:

- Custo total da solução

- Tempo gasto paa alcançar a solução

- Descrição das rotas

Arquivos .txt para os resultados individualmente e um .csv com uma tabela com todos os resultados 
