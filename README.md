# 🚚 Projeto VRP com Multigrafo — Construção e Busca Local

Este projeto apresenta uma implementação para solucionar o **Vehicle Routing Problem (VRP)** adaptado para operar em um **Multigrafo**. Ele utiliza um solver exato (gurobi) e a meta-heurística (ALNS).

## ✨ Funcionalidades Principais

* **Leitura de Dados:** Processamento de instâncias VRP a partir de arquivos `.dat`.
* **Construção de Soluções:** Implementação de **heurísticas** utilizando multigrafos.
* **Otimização:** Uso de técnicas exatadas do **Solver - Gurobipy** e da **Meta-Heurística (ALNS)**.
* **Modos de Execução:** Suporte para execução em **uma instância específica** ou em **lote** para todas as instâncias em um diretório.
* **Exportação:** Salvamento detalhado dos resultados em arquivos.

---

## 📁 Estrutura do Projeto

Abaixo está a organização dos principais arquivos e diretórios:
## 📁 Estrutura do Projeto

/TrabalhoPM

├── /algoritmo
    
    ├── /alns
    
        ├── /operators
        
            ├── workers.py
            
            ├── removals.py
            
            ├── insertions.py
            
        ├── /heuristics
        
            ├── construction.py
            
        ├── /evaluaion
        
            ├── feasibility.py
            
            ├── evaluate.py
            
        ├── alns.py
        
    ├── main_unica_instancia.py
    
    ├── main_todas_instancias.py
    
    ├── solution.py
    
    ├── ler_instancia.py
    
    ├── alwabpData.py
    
├── /resultados_solver

├── /resultados_heuristica

├── /instancias

├── solver_unica_instancia.py

├── solver_todas_instancias.py

├── relatorio_final.csv

├── instancias_upperbound.txt

├── gerar_relatorio.py

├── README.md
---


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
python3 solver_unica_instancia.py
```
O próprio código solicitará o nome do arquivo .dat ou carregará automaticamente a instância configurada internamente.

▶️ 2. Rodar todas as instâncias (resolver por solver)
Percorre toda a pasta /instancias e executa o solver para cada arquivo .dat encontrado.
``` 
python3 solver_todas_instancias.py
```
Os resultados serão salvos no diretório:
/resultados_solver

▶️ 3. Rodar uma única instânca (resulver por metaeurística)
```
python3 algoritmo/main_instancia_unica.py
```

▶️ 4. Rodar todas as instâncas (resulver por metaeurística)
Processa automaticamente todos os .dat.
```
python3 algoritmo?main_todas_instancias.py
```
Os resultados serão salvos no diretório:
/resultados_heuristica

## 🧪 Métodos Implementados
Construção
- Nearest Neighbor

- Savings

- Heurística adaptada para multigrafos

## 📊 Resultados
O programa gera:

- Custo total da solução

- Tempo gasto paa alcançar a solução

- Descrição das rotas

Arquivos .txt para os resultados individualmente e um .csv com um atabela com todos os resultados 
