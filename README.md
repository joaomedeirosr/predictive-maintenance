<div align="center">

<p align="center">
  <img src="img/logo-w.png" alt="Logo" width="250">
</p>

# Manutenção Preditiva de Aeronaves com Redes RNN e LSTM


[Descrição inicial do problema](#descricao)
•
[Objetivo geral](#objetivo)
•
[Estrutura inicial](#estrutura)
•
[Como usar?](#documentation)

</div>

## 📑 Menu

- [Manutenção Preditiva de Aeronaves com Redes RNN e LSTM](#manutenção-preditiva-de-aeronaves-com-redes-rnn-e-lstm)
  - [📑 Menu](#-menu)
  - [📝 Descrição Inicial do Problema](#-descrição-inicial-do-problema)
  - [🎯 Objetivo geral](#-objetivo-geral)
  - [✈️ Overview](#-Entendimento)  
  - [🏗️ Estrutura inicial do projeto](#️-estrutura-inicial-do-projeto)
  - [📆 Etapa atual do desenvolvimento](#-etapa-atual-do-desenvolvimento)
    - [💾 1. Carregamento e limpeza dos dados](#-1-carregamento-e-limpeza-dos-dados)
    - [🧹 2. Limpeza dos dados](#-2-limpeza-dos-dados)
    - [🔢 3. Implementação da RUL](#-3-implementacao-da_rul)
    - [🎯 4. Criação da variável alvo](#-4-criacao_variavel_alvo)
    - [⚖️ 5. Normalização dos dados](#-5-normalizacao-dos-dados)
  - [📚 Dependencies and Libs](#-dependencies-and-libs)
  - [❗ Requirements](#-requirements)
  - [✅ Status do projeto](#-status-do-projeto)
  - [🥇 License](#-license)
  - [👨‍💻 Time](#-time)
  - [🔍 Referências](#-referências)

<a id="descricao"></a>

## 📝 Descrição Inicial do Problema

A manutenção preditiva é uma abordagem que busca antecipar falhas em equipamentos antes que elas ocorram, utilizando dados históricos e informações coletadas por sensores. No contexto aeronáutico, essa prática é fundamental para garantir a segurança operacional e reduzir custos de manutenção.
Os motores de aeronaves estão sujeitos a desgaste ao longo do tempo devido às condições de operação. Falhas não detectadas podem ocasionar custos elevados de reparo, substituição de componentes e até riscos à segurança dos passageiros.
Por outro lado, realizar manutenções preventivas em excesso também gera custos desnecessários. Dessa forma, torna-se importante desenvolver mecanismos capazes de identificar, com antecedência, quais motores apresentam maior probabilidade de falha.
Este projeto utiliza dados históricos de operação e leituras de sensores para prever se um motor de aeronave apresenta risco de falha dentro de um determinado número de ciclos operacionais.

<a id="objetivo"></a>

## 🎯 Objetivo geral
Desenvolver um modelo preditivo baseado em Redes Neurais Recorrentes (RNN) e Long Short-Term Memory (LSTM), capaz de identificar motores de aeronaves com risco de falha a partir de séries temporais contendo dados operacionais e medições de sensores.

<a id="Entendimento"></a>

## ✈️ Overview
O projeto utiliza o conjunto de dados NASA C-MAPSS (FD001) para prever se um motor irá falhar dentro de uma janela de 30 ciclos.

### 📊 Estrutura do Dataset NASA C-MAPSS

| **Coluna** | **Descrição** | **Utilização** |
|------------|---------------|----------------|
| **id** | Identificador do motor | Usado para identificar cada motor e calcular sua vida útil. Não entra como atributo de entrada. |
| **cycle** | Ciclo de operação | Representa o desgaste ao longo do tempo e é usado para calcular a RUL. |
| **setting1, setting2, setting3** | Condições operacionais | Variáveis preditoras que representam diferentes condições de operação. |
| **sensor1 a sensor21** | Leituras dos sensores | Variáveis contínuas que refletem o estado operacional do motor. |
| **RUL** | Vida útil restante | Calculada no pré-processamento. Indica quantos ciclos restam até a falha. |
| **failure_within_w1** | Variável alvo (0 ou 1) | Indica se o motor falhará nos próximos 30 ciclos (1) ou não (0). |

### 🔳 Sensores
Os sensor1 a sensor21 monitoram aspectos como temperatura, pressão, fluxo e velocidade. Embora a NASA não forneça o significado físico de cada sensor, para que os pesquisadores foquem no desenvolvimento dos algoritmos de prognóstico e não em um motor aeronáutico específico,  eles refletem o estado operacional do motor e permitem identificar padrões de degradação.
Durante a análise exploratória, sensores com pouca variação podem ser descartados, pois não contribuem para o aprendizado do modelo.

### 🔢 Remaining Useful Life (RUL)
A RUL indica quantos ciclos restam antes da falha do motor.
Exemplo: Motor 10
 * Maior ciclo registrado: 192
 * Observando o ciclo 50:

    𝑅𝑈𝐿 = 192 − 50 = 142

Ou seja, o motor ainda possui aproximadamente 142 ciclos antes da falha.

### 🎯 Criação da Variável Alvo
Definimos uma janela de previsão de 30 ciclos:
 * Se RUL ≤ 30 → failure_within_w1 = 1
 * Caso contrário → failure_within_w1 = 0
Essa janela garante que o modelo alerte sobre falhas com antecedência suficiente para programar a manutenção.

<a id="estrutura"></a>

## 🏗️ Estrutura inicial do projeto

```text
predictive-maintenance/
├── data/
│   ├── PM_train.txt
│   ├── PM_test.txt
│   └── PM_truth.txt
│
├── src/
    ├── constants.py
│   ├── data_loader.py
    ├── feature_engineering.py
│   ├── preprocess.py
│   └── main.py
│
├── requirements.txt
└── README.md
```

---

<a id="etapa-atual-do-desenvolvimento"></a>

## 📆 Etapas do desenvolvimento

* Nesta primeira etapa foi implementada a estrutura inicial do projeto, contemplando:

<a id="load"></a>

### 💾 1. Carregamento e limpeza dos dados

O módulo `data_loader.py` é responsável por carregar os conjuntos de dados utilizados no projeto:

* **PM_train.txt**: histórico completo dos motores até a ocorrência da falha;
* **PM_test.txt**: histórico parcial dos motores, utilizado para avaliação do modelo;
* **PM_truth.txt**: valores reais de vida útil remanescente dos motores do conjunto de teste.

Exemplo de carregamento:

```python
train_df = loader.load_train_data()
test_df = loader.load_test_data()
truth_df = loader.load_truth_data()
```

### 🧹 2. Limpeza dos dados
O módulo `preprocess.py` contém a classe `DataCleaner`, responsável pela remoção de colunas vazias presentes nos arquivos originais.

```python
train_df = cleaner.remove_empty_columns(train_df)
test_df = cleaner.remove_empty_columns(test_df)
truth_df = cleaner.remove_empty_columns(truth_df)
```

Essa etapa garante que apenas colunas com informações relevantes sejam utilizadas nas próximas fases do projeto.

---
* Nesta segunda etapa foi implementada a biblioteca numpy, contemplando:

### 🔢 3. Implementação da RUL

### 🎯 4. Criação da variável alvo

### ⚖️ 5. Normalização dos dados


<a id="documentation"></a>

## 📚 Dependencies and Libs



<a id="requirements"></a>

## ❗ Requirements


<a id="dependencies"></a>

## ✅ Status do projeto

🚧 Em desenvolvimento – Etapa 1 concluída: Estrutura inicial, carregamento e limpeza dos dados.
                      – Etapa 2 concluída: Normalização através da biblioteca numpy.


<a id="license"></a>

## 🥇 License

The [MIT License]() (MIT)

<a id="time"></a>

 ## 👨‍💻 Time

Esse projeto é mantido por:

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/joaomedeirosr">
        <img src="https://github.com/joaomedeirosr.png" width="100px;" alt="João Victor Rocha"/>
        <br />
        <sub><b>João Victor Rocha</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/renattabatista">
        <img src="https://github.com/renattabatista.png" width="100px;" alt="Rafaella"/>
        <br />
        <sub><b>Rafaella Batista</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/izabella-araujo">
        <img src="https://github.com/izabella-araujo.png" width="100px;" alt="Membro 3"/>
        <br />
        <sub><b>Izabella Araujo</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/oxschellen">
        <img src="https://github.com/oxschellen.png" width="100px;" alt="Membro 4"/>
        <br />
        <sub><b>Carlos Schellenberger</b></sub>
      </a>
    </td>
  </tr>
</table>  

## 🔍 Referências
https://www.kaggle.com/code/sharanharsoor/aircraft-predictive-maintenance/notebook


