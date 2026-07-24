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
  - [🏗️ Estrutura inicial do projeto](#️-estrutura-inicial-do-projeto)
  - [📆 Etapa atual do desenvolvimento](#-etapa-atual-do-desenvolvimento)
    - [💾 1. Carregamento e limpeza dos dados](#-1-carregamento-e-limpeza-dos-dados)
    - [🧹 2. Limpeza dos dados](#-2-limpeza-dos-dados)
  - [🔢⚖️ Implementação com NumPy (Entrega 2)](#-implementação-com-numpy-entrega-2)
    - [🔢 3. Implementação da RUL](#-3-implementação-da-rul)
    - [🎯 4. Criação da variável alvo](#-4-criação-da-variável-alvo)
    - [⚖️ 5. Normalização dos dados](#-5-normalização-dos-dados)
  - [🔥 Implementação em PyTorch (Entrega 3)](#-implementação-em-pytorch-entrega-3)
    - [📦 1. Carregamento dos dados em tensores](#-1-carregamento-dos-dados-em-tensores)
    - [🧮 2. Processamento dos dados (RUL e normalização)](#-2-processamento-dos-dados-rul-e-normalização)
    - [🗂️ 3. TensorDataset e DataLoader](#-3-tensordataset-e-dataloader)
    - [🧠 4. Modelo em PyTorch](#-4-modelo-em-pytorch)
    - [🎯 5. Treinamento, avaliação e salvamento](#-5-treinamento-avaliação-e-salvamento)
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
│   ├── data_loader.py
│   ├── preprocess.py
│   └── main.py
│
├── requirements.txt
└── README.md
```

---

<a id="etapa-atual-do-desenvolvimento"></a>

## 📆 Etapa atual do desenvolvimento

Nesta primeira etapa foi implementada a estrutura inicial do projeto, contemplando:

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

<a id="numpy"></a>

## 🔢⚖️ Implementação com NumPy (Entrega 2)

Nesta etapa foi implementado o uso da biblioteca NumPy para calcular o RUL, criar a variável alvo de classificação e normalizar os dados antes da etapa de treinamento. O fluxo completo pode ser executado com `python src/main.py`.

<a id="rul-numpy"></a>

### 🔢 3. Implementação da RUL

O módulo `src/feature_engineering.py` (classe `FeatureEngineering`) calcula o RUL usando NumPy:

* `rul_with_numpy`: agrupa o treino por motor (`id`), obtém o ciclo máximo com `np.max` e calcula `RUL = ciclo_máximo - ciclo_atual`;
* `rul_test`: calcula o RUL do teste combinando o ciclo máximo observado de cada motor com o valor real vindo do `pm_truth.txt`.

```python
train_df = FeatureEngineering.rul_with_numpy(train_df)
test_df = FeatureEngineering.rul_test(test_df, truth_df)
```

<a id="target-variable"></a>

### 🎯 4. Criação da variável alvo

A partir do RUL, `generating_target_variable` cria a coluna binária `failure_within_w1` usando `np.where`: o valor é `1` quando o motor vai falhar dentro de uma janela de `w1` ciclos (padrão: 30) e `0` caso contrário.

```python
train_df = FeatureEngineering.generating_target_variable(train_df)
```

<a id="normalizacao-numpy"></a>

### ⚖️ 5. Normalização dos dados

A classe `Normalize` (em `src/preprocess.py`) normaliza as colunas de sensores/configurações com min-max usando NumPy (`np.min`, `np.max`), calculando os limites **apenas com o treino** e reaplicando os mesmos valores no teste, evitando vazamento de dados:

```python
train_df, data_min, data_max = Normalize.normalize_with_numpy(train_df)
test_df = Normalize.normalize_test(test_df, data_min, data_max)
```

---

<a id="pytorch"></a>

## 🔥 Implementação em PyTorch (Entrega 3)

Nesta etapa foi implementado o pipeline completo em PyTorch, responsável por carregar, processar, treinar e salvar um modelo de previsão de RUL (*Remaining Useful Life* — vida útil restante) dos motores.

<a id="tensor-load"></a>

### 📦 1. Carregamento dos dados em tensores

O módulo `src/torch_data.py` contém a classe `TensorLoader`, responsável por ler os arquivos `.txt` diretamente como tensores do PyTorch (sem depender de pandas ou NumPy):

```python
loader = TensorLoader("data")
train, test, truth = loader.load_all()
```

<a id="rul-norm"></a>

### 🧮 2. Processamento dos dados (RUL e normalização)

Ainda em `src/torch_data.py`:

* `compute_train_rul`: calcula o RUL de cada linha do treino (`RUL = ciclo_máximo_do_motor - ciclo_atual`);
* `compute_test_rul`: calcula o RUL do teste a partir do `pm_truth.txt`;
* `normalize_sensors`: normaliza as colunas de sensores (min-max), ajustando a escala apenas com os dados de treino.

<a id="tensordataset"></a>

### 🗂️ 3. TensorDataset e DataLoader

A função `build_dataset` empacota as features (configs operacionais + sensores) e o RUL em um `torch.utils.data.TensorDataset`, que é consumido em lotes (*batches*) através de um `torch.utils.data.DataLoader`.

<a id="model"></a>

### 🧠 4. Modelo em PyTorch

O módulo `src/model.py` define `RULModel`, uma rede neural feedforward simples (`Linear → ReLU → Linear`) que recebe as colunas de configs/sensores de uma linha e prevê o RUL correspondente.

<a id="training"></a>

### 🎯 5. Treinamento, avaliação e salvamento

O módulo `src/train.py` executa o loop de treinamento:

* calcula o erro (MAE — Mean Absolute Error) em treino e teste a cada época e imprime no terminal;
* usa o otimizador `Adam` para ajustar os pesos do modelo;
* ao final, salva os pesos treinados em `models/rul_model.pt`.

Para rodar o treinamento:

```bash
cd src
uv run python train.py
```

---

<a id="documentation"></a>

## 📚 Dependencies and Libs



<a id="requirements"></a>

## ❗ Requirements


<a id="dependencies"></a>

## ✅ Status do projeto

✅ Entrega 1 concluída: Estrutura inicial, carregamento e limpeza dos dados.

✅ Entrega 2 concluída: implementação da RUL, variável alvo e normalização utilizando NumPy.

✅ Entrega 3 concluída: implementação em PyTorch (carregamento em tensores, processamento/RUL, treinamento com impressão do erro de treino/teste, e salvamento do modelo).


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
        <img src="https://github.com/renattabatista.png" width="100px;" alt="Rafaella Batista"/>
        <br />
        <sub><b>Rafaella Batista</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/izabella-araujo">
        <img src="https://github.com/izabella-araujo.png" width="100px;" alt="Izabella Araujo"/>
        <br />
        <sub><b>Izabella Araujo</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/oxschellen">
        <img src="https://github.com/oxschellen.png" width="100px;" alt="Carlos Schellenberger"/>
        <br />
        <sub><b>Carlos Schellenberger</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/larisse13">
        <img src="https://github.com/larisse13.png" width="100px;" alt="Larisse Carvalho"/>
        <br />
        <sub><b>Larisse Carvalho</b></sub>
      </a>
    </td>
  </tr>
</table>  

## 🔍 Referências
https://www.kaggle.com/code/sharanharsoor/aircraft-predictive-maintenance/notebook


