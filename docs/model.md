# RNN em PyTorch

O modelo recebe janelas de 50 ciclos com as condicoes operacionais, sensores e
`cycle_norm`. A ultima saida de uma RNN `tanh` alimenta uma camada linear que
produz o logit de `failure_within_w1`.

As matrizes NumPy de entrada e os rotulos sao convertidos explicitamente com
`torch.tensor(..., dtype=torch.float32)`, agrupados em `TensorDataset` e
consumidos por `DataLoader`. Antes do treino, o programa mostra as dimensoes de
todos os tensores, do primeiro batch e o dispositivo CPU/CUDA selecionado.

O exemplo foi escrito como um fluxo linear e didatico. Ele usa
`BCEWithLogitsLoss`, o otimizador Adam e mostra a loss de cada epoca. Ao final,
calcula uma acuracia simples no conjunto de teste.

## Execucao

```powershell
python -m pip install -r requirements.txt
python src/main.py --epochs 20 --batch-size 256
```

Ao final, o programa exibe a matriz de confusao e precision, recall e F1 no
conjunto de teste.
