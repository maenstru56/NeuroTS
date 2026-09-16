<h1 align="center">
NeuroTS-Net: Multi-Class Semantic Segmentation of Pediatric Brain Tumors in Multi-Modal MRI
</h1>

<p align="center">
  <a href="README.md">English</a> |
  <a href="README.zh-CN.md">简体中文</a> |
  <a href="README.es.md">Español</a> |
  <a href="README.fr.md">Français</a> |
  <strong>Português</strong> |
  <a href="README.de.md">Deutsch</a> |
  <a href="README.ja.md">日本語</a> |
  <a href="README.ro.md">Română</a>
</p>

Este repositório fornece o código de treinamento, avaliação, inferência e pós-processamento do **NeuroTS-Net**, uma arquitetura tridimensional de segmentação semântica multiclasse desenvolvida para a segmentação de tumores cerebrais pediátricos em MRI multimodal.

A arquitetura NeuroTS-Net foi desenvolvida por Darius Peteleaza. Para dúvidas sobre o artigo ou o código, entre em contato pelo e-mail [darius.peteleaza@ulbsibiu.ro](mailto:darius.peteleaza@ulbsibiu.ro?subject=[GitHub]NeuroTS-Net)

O código acompanha o artigo: **NeuroTS-Net: Multi-Class Semantic Segmentation of Pediatric Brain Tumors in Multi-Modal MRI**

O artigo foi aceito no [BraTS 2026 Challenge: BraTS-PEDs (Tarefa 2)](https://challenges.synapse.org/Challenges/DetailsPage/Overview?id=syn74274097) - MICCAI 2026.

- **Anais da Springer Nature:** aceito; o link será adicionado assim que estiver disponível.
- **Pré-publicação no arXiv:** [https://arxiv.org/abs/2609.16873](https://arxiv.org/abs/2609.16873)
- **MICCAI 2026 - Pôster BraTS-PEDs:** [Ver o pôster](figures/MICCAI-2026_BraTS-PEDs_Poster.png)

> [**Citação.**](#how-to-cite) Por favor, [cite](#how-to-cite) nosso artigo se utilizar o código, a arquitetura ou materiais do artigo associado ao NeuroTS-Net. Este repositório está licenciado sob a [Licença Creative Commons Atribuição 4.0 Internacional](LICENSE).

<p align="center">
  <a href="figures/NeuroTS-Net.png"><img src="figures/NeuroTS-Net.png" alt="Arquitetura NeuroTS-Net" width="32%"></a>
  <a href="figures/NeuroTS_Block.png"><img src="figures/NeuroTS_Block.png" alt="Bloco NeuroTS" width="32%"></a>
  <a href="figures/NeuroTS_Downsampling.png"><img src="figures/NeuroTS_Downsampling.png" alt="Mecanismo de redução de resolução NeuroTS" width="32%"></a>
</p>
<p align="center"><em>Da esquerda para a direita: visão geral da arquitetura NeuroTS-Net, bloco NeuroTS e mecanismo de redução de resolução NeuroTS.</em></p>

## Requisitos

- Sistemas operacionais testados: Windows 11 e Ubuntu 22.04 LTS (o código foi desenvolvido para funcionar em qualquer sistema operacional compatível com Python 3.11, PyTorch e as dependências listadas).
- Python 3.11.
- Uma GPU compatível com CUDA é altamente recomendada para treinamento e inferência. A execução em CPU também é suportada, mas é consideravelmente mais lenta.

Para uma instalação padrão, instale as dependências de execução:

```bash
python -m pip install -r requirements.txt
```

Para uma instalação editável de desenvolvimento, incluindo as dependências de teste, use:

```bash
python -m pip install -e ".[dev]"
```

## Configuração

Este projeto utiliza arquivos de configuração YAML para definir caminhos e divisões dos dados, arquitetura do modelo, função de perda, treinamento, aumento de dados, inferência e configurações de saída.

Para uma descrição detalhada de cada opção, consulte a [referência de configuração](configs/CONFIGURATION.md).

## Conjunto de dados

As configurações padrão destinam-se aos dados BraTS-PED com as seguintes modalidades e classes:

| Canal/rótulo | Significado |
|---|---|
| t1n | MRI ponderada em T1 nativa |
| t1c | MRI ponderada em T1 com contraste |
| t2w | MRI ponderada em T2 |
| t2f | MRI T2-FLAIR |
| 0 | Fundo |
| 1 | Tumor com realce (ET) |
| 2 | Tumor sem realce (NET) |
| 3 | Componente cístico (CC) |
| 4 | Edema (ED) |

A estrutura esperada do conjunto de dados é:

~~~text
data/
|-- BraTS26_PED_training/
|   |-- BraTS-PED-00001-000/
|   |   |-- BraTS-PED-00001-000-t1n.nii.gz
|   |   |-- BraTS-PED-00001-000-t1c.nii.gz
|   |   |-- BraTS-PED-00001-000-t2w.nii.gz
|   |   |-- BraTS-PED-00001-000-t2f.nii.gz
|   |   |-- BraTS-PED-00001-000-seg.nii.gz
|   |-- ...
|-- BraTS26_PED_training_Batch2_Release/
|   |-- ...
|-- BraTS26_PED_validation/
    |-- BraTS-PED-XXXXX-XXX/
    |   |-- BraTS-PED-XXXXX-XXX-t1n.nii.gz
    |   |-- BraTS-PED-XXXXX-XXX-t1c.nii.gz
    |   |-- BraTS-PED-XXXXX-XXX-t2w.nii.gz
    |   |-- BraTS-PED-XXXXX-XXX-t2f.nii.gz
    |-- ...
~~~

Os indivíduos são deduplicados entre as raízes de treinamento pelo ID do caso antes da divisão dos dados.

## Pré-processamento dos dados

O pipeline de pré-processamento:

1. Lê as quatro modalidades NIfTI sem reamostragem;
2. Calcula uma caixa delimitadora de primeiro plano sobre todas as modalidades;
3. Expande a caixa pela margem configurada (32 voxels por padrão);
4. Limita cada modalidade aos percentis de intensidade do primeiro plano;
5. Aplica normalização z-score apenas ao primeiro plano;
6. Armazena as imagens como matrizes float32 juntamente com os rótulos e metadados espaciais.

Crie o cache reutilizável:

~~~bash
python scripts/preprocess_brats26.py --config configs/neurots_brats26_peds_5fold.yaml
~~~

Para reconstruir entradas existentes no cache:

~~~bash
python scripts/preprocess_brats26.py --config configs/neurots_brats26_peds_5fold.yaml --overwrite
~~~

As predições são recolocadas no formato original da imagem e salvas com a affine, o espaçamento, a orientação e os metadados qform e sform da origem.

## Treinamento do modelo

A validação cruzada agrupada e uma única divisão explícita de treinamento/validação são suportadas pelo mesmo ponto de entrada de treinamento.

### Validação cruzada de cinco folds

~~~bash
python scripts/train_crossval.py --config configs/neurots_brats26_peds_5fold.yaml
~~~

O arquivo de folds é gerado a partir dos casos de treinamento em cache quando não existe. Para treinar apenas um fold configurado:

~~~bash
python scripts/train_fold.py --config configs/neurots_brats26_peds_5fold.yaml --fold-index 0
~~~

### Única divisão de treinamento/validação

Crie uma divisão de alto sinal estratificada por rótulo:

~~~bash
python scripts/create_high_signal_split.py --config configs/neurots_brats26_peds_single_split.yaml --output cache/brats26_peds_task2_native_margin32/splits_single_seed42.json --split-name single_seed42 --seed 42 --val-size 40
~~~

Treine usando essa divisão:

~~~bash
python scripts/train_crossval.py --config configs/neurots_brats26_peds_single_split.yaml
~~~

Quando data.split_mode é single, o arquivo de divisão é obrigatório. O treinamento falha com uma mensagem clara em vez de retornar silenciosamente ao modo de folds.

## Amostragem e função objetivo

As configurações públicas reproduzem a estratégia de treinamento sensível a regiões raras utilizada no NeuroTS-Net:

- Amostragem de patches balanceada por componentes para ET, NET, CC e ED;
- Patches negativos difíceis sem ED, centrados em tecido tumoral sem ED;
- Entropia cruzada ponderada e Dice de rótulo por amostra;
- Termos Tversky/Dice para as regiões avaliadas ET, NET, CC, ED, TC e WT;
- Penalidade de probabilidade para classes ausentes ET, CC e ED;
- Supervisão profunda, decaimento cossenoidal da taxa de aprendizado e aquecimento linear.

As estatísticas de solicitação, sucesso, fallback, presença de classes e voxels-alvo da amostragem são gravadas no log de treinamento e em patch_sampling_latest.json.

## Checkpoints e métricas

Cada diretório de treinamento pode conter:

- best_voxel_dice.pt: melhor Dice médio de regiões na validação por patches;
- best_rare_present_score.pt: melhor pontuação de regiões raras presentes em volumes completos;

O avaliador de regiões raras presentes relata separadamente os casos com GT presente, predição presente, TP, FP e FN. Casos vazios/vazios de regiões raras são apenas diagnósticos e não melhoram a seleção do checkpoint por presença rara.

As saídas de treinamento incluem históricos compactos de métricas em CSV/JSON, curvas de aprendizado, relatórios de validação em volume completo, logs e capturas de recursos. A telemetria de seleção de ramificações não é impressa nem exportada intencionalmente.

Valide os arquivos YAML, incluindo a rejeição de chaves duplicadas:

~~~bash
python scripts/check_config_yaml.py configs/neurots_brats26_peds_5fold.yaml configs/neurots_brats26_peds_single_split.yaml
~~~

## Inferência

Pré-processe os dados de validação antes da inferência:

~~~bash
python scripts/preprocess_brats26.py --config configs/neurots_brats26_peds_5fold.yaml --skip-training
~~~

### Checkpoint bruto ou ensemble de folds

Descubra e combine pela média todos os checkpoints de regiões raras presentes em uma execução:

~~~bash
python scripts/predict_validation.py --config configs/neurots_brats26_peds_5fold.yaml --run-dir outputs/neurots_brats26_peds_5fold --candidate rare_present
~~~

Checkpoints explícitos podem ser repetidos para qualquer ensemble:

~~~bash
python scripts/predict_validation.py --config configs/neurots_brats26_peds_single_split.yaml --checkpoint outputs/neurots_brats26_peds_single_split/best_rare_present_score.pt --candidate rare_present
~~~

### Pós-processamento de componentes ET/CC

O pós-processamento público ajustado modifica as probabilidades e os componentes de ET e CC, preservando as predições de ED do modelo principal:

~~~bash
python scripts/predict_postprocessed.py --config configs/neurots_brats26_peds_single_split.yaml --output outputs/neurots_postprocessed --checkpoint outputs/neurots_brats26_peds_single_split/best_rare_present_score.pt --cc-logit-bias 0.75 --cc-prob-thr 0.15 --cc-min-size 20 --cc-max-components 3 --et-logit-bias 0.25 --et-prob-thr 0.30 --et-min-size 20 --et-max-components 4 --protect-tc-wt --zip
~~~

Repita --checkpoint para calcular a média de vários checkpoints. Os arquivos NIfTI são colocados no diretório predictions, e --zip cria um arquivo simples adequado para envio ao desafio.

## Saídas

Os dados gerados são gravados abaixo das raízes configuradas:

~~~text
cache/       matrizes pré-processadas e definições de divisões/folds
outputs/     checkpoints, logs, métricas, predições e arquivos compactados
~~~

Esses diretórios, conjuntos de dados brutos, checkpoints e imagens médicas são excluídos pelo .gitignore.

## Testes

Execute a suíte rápida de testes:

~~~bash
python -m pytest -q
~~~

Execute o teste rápido de propagação direta/inversa do modelo:

~~~bash
python scripts/smoke_test.py
~~~

<a id="how-to-cite"></a>

## Citação

Por favor, cite nosso artigo ao utilizar este repositório:

### Texto simples

```text
Peteleaza, D., Dumitru, R.-G., Neamtu, B., Gellert, A., Sandu, M., & Matei, C. (2026).
NeuroTS-Net: Multi-Class Semantic Segmentation of Pediatric Brain Tumors in Multi-Modal MRI.
arXiv preprint arXiv:2609.16873. https://doi.org/10.48550/arXiv.2609.16873
```

### BibTeX

```bibtex
@misc{peteleaza2026neurotsnet,
  title         = {{NeuroTS-Net}: Multi-Class Semantic Segmentation of Pediatric Brain Tumors in Multi-Modal {MRI}},
  author        = {Peteleaza, Darius and Dumitru, Razvan-Gabriel and Neamtu, Bogdan and Gellert, Arpad and Sandu, Mariana and Matei, Claudiu},
  year          = {2026},
  eprint        = {2609.16873},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  doi           = {10.48550/arXiv.2609.16873},
  url           = {https://arxiv.org/abs/2609.16873}
}
```
