# TCC Pistol Detection

Projeto academico em Python para comparar YOLOv8 Nano e Faster R-CNN na deteccao de pistolas em imagens de seguranca.

## Tema

**Inteligencia Artificial aplicada a seguranca publico-privada: comparacao entre YOLOv8 Nano e Faster R-CNN para deteccao de pistolas em imagens de seguranca.**

O dataset previsto e o **Pistol_detection Computer Vision Dataset**, disponivel no **Roboflow Universe**, com aproximadamente 2 mil imagens e licenca **Creative Commons Attribution 4.0 International (CC BY 4.0)**.

Por causa da licenca CC BY 4.0, qualquer uso academico, apresentacao, artigo, repositorio ou demonstracao deve atribuir a origem do dataset. A atribuicao deve mencionar, no minimo, o nome do dataset, a plataforma Roboflow Universe, o autor/projeto informado na pagina do dataset quando disponivel, a licenca CC BY 4.0 e o link de acesso usado na data da coleta.

As classes informadas no dataset incluem `pistol`, `Handgun`, `0` e `1`. Nesta primeira etapa, somente `pistol` e `Handgun` sao unificadas automaticamente como `pistol`. As classes `0` e `1` devem ser analisadas manualmente antes de qualquer decisao metodologica.

## Objetivos

- Validar a estrutura do dataset exportado do Roboflow.
- Inspecionar classes, imagens e anotacoes.
- Normalizar classes equivalentes (`pistol` e `Handgun`) para uma unica classe `pistol`.
- Treinar um modelo YOLOv8 Nano com `ultralytics`.
- Preparar a base inicial para Faster R-CNN com PyTorch/torchvision.
- Avaliar futuramente precisao, recall, F1-score, mAP, falsos positivos e tempo medio de inferencia.
- Comparar comportamento em baixa qualidade, baixa iluminacao, oclusao parcial e objetos semelhantes.

Nenhum resultado deve ser inventado. As metricas e conclusoes devem ser registradas apenas depois da execucao real dos testes.

## Instalacao

Ordem recomendada para manter compatibilidade com CUDA 11.8, PyTorch `2.7.1+cu118`, Ultralytics e OpenCV:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install numpy==1.26.4
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip uninstall opencv-python-headless -y
pip install --force-reinstall --no-cache-dir --no-deps numpy==1.26.4 pandas==2.2.3 python-dateutil==2.9.0.post0 Pillow==10.4.0 opencv-python==4.10.0.84 matplotlib==3.8.4 contourpy==1.2.1 kiwisolver==1.4.5
```

Sempre execute esses comandos dentro da `.venv` ativa antes de treinar ou avaliar. `torch`, `torchvision` e `torchaudio` nao ficam no `requirements.txt`, porque devem ser instalados separadamente com o indice CUDA `cu118`. O projeto fixa `numpy==1.26.4`, `pandas==2.2.3`, `python-dateutil==2.9.0.post0`, `Pillow==10.4.0`, `opencv-python==4.10.0.84`, `matplotlib==3.8.4`, `contourpy==1.2.1` e `kiwisolver==1.4.5` para evitar upgrades incompativeis com Python 3.11 e CUDA.

Nao mantenha `opencv-python-headless` junto com `opencv-python` neste ambiente. O pacote `roboflow` pode instalar `opencv-python-headless` como dependencia transitiva, por isso a ordem acima remove o headless depois de instalar o `requirements.txt` e reinstala `opencv-python` explicitamente. O uso de `--no-deps` no ultimo comando impede que `pandas`, `matplotlib` ou `opencv` puxem `numpy>=2` novamente.

Antes de iniciar treinamentos com GPU, confira se a `.venv` esta usando o PyTorch com CUDA:

```powershell
python src/utils/check_cuda.py
```

O resultado esperado deve incluir:

```text
torch.cuda.is_available(): True
CUDA compilado no PyTorch: 11.8
```

Se aparecer `False`, reinstale o PyTorch CUDA dentro da `.venv` ativa:

```powershell
pip uninstall torch torchvision torchaudio -y
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118
python src/utils/check_cuda.py
```

Tambem existe um script auxiliar para Windows que executa essa instalacao com as versoes fixas:

```powershell
.\scripts\install_torch_cuda118.ps1
```

## Configuracao do Roboflow

Este projeto baixa a mesma versao do dataset em dois formatos a partir do Roboflow:

- Workspace: `silvaxn`
- Project: `Pistol_detection`
- Version: `1`
- YOLOv8: `project.version(1).download("yolov8")`
- COCO JSON: `project.version(1).download("coco")`

A API Key nao deve ser escrita no codigo. Crie um arquivo `.env` a partir do exemplo:

```powershell
copy .env.example .env
```

Depois edite `.env` e preencha:

```text
ROBOFLOW_API_KEY=sua_api_key_aqui
ROBOFLOW_WORKSPACE=silvaxn
ROBOFLOW_PROJECT=pistol_detection
ROBOFLOW_VERSION=1
```

O arquivo `.env` fica ignorado pelo Git para evitar vazamento de credenciais.

Importante: `ROBOFLOW_PROJECT` deve usar o **slug** do projeto no Roboflow, nao necessariamente o nome visual exibido na interface. Se o download falhar com erro semelhante a `GET /silvaxn/Pistol_detection does not exist`, verifique a URL do projeto no Roboflow Universe e copie o trecho do projeto. Exemplos comuns sao `pistol_detection` ou `pistol-detection`.

## Organizacao

```text
tcc-pistol-detection/
  config/              Configuracoes de dataset e treinamento
  data/                Datasets exportados do Roboflow e dados processados
  models/              Modelos treinados ou checkpoints locais
  notebooks/           Analise exploratoria
  src/                 Codigo-fonte do projeto
  results/             Metricas, graficos, predicoes e relatorios
  app/                 Interface Streamlit futura
  docs/                Documentacao academica
```

## Preparacao do dataset

Baixe o dataset do Roboflow em dois formatos:

- YOLOv8: coloque em `data/roboflow_yolo/`
- COCO JSON: coloque em `data/roboflow_coco/`

Os scripts detectam automaticamente a raiz real criada pelo Roboflow. O dataset pode ficar diretamente em `data/roboflow_yolo/` e `data/roboflow_coco/`, ou dentro de uma subpasta criada pelo download, por exemplo:

```text
data/roboflow_yolo/pistol_detection-1/
data/roboflow_coco/pistol_detection-1/
```

Para YOLOv8, a raiz real e detectada procurando `data.yaml`. Para COCO, a raiz real e detectada procurando `_annotations.coco.json`, `_annotations.json` ou `annotations.json`. Os scripts exibem no terminal qual pasta foi usada.

Registre no TCC e, se possivel, em `docs/referencias.md`:

- URL exata do dataset no Roboflow Universe.
- Data de acesso.
- Versao/exportacao utilizada.
- Licenca CC BY 4.0.
- Forma de atribuicao exigida pela pagina do dataset.

Estrutura YOLOv8 esperada:

```text
data/roboflow_yolo/
  data.yaml
  train/
    images/
    labels/
  valid/
    images/
    labels/
  test/
    images/
    labels/
```

Estrutura COCO esperada:

```text
data/roboflow_coco/
  train/
    _annotations.coco.json
  valid/
    _annotations.coco.json
  test/
    _annotations.coco.json
```

## Fluxo inicial

Execute primeiro o fluxo de download e verificacao do dataset:

```powershell
py src/dataset/download_roboflow_yolo.py
py src/dataset/download_roboflow_coco.py
py src/dataset/check_dataset_versions.py
py src/dataset/inspect_classes.py
py src/dataset/validate_dataset.py
```

Se precisar baixar novamente e substituir o conteudo existente:

```powershell
py src/dataset/download_roboflow_yolo.py --overwrite
py src/dataset/download_roboflow_coco.py --overwrite
```

Depois da inspecao inicial, prepare o dataset final em `data/processed/`:

```powershell
py src/dataset/analyze_ambiguous_classes.py
py src/dataset/split_yolo_dataset.py
py src/dataset/normalize_classes_yolo.py
py src/dataset/split_coco_dataset.py
py src/dataset/normalize_classes_coco.py
py src/dataset/check_processed_dataset.py
```

Somente depois dessa etapa o treinamento deve ser iniciado:

```powershell
python src/training/train_yolo.py
python src/evaluation/evaluate_yolo.py
python src/evaluation/visualize_yolo_predictions.py
```

Para preparar tambem a base do Faster R-CNN:

```powershell
python src/training/train_faster_rcnn.py
python src/evaluation/evaluate_faster_rcnn.py
python src/evaluation/visualize_faster_rcnn_predictions.py
python src/evaluation/compare_models.py
python src/evaluation/generate_comparison_charts.py
```

Para abrir a interface demonstrativa:

```powershell
streamlit run app/app.py
```

Fluxo completo previsto:

1. Configurar `.env` com `ROBOFLOW_API_KEY`.
2. Baixar YOLOv8 com `src/dataset/download_roboflow_yolo.py`.
3. Baixar COCO com `src/dataset/download_roboflow_coco.py`.
4. Validar estrutura com `src/dataset/check_dataset_versions.py` e `src/dataset/validate_dataset.py`.
5. Inspecionar classes com `src/dataset/inspect_classes.py`, incluindo comparacao YOLOv8 x COCO e exemplos das classes `0` e `1`.
6. Gerar exemplos visuais e relatorio com `src/dataset/analyze_ambiguous_classes.py`.
7. Criar splits YOLO em `data/processed/yolo/` com `src/dataset/split_yolo_dataset.py`.
8. Normalizar classes YOLO com `src/dataset/normalize_classes_yolo.py`.
9. Criar splits COCO em `data/processed/coco/` com `src/dataset/split_coco_dataset.py`, seguindo o manifesto do YOLO.
10. Normalizar classes COCO com `src/dataset/normalize_classes_coco.py`.
11. Validar o dataset final com `src/dataset/check_processed_dataset.py`.
12. Treinar YOLOv8 Nano.
13. Preparar e treinar Faster R-CNN.
14. Rodar inferencias nos dados de teste.
15. Comparar metricas e gerar graficos.

## Scripts principais

- `src/dataset/validate_dataset.py`: valida pastas, `data.yaml`, imagens e labels YOLO.
- `src/dataset/download_roboflow_yolo.py`: baixa a versao 1 do dataset em formato YOLOv8.
- `src/dataset/download_roboflow_coco.py`: baixa a versao 1 do dataset em formato COCO JSON.
- `src/dataset/check_dataset_versions.py`: verifica se YOLOv8 e COCO possuem `train`, `valid` e `test`.
- `src/dataset/inspect_classes.py`: compara classes do `data.yaml` YOLOv8 com categorias COCO, conta imagens/anotacoes por classe e separa exemplos de classes ambiguas.
- `src/dataset/dataset_roots.py`: detecta automaticamente a raiz real dos datasets baixados pelo Roboflow.
- `src/dataset/analyze_ambiguous_classes.py`: gera exemplos visuais das classes `0`, `1`, `Handgun`, `pistol` e `pistols` em `results/predictions/class_examples/` e relatorio em `results/reports/class_analysis.md`.
- `src/dataset/split_yolo_dataset.py`: cria `train`, `valid` e `test` em `data/processed/yolo/` usando 70/20/10 quando o dataset original possui apenas `train`.
- `src/dataset/normalize_classes_yolo.py`: normaliza classes YOLO em `data/processed/yolo/` e gera `data.yaml` com apenas `pistol`.
- `src/dataset/split_coco_dataset.py`: cria `train`, `valid` e `test` em `data/processed/coco/`, preferindo os mesmos nomes de imagem do split YOLO.
- `src/dataset/normalize_classes_coco.py`: normaliza categorias COCO em `data/processed/coco/` para uma categoria final `pistol`.
- `src/dataset/check_processed_dataset.py`: valida o dataset final processado antes do treinamento.
- `src/training/train_yolo.py`: treinamento inicial com YOLOv8 Nano.
- `src/evaluation/evaluate_yolo.py`: avalia o `best.pt` no split `test` e salva metricas em JSON/CSV.
- `src/evaluation/visualize_yolo_predictions.py`: gera imagens com bounding boxes em `results/predictions/yolo/`.
- `src/training/train_faster_rcnn.py`: treina Faster R-CNN com `torchvision` usando o dataset COCO processado.
- `src/evaluation/evaluate_faster_rcnn.py`: avalia `models/faster_rcnn/best.pt` no split `test`.
- `src/evaluation/visualize_faster_rcnn_predictions.py`: gera imagens com bounding boxes em `results/predictions/faster_rcnn/`.
- `src/evaluation/compare_models.py`: compara YOLOv8 Nano e Faster R-CNN em CSV, JSON e relatorio Markdown.
- `src/evaluation/generate_comparison_charts.py`: gera graficos PNG comparativos para o TCC.
- `src/utils/check_cuda.py`: verifica se o PyTorch instalado na `.venv` reconhece CUDA e a GPU.
- `app/app.py`: prototipo Streamlit para demonstracao futura.

## Treinamento YOLOv8 Nano

Com o dataset processado validado, execute:

```powershell
python src/training/train_yolo.py
```

Configuracoes iniciais do treinamento:

- Modelo base: `yolov8n.pt`
- Dataset: `data/processed/yolo/data.yaml`
- Epochs: `50`
- Imagem: `640`
- Batch: `16`
- Patience: `10`
- Device: `0`
- Workers: `4`

O experimento completo do Ultralytics fica em `results/training/yolo/yolov8n_pistol/`. Ao final, o script copia os pesos finais para `models/yolo/best.pt` e `models/yolo/last.pt`. O resumo academico do treinamento e salvo em `results/reports/yolo_training_summary.md`.

## Avaliacao YOLOv8 Nano

Depois do treinamento, execute:

```powershell
python src/evaluation/evaluate_yolo.py
```

O script carrega `models/yolo/best.pt`, avalia o split `test` usando `device=0` e salva:

- `results/metrics/yolo_metrics.json`
- `results/metrics/yolo_metrics.csv`

As metricas incluem precision, recall, F1-score, mAP@50, mAP@50-95, tempo medio de inferencia por imagem e falsos positivos estimados por IoU >= 0.5 contra as labels do test.

## Visualizacao YOLOv8 Nano

Para gerar exemplos visuais de predicao no split `test`, execute:

```powershell
python src/evaluation/visualize_yolo_predictions.py
```

As imagens com bounding boxes sao salvas em `results/predictions/yolo/`.

## Treinamento Faster R-CNN

Depois de validar `data/processed/coco/`, execute:

```powershell
python src/training/train_faster_rcnn.py
```

Configuracoes iniciais:

- Modelo: `torchvision.models.detection.fasterrcnn_resnet50_fpn`
- Classes: `0 = background`, `1 = pistol`
- Epochs: `20`
- Batch size: `4`
- Learning rate: `0.005`
- Momentum: `0.9`
- Weight decay: `0.0005`
- Num workers: `2`
- Device: `cuda`
- Metricas IoU/F1 de validacao: calculadas na primeira epoca, a cada `5` epocas e na ultima epoca

Por padrao, `config/training_config.yaml` define `require_cuda: true` para o Faster R-CNN. Se o PyTorch da `.venv` nao reconhecer CUDA, o script interrompe o treinamento e mostra o comando de reinstalacao correto. Isso evita executar o Faster R-CNN por engano na CPU.

O Faster R-CNN e mais pesado que o YOLOv8 Nano. Mesmo usando CUDA, o uso de GPU pode parecer baixo no Gerenciador de Tarefas quando o gargalo esta no carregamento/conversao das imagens, nas validacoes ou quando o grafico exibido e `3D` em vez de `CUDA`/`Compute`. Para acompanhar melhor, o script mostra barras de progresso por batch durante treino e validacao.

Para interromper e continuar em outro momento, pressione `Ctrl+C` no terminal. O script salva `models/faster_rcnn/last.pt` ao fim de cada epoca e, por padrao, `resume: true` em `config/training_config.yaml` faz o treinamento continuar automaticamente da proxima epoca quando `python src/training/train_faster_rcnn.py` for executado novamente. Se a interrupcao acontecer no meio de uma epoca, o progresso retomado sera a partir da ultima epoca concluida.

Os checkpoints sao salvos em `models/faster_rcnn/`, incluindo `best.pt` e `last.pt`. Os logs ficam em `results/training/faster_rcnn/` e o resumo em `results/reports/faster_rcnn_training_summary.md`.

## Avaliacao Faster R-CNN

Depois do treinamento, execute:

```powershell
python src/evaluation/evaluate_faster_rcnn.py
```

O script avalia `models/faster_rcnn/best.pt` no split `test` e salva:

- `results/metrics/faster_rcnn_metrics.json`
- `results/metrics/faster_rcnn_metrics.csv`

As metricas incluem precision, recall, F1-score, mAP@50, mAP@50-95, tempo medio de inferencia, falsos positivos e quantidade de imagens de teste.

## Visualizacao Faster R-CNN

```powershell
python src/evaluation/visualize_faster_rcnn_predictions.py
```

As imagens com bounding boxes sao salvas em `results/predictions/faster_rcnn/`.

## Comparacao dos Modelos

Com as metricas dos dois modelos geradas, execute:

```powershell
python src/evaluation/compare_models.py
python src/evaluation/generate_comparison_charts.py
```

Saidas geradas:

- `results/metrics/model_comparison.csv`
- `results/metrics/model_comparison.json`
- `results/reports/model_comparison_report.md`
- graficos PNG em `results/charts/`

## Observacao metodologica

As classes `0` e `1` podem representar anotacoes residuais, erros de exportacao, classes semanticas diferentes ou rotulos inconsistentes. Por isso, o projeto nao as converte automaticamente para `pistol`. O script de inspecao gera contagens e exemplos para apoiar uma decisao manual documentada no TCC.

Na normalizacao inicial, somente anotacoes `pistol` e `Handgun` sao mantidas. Imagens sem anotacoes mantidas sao descartadas da versao normalizada para evitar que imagens contendo apenas classes ambiguas sejam tratadas indevidamente como exemplos negativos.

O mapeamento das classes fica configuravel no inicio dos arquivos `normalize_classes_yolo.py` e `normalize_classes_coco.py`. A primeira versao mapeia `Handgun`, `pistol` e `pistols` para `pistol`, mantendo `0` e `1` desativadas ate que o relatorio visual confirme que representam pistolas.

## Reprodutibilidade

Os resultados devem ser reproduzidos a partir dos scripts do projeto, sem edicao manual de metricas. Recomenda-se registrar em `results/reports/`:

- Data da execucao.
- Versao do dataset exportado.
- Parametros de treinamento usados em `config/training_config.yaml`.
- Ambiente de execucao, incluindo CPU/GPU e versoes de Python, PyTorch, torchvision e ultralytics.
- Caminho dos pesos treinados avaliados.
- Arquivos de metricas gerados em `results/metrics/`.

Este repositorio nao deve conter conclusoes quantitativas antes da execucao real dos treinamentos, inferencias e avaliacoes.
