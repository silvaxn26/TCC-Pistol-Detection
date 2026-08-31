# Metodologia

Este projeto utiliza um dataset publico do Roboflow Universe, denominado **Pistol_detection Computer Vision Dataset**, licenciado como **Creative Commons Attribution 4.0 International (CC BY 4.0)**. O conjunto possui aproximadamente 2 mil imagens destinadas a tarefas de deteccao de objetos, com anotacoes relacionadas a presenca de armas de fogo do tipo pistola em imagens.

A licenca CC BY 4.0 permite uso, adaptacao e redistribuicao, desde que seja feita atribuicao adequada. Assim, o TCC devera registrar o nome do dataset, a origem no Roboflow Universe, o autor/projeto indicado na pagina do dataset quando disponivel, a URL consultada, a data de acesso e a licenca. Essa atribuicao tambem deve acompanhar apresentacoes, demonstracoes e eventuais repositorios publicos derivados do trabalho.

O objetivo metodologico e comparar dois modelos de deteccao de objetos com caracteristicas distintas: YOLOv8 Nano e Faster R-CNN. O YOLOv8 Nano representa uma abordagem de estagio unico, geralmente associada a maior velocidade de inferencia e menor custo computacional. O Faster R-CNN representa uma abordagem de dois estagios, frequentemente usada como referencia academica por sua arquitetura consolidada em tarefas de deteccao.

## Preparacao dos dados

O dataset sera exportado do Roboflow em dois formatos. Para o YOLOv8 Nano, sera utilizado o formato YOLOv8, composto por imagens, arquivos de anotacao em texto e um arquivo `data.yaml`. Para o Faster R-CNN, sera utilizado o formato COCO JSON, adequado ao fluxo de treinamento com PyTorch e torchvision.

Antes do treinamento, a estrutura dos dados sera validada para verificar a existencia dos subconjuntos de treino, validacao e teste, bem como a correspondencia entre imagens e anotacoes. Em seguida, as classes serao inspecionadas para identificar inconsistencias de rotulagem.

As classes `pistol` e `Handgun` serao tratadas como equivalentes semanticos e unificadas em uma unica classe denominada `pistol`. As classes `0` e `1` nao serao automaticamente consideradas pistolas. Elas serao contabilizadas e amostradas visualmente para avaliacao manual, pois podem representar erro de rotulagem, classe residual, exportacao inconsistente ou outra categoria semantica.

Na primeira versao normalizada do dataset, somente anotacoes `pistol` e `Handgun` serao preservadas. Imagens que contenham exclusivamente anotacoes das classes `0`, `1` ou outras classes nao revisadas serao removidas da versao normalizada, para evitar que sejam interpretadas como imagens negativas sem pistola. Caso a analise manual demonstre que `0` ou `1` representam pistolas, essa decisao devera ser documentada e a regra de normalizacao devera ser alterada explicitamente.

## Treinamento dos modelos

O YOLOv8 Nano sera treinado com a biblioteca `ultralytics`, usando o arquivo `data.yaml` normalizado. O treinamento inicial adotara parametros simples e reprodutiveis, que poderao ser ajustados em experimentos posteriores.

O Faster R-CNN sera preparado com PyTorch e torchvision, a partir das anotacoes no formato COCO. Nesta primeira etapa, a implementacao sera mantida como uma base funcional e didatica, permitindo evolucao gradual para treinamento completo, validacao e comparacao.

## Avaliacao

A comparacao entre os modelos sera realizada apenas apos execucoes reais de treinamento e inferencia. As metricas previstas sao precisao, recall, F1-score, mAP, falsos positivos e tempo medio de inferencia por imagem. Tambem sera feita analise qualitativa em imagens com baixa qualidade, baixa iluminacao, oclusao parcial e objetos visualmente semelhantes a pistolas.

Os resultados serao registrados em arquivos de metricas, graficos e relatorios dentro da pasta `results/`. Nenhuma conclusao quantitativa sera apresentada sem a respectiva execucao experimental.

## Reprodutibilidade e controle experimental

Para permitir reproducao dos testes, cada experimento devera registrar a versao do dataset, data de exportacao, parametros de treinamento, ambiente computacional, pesos treinados utilizados e arquivos de metricas gerados. Alteracoes de hiperparametros ou de regras de normalizacao das classes devem ser documentadas antes de comparar resultados entre modelos.
