from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT_DIR / "3866092_5526412_Artigo TCC Projeto Final de Curso lucas martins da silva - revisado.docx"


TITLE = "INTELIGÊNCIA ARTIFICIAL APLICADA À SEGURANÇA PÚBLICO-PRIVADA: COMPARAÇÃO ENTRE YOLOV8 NANO E FASTER R-CNN PARA DETECÇÃO DE PISTOLAS EM IMAGENS DE SEGURANÇA"


def configure_document(document: Document) -> None:
    section = document.sections[0]
    section.top_margin = Cm(3)
    section.left_margin = Cm(3)
    section.right_margin = Cm(2)
    section.bottom_margin = Cm(2)

    normal = document.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12)
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    normal.paragraph_format.first_line_indent = Cm(1.25)
    normal.paragraph_format.line_spacing = 1.5
    normal.paragraph_format.space_after = Pt(0)


def add_centered(document: Document, text: str, bold: bool = False, size: int = 12) -> None:
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.first_line_indent = Cm(0)
    run = paragraph.add_run(text)
    run.bold = bold
    run.font.name = "Times New Roman"
    run.font.size = Pt(size)


def add_heading(document: Document, text: str) -> None:
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
    paragraph.paragraph_format.first_line_indent = Cm(0)
    paragraph.paragraph_format.space_before = Pt(12)
    paragraph.paragraph_format.space_after = Pt(6)
    run = paragraph.add_run(text.upper())
    run.bold = True
    run.font.name = "Times New Roman"
    run.font.size = Pt(12)


def add_body(document: Document, text: str, first_line: bool = True) -> None:
    paragraph = document.add_paragraph(text)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    paragraph.paragraph_format.first_line_indent = Cm(1.25 if first_line else 0)
    paragraph.paragraph_format.line_spacing = 1.5


def add_keywords(document: Document, text: str) -> None:
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    paragraph.paragraph_format.first_line_indent = Cm(0)
    run = paragraph.add_run("Palavras-chave: ")
    run.bold = True
    run.font.name = "Times New Roman"
    run.font.size = Pt(12)
    tail = paragraph.add_run(text)
    tail.font.name = "Times New Roman"
    tail.font.size = Pt(12)


def add_reference(document: Document, text: str) -> None:
    paragraph = document.add_paragraph(text)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    paragraph.paragraph_format.first_line_indent = Cm(0)
    paragraph.paragraph_format.space_after = Pt(6)
    paragraph.paragraph_format.line_spacing = 1.0


def build_document() -> None:
    document = Document()
    configure_document(document)

    add_centered(document, TITLE, bold=True, size=12)
    add_centered(document, "Lucas Martins da Silva", size=12)
    add_centered(document, "Orientador: Allan Christian Krainski Ferrari", size=12)

    add_heading(document, "Resumo")
    add_body(
        document,
        "Este artigo apresenta um projeto acadêmico de aplicação de inteligência artificial à segurança público-privada, com foco na detecção automática de pistolas em imagens de segurança. A proposta consiste em treinar, avaliar e comparar dois modelos de detecção de objetos: YOLOv8 Nano, uma arquitetura de estágio único voltada à velocidade de inferência, e Faster R-CNN, uma arquitetura de dois estágios tradicionalmente associada à maior precisão. O dataset utilizado é o Pistol_detection Computer Vision Dataset, disponibilizado no Roboflow Universe sob licença CC BY 4.0, exportado nos formatos YOLOv8 e COCO JSON. O processo metodológico inclui validação da estrutura dos dados, análise de classes ambíguas, normalização dos rótulos, treinamento supervisionado, avaliação por métricas quantitativas e comparação do comportamento dos modelos em cenários com baixa qualidade visual, baixa iluminação, oclusão parcial e objetos visualmente semelhantes. Os resultados quantitativos são registrados somente após execução real dos testes, evitando conclusões sem base experimental.",
        first_line=False,
    )
    add_keywords(document, "visão computacional; detecção de objetos; segurança; YOLOv8; Faster R-CNN; pistolas.")

    add_heading(document, "1 Introdução")
    add_body(
        document,
        "A segurança público-privada tem enfrentado o desafio crescente de monitorar grandes volumes de imagens e vídeos gerados por câmeras de vigilância. Em ambientes como empresas, condomínios, centros comerciais e espaços públicos, a análise exclusivamente humana pode ser limitada por fadiga, distração, baixa qualidade das imagens e grande quantidade de eventos simultâneos. Nesse contexto, técnicas de inteligência artificial, especialmente a visão computacional, tornam-se ferramentas relevantes para apoiar a identificação de situações de risco.",
    )
    add_body(
        document,
        "Um dos problemas observados em sistemas de monitoramento é a dificuldade de detectar objetos perigosos de forma rápida e confiável. A identificação de uma pistola em uma imagem não depende apenas da presença do objeto, mas também de fatores como iluminação, ângulo da câmera, resolução, oclusão parcial, distância e semelhança visual com outros itens. Esses fatores podem aumentar falsos positivos, reduzir a taxa de detecção e comprometer a utilidade prática de um sistema automatizado.",
    )
    add_body(
        document,
        "Diante desse problema, este trabalho tem como objetivo comparar o desempenho do YOLOv8 Nano e do Faster R-CNN na detecção de pistolas em imagens de segurança. O YOLOv8 Nano representa uma abordagem leve de estágio único, frequentemente associada a aplicações em tempo real. O Faster R-CNN, por sua vez, utiliza uma abordagem de dois estágios, com geração de propostas de regiões e posterior classificação, sendo historicamente reconhecido por sua robustez em tarefas de detecção.",
    )
    add_body(
        document,
        "A justificativa da pesquisa está na necessidade de compreender os trade-offs entre velocidade e precisão em aplicações de segurança. Um modelo rápido pode ser mais adequado para sistemas embarcados ou monitoramento em tempo real, enquanto um modelo mais robusto pode ser preferível em cenários nos quais a precisão seja prioritária. A comparação entre essas arquiteturas contribui para decisões técnicas mais fundamentadas na adoção de inteligência artificial em ambientes de segurança.",
    )
    add_body(
        document,
        "Os objetivos específicos são: preparar o dataset em formatos compatíveis com cada arquitetura; normalizar as classes equivalentes relacionadas a pistolas; treinar o YOLOv8 Nano e o Faster R-CNN; avaliar precisão, recall, F1-score, mAP, falsos positivos e tempo médio de inferência; e comparar o comportamento dos modelos em imagens com baixa qualidade, baixa iluminação, oclusão parcial e objetos semelhantes.",
    )
    add_body(
        document,
        "Este artigo está organizado em fundamentação teórica, metodologia, resultados e discussões, considerações finais e referências. A fundamentação apresenta os conceitos principais de redes neurais convolucionais e detecção de objetos. A metodologia descreve o dataset, o processo de preparação dos dados e os procedimentos de treinamento e avaliação. Os resultados apresentam as métricas obtidas experimentalmente, e as considerações finais discutem os limites e contribuições do estudo.",
    )

    add_heading(document, "2 Fundamentação teórica")
    add_body(
        document,
        "A fundamentação teórica deste trabalho aborda os principais conceitos utilizados na pesquisa: redes neurais convolucionais, detecção de objetos, arquiteturas de estágio único, arquiteturas de dois estágios e desafios da aplicação de visão computacional em imagens de segurança.",
    )

    add_heading(document, "2.1 Redes neurais convolucionais e detecção de objetos")
    add_body(
        document,
        "As redes neurais convolucionais são arquiteturas de aprendizado profundo amplamente utilizadas em tarefas de visão computacional. Elas processam imagens por meio de camadas capazes de extrair padrões locais, como bordas, texturas, formas e combinações mais complexas de características. Na detecção de objetos, além de classificar a presença de uma classe, o modelo também precisa localizar espacialmente o objeto por meio de caixas delimitadoras.",
    )
    add_body(
        document,
        "Em aplicações de segurança, a detecção de objetos pode apoiar a triagem de eventos e reduzir o tempo de resposta diante de situações críticas. No entanto, a qualidade dos resultados depende diretamente da representatividade do dataset, da consistência das anotações, do processo de treinamento e das condições visuais encontradas nas imagens avaliadas.",
    )

    add_heading(document, "2.2 Arquitetura de estágio único: YOLOv8 Nano")
    add_body(
        document,
        "O YOLO, proposto inicialmente por Redmon et al. (2016), consolidou uma família de detectores de estágio único. Nessa abordagem, a rede realiza a predição de classes e caixas delimitadoras em uma única passagem pela imagem, favorecendo maior velocidade de inferência. A versão YOLOv8 Nano, disponibilizada pela Ultralytics, é uma variação leve, adequada para experimentos em que tempo de resposta e limitação computacional são fatores relevantes.",
    )
    add_body(
        document,
        "Por ser uma arquitetura voltada à eficiência, o YOLOv8 Nano é interessante para aplicações de monitoramento em tempo real. Entretanto, em imagens com oclusão, objetos pequenos, baixa iluminação ou baixa resolução, modelos muito leves podem apresentar maior dificuldade na extração de características discriminativas.",
    )

    add_heading(document, "2.3 Arquitetura de dois estágios: Faster R-CNN")
    add_body(
        document,
        "O Faster R-CNN, apresentado por Ren et al. (2015), é uma arquitetura de detecção de objetos baseada em dois estágios. O primeiro estágio utiliza uma Region Proposal Network para sugerir regiões candidatas que podem conter objetos. O segundo estágio refina essas propostas e realiza a classificação final. Essa estrutura tende a aumentar o custo computacional, mas pode oferecer maior robustez em determinadas condições de detecção.",
    )
    add_body(
        document,
        "No contexto deste trabalho, o Faster R-CNN é utilizado como contraponto ao YOLOv8 Nano. A comparação permite avaliar se a maior complexidade computacional da arquitetura de dois estágios se traduz em ganhos relevantes de precisão para a detecção de pistolas em imagens de segurança.",
    )

    add_heading(document, "2.4 Desafios da detecção em ambientes de segurança")
    add_body(
        document,
        "A aplicação de visão computacional à segurança apresenta desafios específicos. Imagens de câmeras de vigilância frequentemente possuem baixa resolução, ruído, compressão, iluminação insuficiente, ângulos desfavoráveis e oclusões causadas por pessoas, móveis ou outros objetos. Além disso, objetos visualmente semelhantes a armas podem gerar falsos positivos.",
    )
    add_body(
        document,
        "Por esse motivo, a avaliação de modelos de detecção não deve considerar apenas uma métrica isolada. Precisão, recall, F1-score, mAP, falsos positivos e tempo médio de inferência precisam ser analisados em conjunto, pois representam diferentes dimensões do desempenho do sistema.",
    )

    add_heading(document, "3 Metodologia")
    add_body(
        document,
        "A pesquisa foi desenvolvida como um estudo experimental comparativo. O dataset utilizado foi o Pistol_detection Computer Vision Dataset, disponível no Roboflow Universe e licenciado sob Creative Commons Attribution 4.0 International (CC BY 4.0). Por essa licença, o uso acadêmico exige atribuição da origem do dataset, incluindo plataforma, nome do conjunto de dados, versão utilizada e data de acesso.",
    )
    add_body(
        document,
        "O dataset foi exportado em dois formatos: YOLOv8, para treinamento do YOLOv8 Nano com a biblioteca Ultralytics, e COCO JSON, para treinamento do Faster R-CNN com PyTorch e torchvision. Essa decisão permite utilizar a mesma base de imagens e divisões equivalentes de treino, validação e teste, respeitando o formato esperado por cada biblioteca.",
    )
    add_body(
        document,
        "Durante a inspeção inicial foram identificadas as classes pistol, Handgun, 0, 1 e pistols. As classes textuais equivalentes a pistola foram normalizadas para uma única classe denominada pistol. As classes 0 e 1 foram tratadas com cautela, pois rótulos numéricos podem representar inconsistências de exportação ou categorias semanticamente diferentes. A decisão de incorporá-las à classe final depende de análise visual documentada.",
    )
    add_body(
        document,
        "Após a preparação, o dataset processado apresentou 2.491 imagens e 2.857 anotações no conjunto de treino, 653 imagens e 757 anotações no conjunto de validação, e 364 imagens e 440 anotações no conjunto de teste. A classe final utilizada para avaliação foi pistol.",
    )
    add_body(
        document,
        "O YOLOv8 Nano foi treinado a partir do modelo base yolov8n.pt, utilizando imagens de 640 pixels, batch size 16, 50 épocas e patience 10. O Faster R-CNN foi configurado com backbone ResNet-50 FPN pré-treinado, classificador adaptado para duas classes, sendo background e pistol, batch size reduzido e treinamento com GPU CUDA quando disponível.",
    )
    add_body(
        document,
        "A avaliação dos modelos considera precision, recall, F1-score, mAP@50, mAP@50-95, falsos positivos, número de imagens de teste e tempo médio de inferência por imagem. As métricas são geradas por scripts do projeto e registradas em arquivos JSON e CSV, evitando inserção manual ou resultados fictícios.",
    )

    add_heading(document, "4 Resultados e discussões")
    add_body(
        document,
        "Até a etapa atual do experimento, o YOLOv8 Nano foi treinado e avaliado no conjunto de teste. Os resultados obtidos foram: precision de 0,8887, recall de 0,7955, F1-score de 0,8395, mAP@50 de 0,8725, mAP@50-95 de 0,6651, tempo médio de inferência de 0,0252 segundos por imagem e 108 falsos positivos em 364 imagens de teste.",
    )
    add_body(
        document,
        "Esses resultados indicam desempenho consistente do YOLOv8 Nano para uma arquitetura leve, especialmente considerando o tempo médio de inferência. A etapa de treinamento e avaliação do Faster R-CNN deve ser concluída antes da comparação final entre os modelos. Assim, não são apresentados valores quantitativos para o Faster R-CNN enquanto a execução real não estiver finalizada.",
    )
    add_body(
        document,
        "A comparação final deverá observar se o Faster R-CNN apresenta ganhos em precisão, recall ou mAP suficientes para justificar seu maior custo computacional. Também será necessário analisar qualitativamente casos de baixa iluminação, baixa qualidade, oclusão parcial e objetos semelhantes, pois esses cenários representam condições frequentes em ambientes reais de segurança.",
    )

    add_heading(document, "5 Considerações finais")
    add_body(
        document,
        "Este trabalho propõe uma comparação experimental entre YOLOv8 Nano e Faster R-CNN para detecção de pistolas em imagens de segurança. A preparação do dataset, a normalização de classes e a separação entre treino, validação e teste constituem etapas essenciais para garantir maior confiabilidade metodológica.",
    )
    add_body(
        document,
        "Os resultados parciais do YOLOv8 Nano demonstram que modelos leves podem atingir desempenho relevante com baixo tempo médio de inferência. Entretanto, a conclusão comparativa depende da finalização do treinamento e da avaliação do Faster R-CNN. A partir dessa etapa, será possível discutir com maior segurança o equilíbrio entre velocidade, precisão e aplicabilidade em cenários de segurança público-privada.",
    )
    add_body(
        document,
        "Como trabalhos futuros, recomenda-se ampliar a base de teste, avaliar vídeos reais de câmeras de segurança, analisar diferentes limiares de confiança, testar modelos intermediários da família YOLO e investigar estratégias de redução de falsos positivos.",
    )

    add_heading(document, "Referências")
    references = [
        "MATHWORKS. Getting Started with R-CNN, Fast R-CNN, and Faster R-CNN. Disponível em: https://www.mathworks.com/help/vision/ug/getting-started-with-r-cnn-fast-r-cnn-and-faster-r-cnn.html. Acesso em: 30 ago. 2026.",
        "REDMON, Joseph; DIVVALA, Santosh; GIRSHICK, Ross; FARHADI, Ali. You Only Look Once: Unified, Real-Time Object Detection. 2016. Disponível em: https://arxiv.org/abs/1506.02640. Acesso em: 30 ago. 2026.",
        "REN, Shaoqing; HE, Kaiming; GIRSHICK, Ross; SUN, Jian. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. 2015. Disponível em: https://arxiv.org/abs/1506.01497. Acesso em: 30 ago. 2026.",
        "ROBOFLOW UNIVERSE. Pistol_detection Computer Vision Dataset. Disponível em: https://universe.roboflow.com/. Licença: Creative Commons Attribution 4.0 International (CC BY 4.0). Acesso em: 30 ago. 2026.",
        "ULTRALYTICS. Ultralytics YOLO Documentation. Disponível em: https://docs.ultralytics.com/. Acesso em: 30 ago. 2026.",
        "ULTRALYTICS. Ultralytics GitHub Repository. Disponível em: https://github.com/ultralytics/ultralytics. Acesso em: 30 ago. 2026.",
    ]
    for reference in references:
        add_reference(document, reference)

    document.save(OUTPUT_PATH)
    print(f"Documento revisado salvo em: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_document()
