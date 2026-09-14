# Componentes e referências

O código original deste repositório está sob MIT. Isso não altera a licença dos
pacotes e modelos usados opcionalmente.

| Componente | Papel | Referência |
|---|---|---|
| OpenCV | Câmera, desenho, HOG/SVM e codecs | [API HOG](https://docs.opencv.org/4.x/d5/d33/structcv_1_1HOGDescriptor.html) |
| NumPy / SciPy | Matrizes e atribuição global | [linear_sum_assignment](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html) |
| Ultralytics YOLO11 | Detector neural pré-treinado da classe person | [YOLO11](https://docs.ultralytics.com/models/yolo11/) / [predict](https://docs.ultralytics.com/modes/predict/) |
| Ultralytics | Pacote opcional e pesos, condições próprias | [Licença do projeto](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) / [licenciamento](https://www.ultralytics.com/license) |

Ultralytics publica sob AGPL-3.0 e oferece modalidade Enterprise. Não presuma que
a licença MIT deste código torna o pacote/peso YOLO MIT nem resolve a licença de
uma distribuição integrada. Avalie as condições aplicáveis antes de distribuir
ou usar comercialmente. Pesos e a fotografia `bus.jpg` não são redistribuídos aqui.
O teste opcional lê a fotografia de exemplo instalada pelo próprio pacote.

O autor deste projeto implementou a orquestração, rastreamento geométrico, contagem,
persistência, interface e testes. Não treinou o YOLO11 e não reivindica sua autoria.
