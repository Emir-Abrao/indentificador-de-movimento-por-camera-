# Evidências de teste — 14/09/2026

## Resultado executado, não apenas testes escritos

| Verificação | Resultado |
|---|---|
| `ruff check .` | Aprovado |
| `ruff format --check .` | Aprovado |
| `python -m pip check` | Nenhum requisito quebrado |
| CLI instalado: `people-counter doctor`, `--help`, `python -m people_counter --version` | Aprovado |
| Suíte com peso local: `PEOPLE_COUNTER_MODEL=yolo11n.pt pytest --cov=people_counter --cov-report=term-missing --cov-fail-under=80` | **76 testes passaram**, nenhum skip/falha, 7,81 s nesta execução |
| Cobertura combinada de statements/branches reportada pelo coverage.py | **97,11%** (não é acurácia de detecção) |
| Smoke de ponta a ponta: `python tools/smoke_model.py --model yolo11n.pt` | Aprovado |

## Ambiente da execução

Linux x86_64, Python 3.12.14, CPU; sem câmera física, servidor RTSP ou GUI real.

| Pacote | Versão executada |
|---|---|
| NumPy | 2.5.3 |
| SciPy | 1.18.1 |
| opencv-python | 4.14.0.94 |
| Ultralytics | 8.4.152 |
| torch | 2.14.0+cpu |
| torchvision | 0.29.0+cpu |
| pytest | 9.1.1 |
| pytest-cov | 7.1.0 |
| coverage | 7.16.1 |
| Ruff | 0.16.7 |

OpenCV 5 foi rejeitado durante o desenvolvimento: o binding instalado não expunha
`HOGDescriptor`. O projeto limita esse pacote a `>=4.10,<5` e repetiu a suíte com
OpenCV 4.14.0.94. Não houve exclusão dos testes que encontraram o problema.

## Testes de integração reais

1. HOG/SVM real em frames vazios: nenhuma pessoa detectada; também foram exercitados
   decodificação de AVI, escrita/leitura de MP4 e encerramento do SQLite.
2. YOLO11n real na imagem `bus.jpg` distribuída com Ultralytics: múltiplas pessoas
   detectadas; IDs mantidos em inferências repetidas da mesma imagem.
3. Smoke separado montou um AVI de **8 frames repetidos dessa fotografia**, passou
   pelo detector real, tracker, contador, desenho, codec MP4 e banco SQLite:
   **4 pessoas visíveis, 4 trajetórias confirmadas, 0 entradas, 0 saídas**. O arquivo
   de saída foi reaberto com 8 frames; sessão final `completed`. Como não existe
   movimento nessa amostra, zero cruzamentos é o resultado esperado.

Peso utilizado: `yolo11n.pt`.
SHA-256: `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`.
O peso não é incluído no repositório. Downloads futuros podem mudar; compare o hash
para reproduzir exatamente o mesmo artefato de inferência.

O smoke registrou 15,31 FPS de processamento nessa execução curta. Isso **não é**
benchmark nem promessa de FPS em webcam/hardware do usuário: há poucos frames,
aquecimento de modelo e ausência de captura física/GUI.

## O que foi simulado de forma explícita

Passagens em sentidos opostos, estabilização da caixa, perda/retorno de IDs,
reconexão, falhas de detector e clique de saída usam sequências determinísticas
e objetos substitutos. Isso verifica invariantes de lógica e encerramento; não
prova capacidade de detectar pessoas sob oclusão ou reconhecer o mesmo indivíduo.

A suíte padrão sem `PEOPLE_COUNTER_MODEL` pula **somente** o teste neural opt-in;
ela não baixa pesos nem abre câmeras. O smoke é um comando manual separado.

## Ainda não validado

- Webcam física ou stream RTSP com credenciais do usuário.
- Janela gráfica real: o fluxo GUI é testado com funções de janela substituídas.
- Windows/macOS e GPU/CUDA executados fisicamente neste ambiente.
- Vídeo de pessoas realmente em movimento, multidões, baixa luz ou campo real.
- Teste prolongado de consumo de recursos, corte de energia ou vídeo VFR.
- Precisão/recall ou taxa de erro de contagem no local de instalação.

O workflow do GitHub executa uma matriz Linux/Python 3.11/3.12; consultar a aba
Actions para o estado de cada execução. Não confundir workflow configurado com
execução remota aprovada. O roteiro de [validação em campo](VALIDATION.md) completa
o aceite antes de uso operacional.
