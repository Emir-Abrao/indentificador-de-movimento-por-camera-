# People Counter — detecção e contagem local por câmera

Aplicação Python para detectar pessoas, desenhar **retângulos verdes**, acompanhar
trajetórias e contar entradas/saídas por uma linha virtual. Suporta webcam USB,
arquivo de vídeo e câmeras IP/RTSP. Não é um simples detector de pixels em movimento:
o detector principal usa a classe **person** do YOLO11.

**Estado:** implementação executada em Linux/CPU com testes automatizados e inferência
real em imagem de exemplo. A webcam, o RTSP e a precisão no seu ambiente ainda
precisam de validação em campo. Evidências e limites em [TEST_REPORT.md](docs/TEST_REPORT.md).

## O que aparece na tela

- Retângulo verde, ID temporário e score por pessoa confirmada.
- Quantidade de pessoas visíveis **agora**, sem incluir trajetórias perdidas.
- Entradas, saídas, fluxo líquido e ocupação estimada.
- Linha amarela e seta indicando a direção de entrada.
- FPS de processamento, quantidade de trajetórias e alertas de saldo inconsistente.
- `Q` ou `Esc` encerra a aplicação e libera a câmera.

## Iniciar no Windows — Python 3.11 ou 3.12

Execute no PowerShell; não é necessário ativar scripts nem alterar a política do Windows:

```powershell
git clone https://github.com/Emir-Abrao/indentificador-de-movimento-por-camera-.git people-counter
cd people-counter
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[desktop,yolo]"
.\.venv\Scripts\people-counter.exe run --source 0 --allow-model-download
```

A flag autoriza **somente** o download inicial do modelo oficial `yolo11n.pt` quando
ele ainda não existe. Após o primeiro download, ela pode ser retirada. Não exige
chave de API nem envia os frames a um serviço de inferência. O pacote YOLO e os
pesos são de terceiros: consulte [THIRD_PARTY.md](THIRD_PARTY.md).

Se `py -3.12` não existir, instale Python 3.12 ou use `py -3.11`. Se a webcam não for
`0`, experimente `--source 1`. Permita o acesso à câmera nas configurações de privacidade.
Feche Teams/Zoom se estiverem ocupando o dispositivo.

### Linux/macOS

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[desktop,yolo]'
.venv/bin/people-counter run --source 0 --allow-model-download
```

No Linux, o pacote desktop pode exigir as bibliotecas de sistema `libgl1` e
`libglib2.0-0`. Permissões de webcam dependem da configuração local. Não use `sudo`
para executar a aplicação como solução genérica.

### Modo leve, sem baixar pesos

```bash
python -m pip install -e '.[desktop]'
people-counter run --source 0 --detector hog
```

O HOG/SVM do OpenCV é um **baseline**, mais limitado em oclusão, pessoas sentadas,
multidões e ângulos de câmera. Não existe fallback silencioso: se o YOLO falhar,
a aplicação informa o erro, não troca de detector sem avisar.

## Vídeo, câmera IP e relatórios

```bash
# Processar arquivo sem abrir janela; imprimir resumo JSON ao terminar
people-counter run --source entrada.mp4 --headless --max-frames 300

# Gravação somente quando pedida; o destino não pode existir
people-counter run --source entrada.mp4 --headless --output-video data/anotado.mp4

# Banco contém eventos e métricas, não imagens
people-counter export --database data/counts.db --output contagens.csv

# Não abre câmera, não baixa modelo e não faz chamadas de inferência
people-counter doctor
```

Para RTSP, coloque a URL em uma variável de ambiente definida localmente e use
`--source-env CAMERA_URL`. Evite colocar senhas no histórico do terminal, prints,
issues ou no Git. Só conecte câmeras que você tem autorização para utilizar.
Diagnósticos nativos do backend FFmpeg podem conter detalhes da conexão; trate
stderr como sensível. O código da aplicação não registra a URL nos eventos.

Em servidor sem interface: instale `.[headless,dev]` para HOG/testes. **Não instale
opencv-python e opencv-python-headless juntos**: ambos fornecem o módulo `cv2`.
O extra YOLO instala o OpenCV desktop; use-o com `--headless` e as bibliotecas de
sistema quando quiser inferência YOLO sem janela.

## Configurar a contagem

Copie `config.example.toml` para `config.local.toml` e execute:

```bash
people-counter run --config config.local.toml --source 0
```

| Configuração | Efeito |
|---|---|
| `line_start`, `line_end` | Pontos normalizados de 0 a 1; independem da resolução |
| `confidence` | Limiar mínimo do detector; ajuste com vídeo anotado localmente |
| `min_hits` | Frames consecutivos para confirmar uma trajetória |
| `stable_frames` | Observações estáveis necessárias em cada lado da linha |
| `deadband` | Margem sem contagem em torno da linha, fração da menor dimensão |
| `cooldown_frames` | Intervalo mínimo entre eventos da mesma trajetória |
| `max_crossing_gap` | Gap máximo em frames para inferir uma passagem |
| `initial_occupancy` | Pessoas que já estavam dentro no início da sessão |
| `max_age` | Quantos frames uma trajetória ausente permanece disponível |
| `match_distance` | Distância máxima de associação, fração da diagonal |
| `camera_label` | Identificação lógica da câmera no banco, nunca a URL/senha |

O ponto usado para cruzamento é o **centro inferior da caixa**, próximo dos pés.
Uma linha da esquerda para a direita conta a descida na imagem como entrada.
Inverta seus pontos para inverter entrada/saída. Cruzar fora das extremidades
da linha não conta. Instale a câmera fixa, deixe espaço visível antes/depois da
linha e calibre com [o roteiro de validação](docs/VALIDATION.md).

## Definições que evitam métricas enganosas

- **Visíveis:** trajetórias confirmadas detectadas no frame atual.
- **Trajetórias confirmadas:** IDs confirmados na sessão, não indivíduos únicos.
  Uma pessoa que reaparece após expiração pode receber outro ID.
- **Entradas/saídas:** passagens observadas pela linha, não novos rostos na cena.
- **Fluxo líquido:** entradas menos saídas; pode ser negativo.
- **Ocupação estimada:** saldo inicial mais passagens, limitado a zero. Saídas sem
  saldo geram alertas, não são escondidas. Não serve como medição certificada de lotação.

Na reconexão/mudança de resolução, associações são limpas para evitar trajetórias
imaginárias. Totais e sequências continuam na sessão. Reiniciar o programa cria
outra sessão UUID, preservando as anteriores no SQLite. Sem reidentificação entre
câmeras, reconhecimento facial, classificação demográfica ou biometria.

## Arquitetura e testes

`capture` lê frames; `detection` detecta pessoas; `tracking` associa caixas por
atribuição global e previsão de velocidade; `counting` aplica a regra de passagem;
`storage` grava eventos atomicamente; `overlay` apresenta a imagem. São módulos
independentes, com detector/captura injetáveis nos testes. Consulte as
[decisões de arquitetura](docs/ARCHITECTURE.md).

```bash
python -m pip install -e '.[headless,dev]'
ruff check .
ruff format --check .
pytest --cov=people_counter --cov-report=term-missing --cov-fail-under=80
```

A suíte comum não acessa webcam, RTSP nem baixa modelos. Para testar o YOLO real,
instale os extras `desktop,yolo,dev` em outro ambiente e forneça pesos existentes:

```powershell
$env:PEOPLE_COUNTER_MODEL = "yolo11n.pt"
.\.venv\Scripts\python.exe -m pytest -m model -v
```

Em Linux/macOS: `PEOPLE_COUNTER_MODEL=yolo11n.pt pytest -m model -v`.
A imagem de teste vem do pacote Ultralytics; não há vídeo pessoal no repositório.
GitHub Actions executa lint e testes em Python 3.11/3.12, sem webcam ou pesos.

## Limites e privacidade

Uma câmera por processo. Acurácia depende de lente, iluminação, perspectiva,
densidade, oclusão e taxa de frames. Não prometemos 100% de precisão ou desempenho
em tempo real em qualquer hardware. O tracker geométrico pode trocar IDs quando
pessoas se sobrepõem; este é um projeto funcional de portfólio, não um sistema de
segurança certificado. Não faz treinamento de um modelo novo.

Nenhum vídeo é gravado por padrão. Contagens e histórico local também precisam
de controle de acesso e retenção. O SQLite não é criptografado nem inviolável.
Veja [SECURITY.md](SECURITY.md). Nada é publicado na internet pela aplicação.
