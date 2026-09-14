# Decisões técnicas

## Detecção não é rastreamento, e rastreamento não é identificação

YOLO detecta a classe `person` em cada frame. As caixas são recortadas aos limites
da imagem e resultados inválidos são descartados. O adapter restringe classes,
score e quantidade máxima de detecções. HOG é uma alternativa explícita para
execução sem download, não um substituto de mesma precisão.

O tracker usa `scipy.optimize.linear_sum_assignment` para minimizar globalmente
um custo de distância normalizada + IoU. Há previsão por velocidade constante,
limite de deslocamento e limite de mudança de área. Colunas dummy permitem
não-associação sem forçar pareamentos inválidos. Confirmação exige observações
consecutivas; a trajetória confirmada pode sobreviver a uma oclusão curta.

Associações expiram após `max_age`. A quantidade ativa é limitada a 300 para
conter o custo da matriz de associação. O ID cresce monotonamente e não é
reutilizado na sessão. Não são mantidos embeddings, descritores faciais ou nomes.
Se a capacidade for atingida, o resumo informa `untracked_detections` para tornar
essa perda observável; a ocupação não deve ser tomada como confiável nesse caso.
O método não é ByteTrack/DeepSORT; o nome do algoritmo não é usado indevidamente.

## Regra de passagem

O centro inferior da caixa é projetado contra uma linha dirigida finita. A
distância assinada define o lado. A faixa morta reduz jitter; confirmação de lado
e cooldown reduzem oscilações. Uma interseção fora do segmento não produz evento.
Um gap longo invalida o lado anterior: não há como provar a passagem no trecho
não observado. Isso privilegia evitar falsas passagens, podendo perder passagens
verdadeiras durante falhas. Reentrada legítima após cooldown conta novamente.

Eventos simultâneos atualizam a ocupação pelo saldo do frame, para que a ordem
das caixas do detector não altere o resultado. Saídas abaixo de zero são
registradas como alertas e mantidas no fluxo líquido.

## Persistência e falhas

SQLite em WAL, com foreign keys, timeout de lock e transações. A chave composta
`(session_id, sequence)` torna a repetição do mesmo evento idempotente no banco.
Eventos e o snapshot associado são gravados na mesma transação. O programa para
em falha de persistência: não anuncia métricas duráveis após perder o banco.
Snapshots finais são tentados no encerramento; Ctrl+C fecha a sessão como interrompida.

Isso não significa exactly-once frente a corte de energia: um evento contado na
memória antes do commit pode se perder se o processo morrer abruptamente. Não há
retomada de vídeo/frame após crash. Cada execução inicia sessão e trackers novos.
Registros antigos ficam preservados; sessões `running` antigas indicam interrupção
sem encerramento. Não há claim de auditoria imutável.

## Captura e vídeo

Leitura sequencial; nenhuma fila ilimitada de frames em Python. A aquisição é
síncrona, por isso throughput menor que o FPS da câmera pode gerar latência nos
buffers do backend. Limites de abertura/leitura são configurados para FFmpeg de
streams de rede; drivers locais não garantem o mesmo timeout.

Reconexões são limitadas por falhas consecutivas e registradas em métricas. EOF de
arquivo encerra normalmente; falha antes do primeiro frame é erro. OpenCV não
distingue com confiança EOF normal de corrupção no meio de todo tipo de arquivo.
Os timestamps de arquivo são calculados pelo FPS nominal: fonte VFR pode precisar
de normalização prévia. Nenhum frame é descartado intencionalmente em arquivos.

MP4 anotado só é criado com opção explícita e destino novo. Uma mudança de
resolução durante gravação interrompe a execução para evitar um arquivo inválido.
FPS exibido mede processamento do frame, não é benchmark certificado fim a fim.

## Superfície de segurança

Não há servidor HTTP, login externo nem porta de dashboard aberta. O operador
controla arquivos, ambiente e câmera. Pesos `.pt` são carregáveis como modelos
Python: use somente pesos confiáveis. Download automático é limitado à opção
explícita do peso oficial padrão. Dependências e modelos conservam suas licenças.
Não há dados pessoais no seed/testes, e frames não são enviados pelo pipeline
à nuvem. Bibliotecas de terceiros têm seus próprios mecanismos de configuração.
