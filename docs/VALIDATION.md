# Validação na câmera real

Os testes automatizados verificam o programa; não medem a acurácia da sua câmera.
Antes de usar contagens para decisões operacionais:

1. Use uma câmera autorizada, fixa e apontada para a passagem. Evite capturar áreas
   sem relação com a finalidade e minimize a retenção.
2. Posicione a linha sobre o ponto de passagem dos pés. Inverta início/fim se a
   seta de entrada estiver errada. Fixe parâmetros em `config.local.toml`.
3. Observe 20 entradas e 20 saídas individuais e anote manualmente os resultados.
4. Faça pares passando simultaneamente, em sentidos opostos e sobrepostos.
5. Teste pessoa parada, mudança de direção sem atravessar, oclusão breve, mochila,
   iluminação ruim e retorno após desaparecer da câmera.
6. Desconecte/reconecte a câmera de teste. Confirme alertas, limite de tentativas e
   ausência de passagem falsa através do gap.
7. Compare contagens reais com eventos exportados. Registre falsos positivos,
   passagens perdidas e trocas de ID separadamente.
8. Para vídeos longos, registre memória/RAM, CPU/GPU e atraso do vídeo; não trate
   o contador `processing_fps` como latência da câmera.

## Tabela para preencher

| Cenário | Entradas reais | Entradas medidas | Saídas reais | Saídas medidas | Falsas/Perdidas | Trocas de ID |
|---|---:|---:|---:|---:|---|---:|
| Uma pessoa por vez | | | | | | |
| Passagem simultânea | | | | | | |
| Oclusão / contraluz | | | | | | |
| Câmera interrompida | | | | | | |

Erro absoluto de contagem = `abs(medido - real)`. Para medir precisão/recall de
passagens, relacione eventos individualmente com anotações de tempo e sentido;
somente comparar totais pode esconder erros que se cancelam. Registre duração,
resolução, FPS de entrada, modelo, threshold e configuração usados.

## Limites de segurança de operação

Não use como único mecanismo de evacuação, acesso ou capacidade máxima. A
estimativa começa em `initial_occupancy`, não deduz pessoas já presentes fora do
enquadramento. Há apenas uma linha/câmera por processo nesta versão.
