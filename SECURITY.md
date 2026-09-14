# Segurança e privacidade

- Use apenas câmeras e gravações autorizadas. O projeto conta trajetórias, não
  identifica pessoas, rostos, idade, gênero ou condição de saúde.
- O pipeline executa inferência local e não contém endpoint para upload de frames.
- Sem gravação por padrão. `--output-video` é opt-in; trate esse arquivo como
  informação sensível, não como material para commitar.
- Dados, pesos e `config.local.toml` ficam fora do Git por padrão. Revise o que
  será enviado em todo commit; `.gitignore` não remove arquivos já versionados.
- Guarde credenciais RTSP em ambiente local; não publique em issues. Bibliotecas
  nativas podem emitir diagnósticos que contenham URLs: proteja também stderr.
- `data/counts.db` armazena horários, IDs de sessão/trajetória e métricas. Não é
  criptografado e requer permissões de sistema e política de retenção do operador.
- Modelos `.pt` devem vir de fontes confiáveis. Nunca carregue pesos desconhecidos.
- Exportação CSV neutraliza prefixos de fórmulas na identificação da câmera.
- Nenhuma imagem, vídeo de usuário ou credencial faz parte da suíte de testes.
- As versões de pacotes têm limites de compatibilidade, não garantia de ausência
  de vulnerabilidades. Revise atualizações antes de uso operacional.

Relate falhas de segurança ao responsável pelo repositório sem incluir senhas,
URLs privadas de câmeras ou imagens de pessoas em uma issue pública.
