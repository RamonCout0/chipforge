# Chipforge Studio — Arquitetura

> DAW de chiptune em Python, com medidor de harmonia integrado.
> Feito para quem compõe trilha de jogo e quer o som de 8/16 bits sem
> abrir mão de saber se o que está escrevendo *soa bem*.

---

## 1. A ideia central

FL Studio é grande porque tenta ser tudo. O Chipforge escolhe um recorte e
faz esse recorte muito bem:

| O que ele é | O que ele **não** é |
|---|---|
| Sequenciador de padrões (tracker/piano roll) | Estação de gravação multipista de áudio |
| Sintetizador chiptune próprio (8/16 bits) | Host de plugins VST |
| Leitor de SoundFont para instrumentos "de verdade" | Sampler com time-stretch e slicing |
| Analisador harmônico em tempo real | Editor de partitura |
| Exportador de WAV/MIDI para engines de jogo | Masterizador profissional |

O diferencial é o **medidor de harmonia**: em vez de você descobrir que a
faixa está estranha só depois de colocar no jogo, o app aponta o compasso, as
duas notas que estão brigando e o que fazer a respeito.

---

## 2. Estrutura de pastas

```
chipforge/
├── main.py                    # ponto de entrada da aplicação
├── requirements.txt
├── chipforge/
│   ├── core/                  # 🧠 lógica pura — zero dependência de UI/áudio
│   │   ├── theory.py          #    notas, escalas, acordes, tonalidade
│   │   ├── harmony.py         #    o medidor: métricas + laudo em português
│   │   ├── generators.py      #    arpejo, baixo, bateria, melodia guiada
│   │   ├── orquestra.py       #    tessitura, condução de vozes, ostinato
│   │   ├── dinamica.py        #    mapa de tempo e automação de parâmetro
│   │   └── project.py         #    modelo de dados + save/load .chip
│   ├── audio/                 # 🔊 síntese e render
│   │   ├── oscillators.py     #    pulse, triangle, saw, LFSR noise, wavetable
│   │   ├── envelope.py        #    ADSR + modo "degrau" dos chips antigos
│   │   ├── effects.py         #    bitcrush, downsample, delay, filtro, pan
│   │   ├── instrument.py      #    Instrument + banco de presets de fábrica
│   │   ├── sampler.py         #    leitor .sf2 + SamplerInstrument
│   │   ├── fx.py              #    biquads, Freeverb, amp de guitarra, comp
│   │   ├── voz.py             #    voz cantada e coro por formantes
│   │   ├── espaco.py          #    sala por convolução, largura, Haas, eco
│   │   ├── cinema.py          #    impacto, riser, sub drop, braam, taiko
│   │   ├── master.py          #    LUFS, LRA, dBTP, multibanda, excitador
│   │   ├── jogo.py            #    regiões, pontos de loop, stems, manifesto
│   │   └── engine.py          #    render offline, export WAV, playback
│   ├── midiio/                # 🎹 interoperabilidade
│   │   ├── smf.py             #    leitor/escritor MIDI em Python puro
│   │   └── library.py         #    catálogo e download de bancos/MIDIs
│   └── ui/                    # 🖥️ PyQt6
│       ├── theme.py           #    paleta + stylesheet
│       ├── piano_roll.py      #    o piano roll (widget custom)
│       ├── harmony_panel.py   #    o mostrador do medidor
│       └── main_window.py     #    montagem da janela
├── tests/                     # 35 testes do núcleo
├── examples/demo_song.py      # monta um loop do zero e exporta
└── docs/ARQUITETURA.md
```

**Regra de dependência (importante manter):**

```
ui  →  core + audio + midiio
audio → core
midiio → core + audio
core → nada
```

`core` não importa PyQt nem sounddevice. É por isso que os 35 testes rodam em
meio segundo, sem tela e sem placa de som — e é por isso que amanhã dá para
colocar uma interface web ou um plugin de engine em cima do mesmo núcleo.

---

## 3. Modelo de dados

```
Project
├── bpm, ppq, compasso, sample_rate, bit_depth   ← "cara" do projeto
├── key_root + scale + scale_lock                ← tonalidade ativa
├── tracks: [Track]      → cada uma com um Instrument
├── patterns: {id: Pattern}   → blocos reutilizáveis de Notes
└── arrangement: [ClipRef]    → (pattern, track, posição) na linha do tempo
```

Igual ao FL Studio: você desenha **patterns** e depois espalha eles pela
timeline. Um pattern usado em 8 lugares ocupa memória uma vez só — e editar
ele muda os 8 lugares.

**Tempo em ticks.** `ppq = 480` ticks por semínima (o mesmo padrão do MIDI).
Tercina, swing e quantização ficam exatos, sem erro de ponto flutuante
acumulando ao longo de 5 minutos de música.

**Arquivo `.chip`** é um ZIP com `project.json` dentro. Legível, versionável
no git, e já tem espaço reservado para samples e wavetables customizadas.

---

## 4. Pipeline de áudio

```
Note (pitch, dur, velocity)
   ↓
Instrument.render_note()
   ├─ curva de semitons  ← arpejo + vibrato + pitch sweep + portamento
   ├─ → frequência (Hz), amostra a amostra
   ├─ oscilador          ← pulse / triangle / saw / noise LFSR / wavetable
   ├─ × envelope ADSR    ← opcionalmente quantizado em 16 degraus (4 bits)
   └─ efeitos            ← lowpass → drive → downsample → bitcrush
   ↓
render_notes()  soma as notas da track no buffer, aplica volume e pan
   ↓
render_project()  soma as tracks, master gain, soft clip, bitcrush do projeto
   ↓
export_wav()  ou  RealtimePlayer (stream do sounddevice)
```

**Por que acumulação de fase e não `sin(2πft)`?** Porque com fase acumulada o
pitch pode variar *dentro* da nota — e é isso que dá vibrato, portamento e o
"pew" de laser (pitch sweep) de graça, sem código especial para cada um.

**Por que PolyBLEP?** A onda quadrada ingênua faz aliasing feio acima de
~2 kHz. O PolyBLEP suaviza só a descontinuidade, mantendo o caráter quadrado.
O canal triangular, ao contrário, é **quantizado de propósito** em 16 degraus,
porque é exatamente isso que o chip do NES fazia.

**O bit depth é do projeto, não do instrumento.** Escolher "8 bits (NES)" na
barra superior passa o mix inteiro por um quantizador mid-riser de 256 níveis.
É o que dá unidade sonora — todos os instrumentos "moram no mesmo console".

---

## 5. O medidor de harmonia

O coração do app. Recebe uma lista de eventos `(início, duração, altura,
velocity)` em beats e devolve um `HarmonyReport`.

### Como funciona

1. **Fatiamento.** A linha do tempo é cortada em toda entrada e saída de nota.
   Cada fatia é um "acorde instantâneo" — o conjunto exato do que soa junto.

2. **Cinco métricas**, cada uma 0..1:

| Métrica | Peso | O que mede |
|---|---|---|
| Consonância vertical | 38% | Aspereza dos intervalos que soam juntos, ponderada por duração e registro |
| Aderência tonal | 22% | Quanto do tempo total está dentro da tonalidade percebida |
| Condução de vozes | 10% | Quintas/oitavas paralelas reais (mesmo par de vozes) e saltos grandes |
| Grave limpo | 16% | Intervalos apertados abaixo de C3 — a "lama" clássica |
| Progressão | 12% | Quantos acordes são funcionais na tonalidade |

3. **Score** = soma ponderada × 100, menos até 18 pontos por choques duros
   distintos (segunda menor, sétima maior, trítono em posição próxima).

### As duas decisões que evitam falso positivo

**A confiança da tonalidade escala a métrica tonal.** A detecção
Krumhansl-Schmuckler sempre devolve *alguma* tonalidade — um cluster atônico
"encaixa" perfeitamente em algum tom e ganharia 100%. Por isso a métrica é
multiplicada por `0,55 + 0,45 × confiança`. Sem tonalidade clara, ela não
carrega o score.

**Sobreposições curtas não viram alerta.** O rabo de uma nota encostando na
seguinte por 20 ms não é dissonância — é articulação. Fatias abaixo de 0,06
beat continuam entrando na média ponderada, mas não aparecem na lista de
problemas.

**Nota de passagem não é erro.** Uma dissonância só vira "problema" quando ela
*sustenta*: a mais curta das duas notas precisa durar pelo menos 0,3 beat e o
choque precisa cair no tempo forte. Uma semicolcheia de arpejo roçando na
melodia é vocabulário normal — arpejo de tracker vive disso — e sai como
"passagem", severidade mínima. Sem essa regra, qualquer arpejo rápido derruba
o score sozinho.

### Estilo

`analyze(..., style="chiptune")` (padrão) trata quinta paralela como
vocabulário do gênero, não erro. `style="strict"` liga as regras de
contraponto para quem quer escrever coral a quatro vozes.

### Saída

Além do número, cada problema vira um `Issue` com o **beat exato**, a
severidade e um texto em português nomeando as duas notas. Na interface, dar
duplo clique num problema leva o piano roll até ele.

---

## 6. Interoperabilidade

**MIDI em Python puro** (`midiio/smf.py`). Leitor e escritor de SMF formato
0 e 1 escritos à mão — VLQ, running status, meta-eventos de tempo e compasso.
Validado contra a `mido` em teste de round-trip. Zero dependência a mais.

**SoundFont** (`audio/sampler.py`). Percorre o RIFF completo
(`phdr → pbag → pgen → inst → ibag → igen → shdr`) e monta o mapa
preset → zonas → sample, com key range, root key override, afinação fina e
modo de loop. Validado contra o GeneralUser GS 1.471 (30 MB): 269 presets,
11 kits de bateria, 864 samples, lidos em cerca de um segundo.

**Escolher a zona certa importa mais do que parece.** Um preset como o
"Stereo Grand" traz mais de mil zonas — cada tecla, cada camada de velocity,
canal esquerdo e direito. Pegar a primeira que serve devolve som errado: às
vezes um layer de release, às vezes só o canal direito. A ordem de preferência
é key range e velocity range, depois sample mono ou canal esquerdo (a saída é
mono), e por fim o root mais perto da nota pedida — menos transposição,
menos artefato.

Não implementa modulators, filtros e LFOs do padrão SF2. Se um dia precisar de
fidelidade total, o caminho é `pyfluidsynth` como backend opcional.


---

## 6.5 O modo cinematográfico

A fase 2 nasceu de uma pergunta com resposta mensurável: *o que separa um
render deste app de uma trilha de OST comercial?* A resposta não foi opinião —
foi `analysis/reference.py` medindo as duas coisas lado a lado. O diagnóstico:

| Banda | Referência | Render do app | Falta |
|---|---|---|---|
| médio (800–2500 Hz) | −14,0 dB | −24,4 dB | **10 dB** |
| brilho (7–11 kHz) | −35,7 dB | −48,0 dB | **12 dB** |
| faixa de loudness | 3,4 LU | 1,6 LU | arco |

Três buracos, três módulos.

### O médio estava vazio porque faltava a voz

800–2500 Hz é onde a melodia principal mora. Numa mixagem só de SoundFont não
há nada ali com presença suficiente para carregar a faixa — e o "Choir Aahs"
de um banco GM é reconhecível como sample de teclado dos anos 90.

`audio/voz.py` sintetiza voz por modelo **fonte-filtro** (Klatt): pulso glotal
de Rosenberg → jitter e shimmer → banco paralelo de 5 formantes → formante do
cantor. Três decisões carregam o resultado:

1. **O pulso é avaliado a partir da fase acumulada**, não gerado período a
   período. É o que deixa vibrato, portamento e jitter existirem *dentro* da
   nota sem descontinuidade na forma de onda.
2. **Os ressoadores são normalizados no pico, não em DC.** O Klatt normaliza em
   DC; num banco paralelo isso faz o F1 engolir todos os outros, porque um polo
   grave tem ganho enorme em DC. Normalizando no pico, a tabela de amplitudes
   da vogal passa a significar o que diz.
3. **A velocity controla a duração do FECHO da glote**, não o quociente aberto.
   Isto foi descoberto por medição, não por teoria: variar o quociente aberto
   entre 0,40 e 0,70 mudou a inclinação espectral em 0,2 dB — nada. Encurtar o
   fecho de 24% para 7% do período mudou 8,6 dB. A largura da descontinuidade é
   que decide até onde vai o espectro dela.

O **formante do cantor** (~3 kHz) é o detalhe que faz a voz atravessar a
orquestra. Sem ele a voz soma e some.

E a semente do gerador aleatório deriva de `detune_cents`. Parece detalhe de
implementação e é o que faz a dobra de seção virar coro: o engine clona o
instrumento mudando só esse campo, então é por ele que cada garganta precisa
tremer diferente. Quatro vozes com o mesmo jitter continuam sendo uma voz, só
mais alta.

### O brilho não podia ser equalizado, tinha que ser criado

EQ multiplica o que existe. Um banco de SoundFont já vem filtrado por volta de
6–8 kHz: não há o que multiplicar. `master.excitador` **gera** — satura uma
cópia da banda de cima, o que produz harmônicos em 2f e 3f, e devolve só a
parte nova. A saturação é assimétrica de propósito: assimetria gera harmônico
de ordem par, que o ouvido lê como "brilho", enquanto só ordem ímpar soa como
distorção.

A frequência de corte do excitador é a decisão mais consequente do perfil de
master. Excitar a partir de 3,8 kHz joga tudo acima de 7 kHz e produz uma faixa
chiada e *ainda opaca*; a faixa que falta de verdade é a **presença**, de 2,5 a
7 kHz — o ataque do arco, o metal do prato, a consoante da voz.

**A ordem da cadeia não é negociável**, e é a razão de `masterizar()` ser uma
função e não seis chamadas soltas. O casamento de perfil espectral vem *depois*
do excitador: na primeira versão vinha antes, e o resultado media 13 dB de
brilho acima da referência — o casamento equilibrava, o excitador despejava
harmônicos por cima e desfazia tudo. Só dá para esculpir o que já chegou.

### O arco não existia porque não havia onde escrevê-lo

O projeto tinha um `bpm` e um `volume` por track, do primeiro ao último
compasso. `core/dinamica.py` acrescenta os dois eixos que faltavam:

* **`MapaDeTempo`** — marcas `(tick, bpm)` com interpolação linear. A conversão
  tick → segundos é a integral de 1/bpm, resolvida uma vez numa tabela
  cumulativa: o render pergunta o instante de cada nota, uma a uma, e numa peça
  de sete mil notas integrar a cada pergunta custaria mais que sintetizar o
  áudio.
* **`Automacao`** — curva por parâmetro, desenhada em ticks e lida em amostras.
  Interpolada, nunca em degraus: multiplicar um sinal por uma escada é modulação
  em anel de banda larga, e o ouvido escuta isso como clique.

### Medir, não achar

`audio/master.py` implementa **LUFS integrado** com os dois portões do
ITU-R BS.1770, **LRA** do EBU R128 e **pico real** por sobreamostragem. Os
testes verificam contra a calibração do próprio padrão: senoide de 1 kHz a
−20 dBFS lê −20 LUFS em dual mono e −23 LUFS num canal só. Sem isso, "está no
nível de uma trilha comercial" seria opinião.

O limitador tem **antecipação**: vê o pico chegar antes de ele chegar, e por
isso pode abaixar o ganho suavemente em vez de esconder a reação atrás de um
ataque tão rápido que distorce o grave. E o teto que ele persegue é o pico
*real* — se a onda reconstruída entre as amostras passar por cima, o limitador
roda de novo com o teto interno mais baixo. Abaixar a faixa inteira pelo
excedente seria mais simples e custou 3 LU de volume numa medição real: paga
com a música toda o que apenas alguns transientes deviam.

### A trilha que a engine consome

`export_wav` entrega um arquivo e pronto. Isso serve para ouvir; não serve para
jogar. Uma trilha de raid é um conjunto de faixas que a engine troca conforme a
luta anda, cada uma capaz de repetir sem que o jogador perceba a emenda.

`audio/jogo.py` resolve os três problemas que aparecem aí:

**A emenda do loop.** Cortar no compasso certo é necessário e insuficiente: no
último compasso ainda está soando a cauda da sala e o release das notas. Cortar
ali deixa um silêncio abrupto a cada volta. `costurar_loop` renderiza além do
fim e **dobra essa sobra de volta sobre o início** — na segunda volta a cauda da
primeira chega por cima da repetição, exatamente como chegaria se a música
tivesse continuado.

**O ponto de troca.** O manifesto lista as bordas de compasso, para a engine
nunca cortar no meio de uma frase. E cada fase declara um `loop_inicio` que pode
ser depois do começo: a entrada toca uma vez, a repetição acontece só do corpo
em diante. É o que faz uma fase durar dez segundos ou dez minutos sem soar
errado em nenhum dos dois casos.

**A conversão de tempo.** Com mapa de tempo, o compasso 32 não está em
`32 × 4 × 60 / bpm`. Só o projeto sabe onde ele está, e é a ele que o manifesto
pergunta. Tudo sai em três unidades: segundos para a lógica, **amostras** para o
callback de áudio (que precisa de precisão de amostra, não de milissegundo) e
compassos para conferir contra a partitura.

---

## 7. Roadmap

### ✅ Fase 1 — Núcleo (pronto)
Teoria musical · medidor de harmonia · síntese chiptune completa ·
modelo de projeto com save/load · MIDI in/out · leitor de SoundFont ·
geradores · janela principal com piano roll e medidor · 35 testes

**Validado em escala.** O exemplo `raid_coroa_de_cinzas.py` monta 112 compassos,
15 tracks, 7.400 notas e três modulações, e renderiza em ~15 s. Foi ele que
expôs quatro defeitos reais: track de bateria tocando tudo com o mesmo tambor,
ruído LFSR em laço Python puro, tonalidade relativa trocada (Ré menor lido como
Fá maior) e o medidor tratando semicolcheia de arpejo como erro.

### ✅ Fase 2 — Cinematográfico (pronto)
Voz cantada e coro por formantes · sala por convolução com IR sintetizada ·
desenho de som (impacto, riser, sub drop, braam, swell, taiko) · mapa de tempo
e automação de volume · escrita orquestral com condução de vozes · master com
LUFS/LRA/dBTP, multibanda, excitador e limitador com antecipação · casamento de
perfil espectral contra uma faixa de referência · exportação para
engine de jogo (regiões, loop costurado, stems, manifesto) · 177 testes

**Validado por medição.** `cantiga_da_torre_afogada.py` monta 84 compassos,
22 tracks e ~3.500 notas com voz solista, coro SATB e orquestra, e é comparada
banda a banda com uma OST comercial. Foi ela que expôs quatro defeitos reais:
velocity ligada ao parâmetro errado do pulso glotal, casamento de perfil rodando
antes do excitador, correção de pico real cobrando 3 LU de volume da faixa
inteira, e um si natural escrito onde a tonalidade pedia si bemol.

### 🔜 Fase 3 — Fluxo de trabalho
- Vista de arranjo (arrastar clips na timeline, hoje só via código)
- Desfazer/refazer com histórico de comandos
- Mixer com faders, VU e envio de delay/reverb
- Automação de parâmetro (volume, duty cycle, cutoff) desenhada no tempo
- Teclado do computador como controlador (linha ZXCVB = oitava)
- **Tonalidade por seção** no projeto — hoje o Scale Lock é global, mas uma
  faixa que modula precisa de marcações de tom na timeline

### 🔜 Fase 4 — Para jogo, de verdade
- **Loop points** e regiões de intro/loop — o que uma engine precisa
- **Camadas dinâmicas**: stems que entram e saem conforme o estado do jogo
  (exploração → combate), exportados em sincronia
- Exportação em lote de SFX com randomização (jsfxr integrado)
- Exportador para formatos de engine: `.ogg` + JSON de metadados,
  banco FMOD/Wwise, e `.nsf` para quem quer rodar no console real

### 🔜 Fase 5 — Assistência
- Sugestão de próximo acorde baseada na progressão atual
- "Corrigir isto" nos problemas do medidor: um clique reposiciona a nota
- Análise de referência: importa um MIDI que você gosta e mostra as métricas
  dele, para você comparar

---

## 8. Como estender

**Novo oscilador:** escreva a função em `audio/oscillators.py` (recebe array
de frequência + sr, devolve array float32), registre o nome em
`OSC_LABELS_PT` e adicione o `elif` em `Instrument.render_note`.

**Nova métrica no medidor:** escreva `_minha_metrica(slices, issues) -> float`
em `core/harmony.py`, adicione a chave em `WEIGHTS` (os pesos devem somar 1) e
o rótulo em `ui/harmony_panel.METRIC_LABELS`. A interface se ajusta sozinha.

**Novo gerador:** função em `core/generators.py` devolvendo `list[Note]`, e um
item no menu Ferramentas.

---

## 9. Decisões que valem a pena manter

1. **`core` sem dependências.** É o que deixa testar rápido e portar depois.
2. **Ticks, não segundos.** Tempo musical é discreto; áudio é contínuo. A
   conversão acontece num lugar só (`Project.ticks_to_seconds`).
3. **Render offline + playback de buffer.** Sem xruns, previsível, e o mesmo
   código serve para exportar. Síntese em tempo real amostra a amostra em
   Python puro não fecha o orçamento de latência — se um dia precisar, o
   caminho é mover só o loop de voz para Cython/numba.
4. **Português na interface e nos laudos.** O medidor só serve se a mensagem
   for entendida na hora.
