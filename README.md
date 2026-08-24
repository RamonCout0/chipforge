# 🎮 Chipforge Studio

DAW de chiptune em Python — piano roll, síntese 8/16 bits, e um **medidor de
harmonia** que diz, em português, se o que você escreveu está soando bem.

![status](https://img.shields.io/badge/fase-2%20(cinematográfico)-brightgreen)
![tests](https://img.shields.io/badge/testes-177%20passando-brightgreen)

---

## Instalar e rodar

```bash
pip install -r requirements.txt
python3 main.py
```

Precisa de Python 3.10+. As dependências são quatro: `numpy`, `sounddevice`
(áudio), `PyQt6` (interface) e `scipy`. O scipy é o único opcional — todo
filtro tem laço direto de reserva — mas o modo cinematográfico faz dezenas de
milhares de passagens de filtro por faixa, e sem ele o render de três minutos
sai de segundos para horas.

Para ouvir os exemplos sem abrir a interface:

```bash
python3 examples/demo_song.py               # loop chiptune puro, 8 compassos
python3 examples/demo_ponte_de_pedra.py     # 24 compassos: chip → acústico → os dois
python3 examples/raid_coroa_de_cinzas.py    # 3 minutos, 15 tracks, tema de raid
python3 examples/vigilia_do_colosso.py      # orquestra + banda de rock, ~2:50
python3 examples/cantiga_da_torre_afogada.py  # voz + coro + orquestra, ~3:25
python3 examples/guardia_dos_mares.py          # raid em 3 fases com loop, ~12 min
```

Os três últimos baixam sozinhos o banco de instrumentos na primeira execução
(GeneralUser GS, ~28 MB, licença livre para uso musical) e ficam em cache.

`raid_coroa_de_cinzas` é o teste de estresse do sequenciador: 112 compassos,
sete seções, três modulações, orquestra misturada com os canais do NES.

`cantiga_da_torre_afogada` é a peça de referência do modo cinematográfico: voz
cantada sintetizada, coro SATB, sala por convolução, mapa de tempo com
accelerando e ritardando, e cadeia de master com alvo de LUFS. Passando o
caminho de uma faixa que você admira, ela **mede** essa faixa e persegue o
balanço espectral e o loudness dela — sem copiar nenhuma nota:

```bash
python3 examples/cantiga_da_torre_afogada.py "minha_referencia.mp3"
```

`guardia_dos_mares` é o teste da **trilha de jogo**: 12 minutos em três fases
que se conectam e repetem — a sereia com piano e voz, os sinos com guitarras e
violinos, e o Leviatã com tudo junto. Sai a faixa inteira, um stem por fase com
o loop já costurado, e o manifesto JSON com os pontos de troca para a engine.

`vigilia_do_colosso` é o teste da **mixagem**: mesmo banco de samples, mas com
naipe de cordas por dobra, guitarras com simulador de amplificador gravadas em
dobro (uma em cada lado) e sala de concerto por envio de reverb. Ele também
exporta a curva de intensidade em JSON, para o jogo animar o mesmo personagem
que o Palco anima aqui dentro.

Para rodar os testes:

```bash
python3 -m pytest tests -q
```

---

## O que já funciona

**Composição**
- Piano roll com desenho, arraste, redimensionamento e apagar por clique direito
- Escala ativa iluminada nas linhas do teclado; **Scale Lock** prende toda nota
  nova na escala
- Snap de 1/1 a 1/16 e tercina; quantizar, humanizar, transpor, dobrar pattern
- Geradores: progressão harmônica, arpejo, baixo (5 estilos), bateria
  (5 estilos), melodia guiada por acorde

**Som**
- Osciladores dos chips reais: pulse com 4 duty cycles, triangular quantizada
  em 16 degraus (NES), serra, ruído por LFSR de 15 bits, wavetable de 4 bits
  (canal WAV do Game Boy)
- 22 presets de fábrica em 4 famílias (leads, baixos, pads, bateria)
- Envelope ADSR com modo "degrau", vibrato, arpejo, pitch sweep, portamento
- Bit depth do projeto: **8 bits (NES) / 16 bits (SNES) / 24 bits**
- Leitor de SoundFont `.sf2` — qualquer banco GM gratuito vira instrumento,
  com escolha de zona por key range, velocity e proximidade de root, e o mesmo
  bitcrusher do projeto por cima (piano de verdade "morando no console")

**Medidor de harmonia**
- Score 0–100 com veredito em português
- Detecção de tonalidade (Krumhansl-Schmuckler) e de acordes com inversão
- 5 métricas: consonância, aderência tonal, condução de vozes, grave limpo,
  progressão
- **Análise por seção**: música que modula não tem uma tonalidade só, então
  cada trecho é julgado no próprio tom
- Lista de problemas com o **compasso exato**, as notas envolvidas e **quais
  tracks estão brigando** — duplo clique leva o piano roll até lá
- Distingue erro de intenção: nota de passagem e tensão diatônica espalhada
  entram como cor, não como problema
- Sugestões acionáveis, não só o diagnóstico

**Mixagem** — o que faz orquestra soar como orquestra e rock como rock
- **Reverb** de sala/hall (topologia Freeverb) num barramento único, com envio
  por track: todos os instrumentos passam a soar no mesmo lugar
- **Simulador de amplificador** de guitarra: corte de grave, dois estágios de
  válvula com clipping assimétrico, tonestack e cabine — com oversampling 4×,
  porque distorção sem oversampling volta rebatida como chiado digital
- **Dobra de seção**: um violino vira naipe. N vozes em uníssono, cada uma
  desafinada alguns cents, entrando alguns milissegundos depois e sentada
  noutro lugar do palco
- Chorus, compressor, limitador e biquads (peak / shelf / passa-alta e baixa)
- Cadeia declarativa e serializável: os efeitos de cada track vão no `.chip`

**Voz e coro** — sintetizados, não sampleados
- **Voz cantada por formantes** (modelo fonte-filtro, o mesmo do Klatt): pulso
  glotal de Rosenberg com jitter e shimmer, banco paralelo de 5 formantes com
  tabela de vogal cantada por naipe, vibrato com entrada atrasada, portamento
  entre notas próximas e um caminho de ar que não passa pelo trato
- **Formante do cantor** em ~3 kHz: a ressonância que faz uma voz treinada
  passar por cima de oitenta músicos sem gritar. É ela que resolve a faixa de
  800–2500 Hz, onde toda mixagem só de sample fica magra
- Velocity muda o **timbre**, não só o nível: cantar forte encurta o fecho da
  glote e abre o espectro em ~8 dB
- 10 presets (solista feminina/masculino, SATB de coro, boca fechada, sussurro),
  todos calibrados no mesmo volume, e `satb()` distribui o acorde nos 4 naipes
- Com a dobra de seção, cada garganta treme por conta própria — é o que separa
  um coro de uma voz só, mais alta

**Sala por convolução**
- Resposta ao impulso **sintetizada** (nada para baixar): reflexões iniciais
  discretas + cauda de ruído com decaimento **por banda**, que é o que
  diferencia pedra de cortina
- 8 salas medidas e verificadas em teste (estúdio, câmara, salão, hall,
  catedral, caverna, placa, épico)
- Convolução por FFT com sobreposição e soma: 30 s de áudio em 0,2 s
- Largura meio/lado que **mantém o grave no centro**, Haas e eco ping-pong

**Desenho de som cinematográfico**
- Impacto (sub + corpo + metal, com o sub atrasado — é o que faz soar grande em
  vez de só alto), riser com corte seco, sub drop, braam, swell invertido e taiko
- Todos respeitam o contrato de `Instrument`: entram numa track normal, aceitam
  inserts e envio de sala, e aparecem no piano roll

**Arco: mapa de tempo e automação**
- **Mapa de tempo** com interpolação linear de BPM: accelerando, ritardando e
  fermata. A peça deixa de andar no metrônomo
- **Automação de volume** por track, desenhada em ticks e lida em amostras —
  o crescendo de doze compassos que velocity de nota não faz
- `arco_de_secoes()` transforma a lista de seções da peça na curva de dinâmica

**Escrita orquestral**
- Tessitura por instrumento, `limpar_grave()` (a lama abaixo de Dó3) e
  condução de vozes que escolhe a inversão de menor movimento
- Ostinato, tremolo, pedal com oitava dobrada, divisi e padrões de taiko
- `humanizar_naipe()` espalha as vozes **dentro** do acorde e atrasa o grave —
  um contrabaixo leva mais tempo para a corda falar que um violino

**Master com medição de verdade**
- **LUFS integrado** com os dois portões do ITU-R BS.1770, **LRA** do EBU R128
  e **pico real** (dBTP) por sobreamostragem — validados contra a calibração do
  padrão nos testes
- **Excitador de harmônicos**: EQ só multiplica o que já existe, e um banco de
  SoundFont não tem nada acima de 8 kHz para multiplicar. O excitador **cria**
- Compressor multibanda (crossover Linkwitz-Riley), EQ meio/lado e limitador
  com antecipação, que recontrola o teto até o **pico real** caber
- **Casar perfil**: meça uma faixa que você quer alcançar e o master persegue o
  balanço espectral dela — sem copiar uma única nota

**Para a engine do jogo**
- **Regiões e pontos de loop**: cada fase da luta declara onde entra, onde
  repete e para onde vai depois
- **Loop costurado**: a cauda de sala que vinha depois do corte é dobrada de
  volta sobre o início, então a repetição não tem o silêncio abrupto que
  denuncia um loop mal cortado
- **Stems por fase** + manifesto JSON com os pontos de troca em segundos,
  amostras e compassos — e as bordas de compasso, para a engine nunca cortar no
  meio de uma frase

**Palco de sprites**
- Um personagem animado que reage à **intensidade** da música — 5 níveis, do
  parado ao surto, com a animação acelerando junto
- Intensidade não é volume: é densidade de notas, polifonia, camadas ativas e
  bateria. Trecho baixinho e frenético marca alto; acorde longo fortíssimo
  marca baixo
- Coloque seus PNGs em `chipforge/assets/sprites/<personagem>/nivel1..5.png`
  (folhas horizontais de quadros quadrados) e o app acha sozinho
- `Arquivo → Exportar curva de intensidade` gera um JSON tempo → nível, para
  a engine do jogo animar o mesmo personagem sem analisar áudio em tempo real

**Entrada e saída**
- Salvar/abrir `.chip` (ZIP com JSON — versionável no git)
- Importar e exportar MIDI (leitor/escritor próprio, em Python puro)
- Exportar WAV 8 ou 16 bits, e a curva de intensidade em JSON
- Bit depth 24 desliga o bitcrush do projeto: é o modo em que uma
  trilha orquestral deve ser exportada

---

## Atalhos

| Tecla | Ação |
|---|---|
| `Espaço` | Tocar / pausar |
| `Esc` | Parar |
| Clique esquerdo | Criar nota |
| Arrastar borda direita | Mudar duração |
| Clique direito | Apagar nota |
| `Ctrl` + arrastar | Duplicar nota |
| `Shift` + arrastar | Mover só no tempo |
| `Ctrl` + roda | Zoom horizontal |
| `Alt` + roda | Zoom vertical |
| `↑` `↓` | Transpor seleção (com `Shift`: oitava) |
| `Delete` | Apagar seleção |

---

## Biblioteca de instrumentos

O menu **Biblioteca** traz um catálogo de bancos gratuitos com a licença
declarada em cada entrada — porque licença errada em trilha de jogo vira
problema na hora de publicar.

| Banco | Presets | Licença |
|---|---|---|
| [GeneralUser GS 1.471](https://www.schristiancollins.com/generaluser.php) | 269 + 11 kits | Uso livre em música, inclusive comercial |
| [FluidR3 GM](https://member.keymusician.com/Member/FluidR3_GM/) | GM completo | MIT |
| [500 SoundFonts Collection](https://archive.org/details/500-soundfonts-full-gm-sets) | coleção | ver página do item |
| [Wikimedia Commons — MIDI](https://commons.wikimedia.org/wiki/Commons:Free_media_resources/Music) | MIDI | domínio público / CC |

**Baixar e usar**, tudo dentro do app:

1. `Biblioteca → Baixar banco de instrumentos` — barra de progresso, cache em
   `~/.chipforge/library`, baixar de novo não baixa de novo.
2. `Biblioteca → Instrumento do banco na track atual` — digite para filtrar
   entre os 269 presets e aplique na track selecionada.

O GeneralUser GS é servido pelo registro do npm, que costuma passar por
firewall corporativo quando o site do autor não passa; o espelho oficial fica
como segunda opção no catálogo. O catálogo mora em `~/.chipforge/sources.json`
e é editável — achou um pack bom, é só acrescentar. Se a rede estiver
bloqueada de vez, baixe pelo navegador e use
**Biblioteca → Abrir banco .sf2 do computador**.

---

## Documentação

- [`docs/ARQUITETURA.md`](docs/ARQUITETURA.md) — estrutura, pipeline de áudio,
  o algoritmo do medidor, roadmap por fases e como estender
