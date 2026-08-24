"""
Gera os sprites de exemplo do Palco — para o app ter o que mostrar antes de
voce desenhar os seus.

Convencao (a mesma que os seus arquivos precisam seguir):

    chipforge/assets/sprites/<personagem>/nivel1.png
                                          nivel2.png
                                          ...
                                          nivel5.png

Cada PNG e uma FOLHA horizontal de quadros quadrados: se a imagem tem 256x64,
sao 4 quadros de 64x64, lidos da esquerda para a direita. Pode ter quantos
quadros quiser em cada nivel, e cada nivel pode ter uma quantidade diferente.

Alternativa: em vez do PNG unico, uma pasta `nivel1/` com os quadros soltos
em ordem alfabetica (quadro01.png, quadro02.png, ...).

    python3 tools/gerar_sprites_placeholder.py
"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw

S = 64          # lado do quadro
QUADROS = 6

# nivel -> (corpo, brilho, olho, n_espinhos, tremor, boca)
ESTILO = [
    ((38, 52, 92), (70, 95, 150), (150, 200, 255), 0, 0.0, "linha"),
    ((36, 92, 96), (60, 150, 150), (180, 255, 245), 2, 0.6, "linha"),
    ((150, 140, 42), (210, 200, 70), (255, 250, 190), 4, 1.4, "aberta"),
    ((176, 92, 32), (240, 140, 50), (255, 220, 160), 6, 2.4, "dentes"),
    ((168, 40, 52), (255, 80, 90), (255, 240, 120), 8, 3.6, "dentes"),
]


def desenha(nivel: int, q: int) -> Image.Image:
    corpo, brilho, olho, espinhos, tremor, boca = ESTILO[nivel]
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    fase = q / QUADROS * 2 * math.pi

    bob = math.sin(fase) * (1.2 + nivel * 0.5)
    dx = math.sin(fase * 3.1) * tremor
    cx, cy = S / 2 + dx, S / 2 + 4 + bob
    rx = 19 + nivel * 0.8 + math.sin(fase * 2) * (0.8 + nivel * 0.4)
    ry = 17 + nivel * 0.6 - math.sin(fase * 2) * (0.8 + nivel * 0.4)

    # os espinhos abrem em leque para CIMA, nunca na horizontal: espinho
    # horizontal vaza do quadro e aparece dentro do quadro vizinho da folha
    for i in range(espinhos):
        a = -math.pi * 0.88 + i * (math.pi * 0.76) / max(1, espinhos - 1)
        comp = 6 + nivel + math.sin(fase * 2 + i) * (1 + nivel * 0.5)
        alcance_x = min(rx + comp, S / 2 - 3)
        alcance_y = min(ry + comp, S / 2 - 3)
        x0, y0 = cx + math.cos(a) * rx * 0.92, cy + math.sin(a) * ry * 0.92
        x1, y1 = cx + math.cos(a) * alcance_x, cy + math.sin(a) * alcance_y
        d.line([(x0, y0), (x1, y1)], fill=brilho + (255,), width=3)

    d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=corpo + (255,))
    d.ellipse([cx - rx * 0.72, cy - ry * 0.95, cx + rx * 0.28, cy - ry * 0.15],
              fill=brilho + (110,))

    abertura = 0.35 + nivel * 0.13
    for lado in (-1, 1):
        ox = cx + lado * rx * 0.36
        oy = cy - ry * 0.14
        r = 3.0 + nivel * 0.55
        d.ellipse([ox - r * 1.5, oy - r * abertura * 2.2,
                   ox + r * 1.5, oy + r * abertura * 2.2], fill=(12, 12, 18, 255))
        pupila = r * (0.55 + 0.12 * math.sin(fase * 2))
        d.ellipse([ox - pupila, oy - pupila, ox + pupila, oy + pupila],
                  fill=olho + (255,))

    by = cy + ry * 0.42
    largura = rx * (0.3 + nivel * 0.07)
    if boca == "linha":
        d.line([(cx - largura, by), (cx + largura, by)],
               fill=(10, 10, 16, 220), width=2)
    else:
        alt = 3 + nivel * 1.4 + math.sin(fase * 2) * nivel * 0.6
        d.ellipse([cx - largura, by - alt / 2, cx + largura, by + alt / 2],
                  fill=(14, 8, 12, 255))
        if boca == "dentes":
            passo = max(3, int(largura * 2 / 4))
            for x in range(int(cx - largura) + 1, int(cx + largura) - 1, passo):
                d.polygon([(x, by - alt / 2), (x + passo * 0.5, by),
                           (x + passo, by - alt / 2)], fill=(240, 240, 230, 255))
    return img


def main():
    destino = Path(__file__).resolve().parents[1] / "chipforge/assets/sprites/guardiao"
    destino.mkdir(parents=True, exist_ok=True)
    for nivel in range(5):
        folha = Image.new("RGBA", (S * QUADROS, S), (0, 0, 0, 0))
        for q in range(QUADROS):
            folha.paste(desenha(nivel, q), (q * S, 0))
        caminho = destino / f"nivel{nivel + 1}.png"
        folha.save(caminho)
        print(f"{caminho.name}: {QUADROS} quadros de {S}x{S}")
    (destino / "LEIA-ME.txt").write_text(
        "Troque estes arquivos pelos seus.\n\n"
        f"Cada nivelN.png e uma folha horizontal de quadros quadrados de {S}x{S}.\n"
        "Pode usar outro tamanho: o app deduz o lado do quadro pela ALTURA da\n"
        "imagem, entao uma folha de 8 quadros de 96px fica 768x96.\n\n"
        "nivel1 = parado   ...   nivel5 = surto total\n"
        "Quanto maior a intensidade da musica, maior o nivel e mais rapida a\n"
        "animacao. Da tambem para usar uma pasta nivelN/ com os quadros soltos.\n",
        encoding="utf-8")
    print(f"em {destino}")


if __name__ == "__main__":
    main()
