import base64
import io
import json
import numpy as np
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components

from motor_ct import RenderizadorCT, construir_Mwc

try:
  from streamlit_image_coordinates import streamlit_image_coordinates

  TEM_IMG_COORDS = True
except ImportError:
  TEM_IMG_COORDS = False


# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA & ESTILO GLOBAL
# ==============================================================================
st.set_page_config(
    page_title="Bloco 734 (CT/UFC) — Ray Casting & Walkthrough 3D",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .block-container {
        padding-top: 1.1rem;
        padding-bottom: 1.5rem;
    }
    .pick-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #38bdf8;
        border-left: 5px solid #38bdf8;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 8px 0 12px 0;
        color: #f8fafc;
        box-shadow: 0 8px 24px rgba(2, 6, 23, 0.35);
    }
    .pick-banner-title {
        font-size: 1.02rem;
        font-weight: 700;
        color: #38bdf8;
        margin-bottom: 4px;
    }
    .pick-banner-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
        gap: 6px 14px;
        font-size: 0.85rem;
        color: #cbd5e1;
        margin-top: 6px;
    }
    .pick-banner-desc {
        font-size: 0.86rem;
        color: #e2e8f0;
        margin-top: 6px;
        padding-top: 6px;
        border-top: 1px solid rgba(148, 163, 184, 0.25);
    }
    .quest-card {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 10px;
        color: #f1f5f9;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ==============================================================================
# GERENCIAMENTO DE ESTADO DA SESSÃO
# ==============================================================================
if "resolucao" not in st.session_state:
  st.session_state.resolucao = 500
if "renderer" not in st.session_state or (
    st.session_state.renderer.W != st.session_state.resolucao
):
  st.session_state.renderer = RenderizadorCT(
      largura=st.session_state.resolucao, altura=st.session_state.resolucao
  )
if "ultima_img_arr" not in st.session_state:
  st.session_state.ultima_img_arr = None
if "ultima_legenda" not in st.session_state:
  st.session_state.ultima_legenda = ""
if "ultimo_pick_info" not in st.session_state:
  st.session_state.ultimo_pick_info = None
if "ultima_coord_clicada" not in st.session_state:
  st.session_state.ultima_coord_clicada = None
if "cfg_renderizada" not in st.session_state:
  st.session_state.cfg_renderizada = None


def renderizar_banner_pick(info, px=None, py=None):
  if not info:
    return
  if not info.get("hit", False):
    st.markdown(
        f"""
        <div class="pick-banner" style="border-left-color:#94a3b8; border-color:#475569;">
            <div class="pick-banner-title" style="color:#cbd5e1;">
                🎯 Pixel ({px}, {py}) — {info.get('nome', 'Céu / Horizonte')}
            </div>
            <div class="pick-banner-desc">{info.get('descricao', '')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return

  coord_txt = f"Pixel ({px}, {py}) • " if (px is not None and py is not None) else ""
  st.markdown(
      f"""
      <div class="pick-banner">
          <div class="pick-banner-title">
              🎯 {coord_txt}{info['nome']} <span style="font-size:0.8rem; color:#94a3b8;">[{info['tipo_primitiva']}]</span>
          </div>
          <div class="pick-banner-grid">
              <div><b>Grupo AR:</b> {info['grupo']}</div>
              <div><b>Material:</b> {info['material']}</div>
              <div><b>Distância (t):</b> {info['distancia']:.2f} m</div>
              <div><b>Ponto 3D (Mundo):</b> {info['ponto_world']}</div>
              <div><b>Normal N:</b> {info['normal_world']}</div>
              <div><b>Kd / Refl:</b> {info['kd']} | R={info['refletividade']:.2f}</div>
          </div>
          <div class="pick-banner-desc">
              <b>Transformação:</b> {info['transformacao']}<br/>
              <b>Descrição:</b> {info['descricao']}
          </div>
      </div>
      """,
      unsafe_allow_html=True,
  )


# ==============================================================================
# HTML / THREE.JS — MODO WEBGL ESTILO ROBLOX COMPLETO + QUESTS + CUTSCENE CAFÉ
# ==============================================================================
def gerar_html_jogo_webgl(modo_noite=False):
  template = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Bloco 734 — Modo Jogo 3D</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; user-select: none; }
  html, body {
    width: 100%; height: 100%; overflow: hidden;
    background: #0b1120; font-family: 'Segoe UI', system-ui, sans-serif;
  }
  #game-wrapper {
    position: relative; width: 100%; height: 680px;
    background: #0b1120; border-radius: 12px; overflow: hidden;
    border: 1px solid #1e293b;
  }
  #game-wrapper.pseudo-fullscreen {
    position: fixed !important; inset: 0 !important;
    width: 100vw !important; height: 100vh !important;
    z-index: 999999 !important; border-radius: 0 !important;
  }
  canvas { display: block; width: 100%; height: 100%; }

  /* HUD MINIMALISTA (SEM BANNERS INVASIVOS) */
  #hud-top-left {
    position: absolute; top: 12px; left: 12px;
    display: flex; flex-direction: column; gap: 6px;
    pointer-events: none; z-index: 10;
  }
  .hud-pill {
    background: rgba(15, 23, 42, 0.72);
    backdrop-filter: blur(6px);
    border: 1px solid rgba(148, 163, 184, 0.28);
    color: #f8fafc; padding: 6px 12px; border-radius: 999px;
    font-size: 12.5px; font-weight: 600;
    display: flex; align-items: center; gap: 8px;
    width: fit-content;
  }
  .bar-bg {
    width: 96px; height: 8px; background: rgba(30, 41, 59, 0.9);
    border-radius: 999px; overflow: hidden;
  }
  #energy-fill {
    width: 100%; height: 100%;
    background: linear-gradient(90deg, #22c55e, #38bdf8);
    transition: width 0.2s;
  }

  #hud-top-right {
    position: absolute; top: 12px; right: 12px;
    display: flex; gap: 8px; z-index: 15;
  }
  .hud-btn {
    background: rgba(15, 23, 42, 0.82);
    backdrop-filter: blur(6px);
    border: 1px solid rgba(56, 189, 248, 0.45);
    color: #f8fafc; padding: 7px 13px; border-radius: 8px;
    font-size: 12.5px; font-weight: 600; cursor: pointer;
    transition: all 0.15s ease;
  }
  .hud-btn:hover { background: rgba(30, 41, 59, 0.95); border-color: #38bdf8; transform: translateY(-1px); }

  /* PROMPT DE PROXIMIDADE DISCRETO */
  #interact-prompt {
    position: absolute; bottom: 82px; left: 50%; transform: translateX(-50%);
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid #38bdf8; color: #f8fafc;
    padding: 7px 16px; border-radius: 999px;
    font-size: 13px; font-weight: 600;
    display: none; z-index: 12; pointer-events: none;
    box-shadow: 0 4px 16px rgba(0,0,0,0.45);
  }

  /* BOTÕES DE AÇÃO NO CANTO INFERIOR DIREITO */
  #hud-bottom-right {
    position: absolute; bottom: 14px; right: 14px;
    display: flex; gap: 8px; z-index: 14;
  }
  .action-btn {
    background: rgba(15, 23, 42, 0.84);
    border: 1px solid rgba(148, 163, 184, 0.4);
    color: #f8fafc; padding: 9px 14px; border-radius: 10px;
    font-size: 12.5px; font-weight: 700; cursor: pointer;
    transition: 0.15s;
  }
  .action-btn:hover { background: #1e293b; border-color: #38bdf8; }
  .action-btn.ar { border-color: #38bdf8; color: #7dd3fc; }
  .action-btn.atk { border-color: #f97316; color: #fdba74; }

  /* CARTÃO DE REALIDADE AUMENTADA ("O QUE É ISSO?") */
  #ar-card {
    position: absolute; top: 64px; right: 14px; width: 310px;
    background: rgba(9, 14, 28, 0.88);
    backdrop-filter: blur(10px);
    border: 1px solid #22d3ee; border-left: 4px solid #22d3ee;
    border-radius: 10px; padding: 12px 14px; color: #e2e8f0;
    display: none; z-index: 18;
    box-shadow: 0 10px 30px rgba(34, 211, 238, 0.22);
    animation: fadeIn 0.2s ease;
  }
  #ar-card h4 { color: #22d3ee; font-size: 14px; margin-bottom: 4px; }
  #ar-card .ar-tag {
    display: inline-block; font-size: 10.5px; padding: 2px 7px;
    background: rgba(34, 211, 238, 0.16); color: #67e8f9;
    border-radius: 4px; margin-bottom: 6px; font-weight: 600;
  }
  #ar-card p { font-size: 12px; line-height: 1.45; color: #cbd5e1; margin-bottom: 6px; }
  #ar-card .ar-meta {
    font-size: 11px; color: #94a3b8;
    border-top: 1px solid rgba(148,163,184,0.2); padding-top: 6px;
  }

  /* TOAST DE FEEDBACK RÁPIDO */
  #toast-msg {
    position: absolute; top: 54px; left: 50%; transform: translateX(-50%);
    background: rgba(15, 23, 42, 0.9); border: 1px solid #38bdf8;
    color: #f8fafc; padding: 7px 16px; border-radius: 8px;
    font-size: 12.5px; font-weight: 600; display: none; z-index: 25;
  }

  /* LETTERBOX DE CUTSCENE (CAFEZINHO) */
  .letterbox {
    position: absolute; left: 0; width: 100%; height: 0;
    background: #020617; z-index: 30; transition: height 0.45s ease;
  }
  #lb-top { top: 0; }
  #lb-bottom { bottom: 0; }
  #cutscene-caption {
    position: absolute; bottom: 22px; left: 50%; transform: translateX(-50%);
    color: #fef3c7; font-size: 15px; font-weight: 600;
    text-shadow: 0 2px 8px #000; z-index: 31; display: none;
    text-align: center; width: 85%;
  }

  /* PAINEL RETRÁTIL DE QUESTS & INFORMAÇÕES (PAUSA O JOGO) */
  #pause-modal {
    position: absolute; inset: 0;
    background: rgba(2, 6, 23, 0.84);
    backdrop-filter: blur(8px);
    display: none; align-items: center; justify-content: center;
    z-index: 40; padding: 18px;
  }
  .modal-box {
    width: 100%; max-width: 860px; max-height: 92%;
    background: #0f172a; border: 1px solid #38bdf8;
    border-radius: 14px; display: flex; flex-direction: column;
    overflow: hidden; box-shadow: 0 20px 50px rgba(0,0,0,0.65);
  }
  .modal-header {
    display: flex; justify-content: space-between; align-items: center;
    padding: 12px 18px; background: #1e293b; border-bottom: 1px solid #334155;
  }
  .modal-header h3 { color: #f8fafc; font-size: 16px; }
  .modal-tabs {
    display: flex; gap: 6px; padding: 10px 18px 0 18px;
    background: #0f172a; border-bottom: 1px solid #1e293b;
  }
  .tab-btn {
    background: #1e293b; color: #94a3b8; border: 1px solid #334155;
    border-bottom: none; padding: 7px 14px; border-radius: 8px 8px 0 0;
    font-size: 12.5px; font-weight: 600; cursor: pointer;
  }
  .tab-btn.active { background: #0f172a; color: #38bdf8; border-color: #38bdf8; }
  .modal-body {
    padding: 16px 18px; overflow-y: auto; color: #e2e8f0;
    font-size: 13px; line-height: 1.5; user-select: text;
  }
  .act-block {
    background: #1e293b; border: 1px solid #334155;
    border-radius: 8px; padding: 10px 14px; margin-bottom: 10px;
  }
  .act-block h4 { color: #38bdf8; font-size: 13.5px; margin-bottom: 6px; }
  .q-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 3px 0; border-bottom: 1px dashed rgba(148,163,184,0.15);
    font-size: 12.2px;
  }
  .q-badge {
    font-size: 11px; padding: 1px 7px; border-radius: 999px; font-weight: 700;
  }
  .q-done { background: rgba(34, 197, 94, 0.2); color: #4ade80; }
  .q-bonus { background: rgba(250, 204, 21, 0.2); color: #facc15; }
  .inv-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 10px; margin-top: 8px;
  }
  .inv-slot {
    background: #1e293b; border: 1px solid #334155;
    border-radius: 8px; padding: 10px; text-align: center;
  }
  .inv-slot.collected { border-color: #22c55e; background: rgba(20, 83, 45, 0.28); }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(-4px); } to { opacity: 1; transform: translateY(0); } }
</style>
</head>
<body>
<div id="game-wrapper">
  <canvas id="c"></canvas>

  <!-- HUD MINIMALISTA -->
  <div id="hud-top-left">
    <div class="hud-pill">
      <span>⚡ Energia</span>
      <div class="bar-bg"><div id="energy-fill"></div></div>
      <span id="buff-tag" style="color:#facc15; display:none;">☕ Cafeinado!</span>
    </div>
    <div class="hud-pill">
      <span>🎒 Inventário: <b id="inv-count" style="color:#38bdf8;">0/8</b></span>
      <span>•</span>
      <span>🪟 Janelas Quebradas: <b id="win-broken" style="color:#fb923c;">0/10</b></span>
    </div>
  </div>

  <div id="hud-top-right">
    <button class="hud-btn" id="btn-quests">📜 Quests & Info (Q)</button>
    <button class="hud-btn" id="btn-fullscreen">⛶ Tela Cheia</button>
  </div>

  <div id="toast-msg"></div>
  <div id="interact-prompt"></div>

  <!-- CARTÃO DE REALIDADE AUMENTADA -->
  <div id="ar-card">
    <span class="ar-tag">📡 REALIDADE AUMENTADA • BOUNDING BOX</span>
    <h4 id="ar-title">Objeto</h4>
    <p id="ar-desc">Descrição</p>
    <div class="ar-meta" id="ar-meta">Primitiva • Transformação</div>
  </div>

  <!-- BOTÕES DE AÇÃO -->
  <div id="hud-bottom-right">
    <button class="action-btn ar" id="btn-interact">🔍 O que é isso? [E]</button>
    <button class="action-btn atk" id="btn-attack">⚔️ Atacar [F]</button>
    <button class="action-btn" id="btn-jump">⬆️ Pular [Espaço]</button>
  </div>

  <!-- LETTERBOX CUTSCENE -->
  <div class="letterbox" id="lb-top"></div>
  <div class="letterbox" id="lb-bottom"></div>
  <div id="cutscene-caption">☕ Preparando um cafezinho expresso no Centro de Tecnologia... Energia restaurada!</div>

  <!-- PAINEL RETRÁTIL DE QUESTS E INFORMAÇÕES (PAUSA O JOGO) -->
  <div id="pause-modal">
    <div class="modal-box">
      <div class="modal-header">
        <h3>⏸️ JOGO PAUSADO — Diário de Quests & Informações (CG I)</h3>
        <button class="hud-btn" id="btn-resume" style="background:#0284c7; border-color:#38bdf8;">▶ Voltar ao Jogo (Q)</button>
      </div>
      <div class="modal-tabs">
        <button class="tab-btn active" data-tab="tab-quests">🎮 Quests da Apresentação (Atos I–VIII)</button>
        <button class="tab-btn" data-tab="tab-inv">🎒 Inventário & Mesas (8 Itens)</button>
        <button class="tab-btn" data-tab="tab-help">🧭 Controles & Mecânicas</button>
      </div>
      <div class="modal-body" id="tab-quests">
        <div class="act-block">
          <h4>🏰 ATO I — A Construção do Mundo</h4>
          <div class="q-row"><span>🎯 Sub-quest 1.1 — "Tudo Tem Um Propósito" (Bloco 734 CT Coerente)</span><span class="q-badge q-done">✔ Concluída</span></div>
          <div class="q-row"><span>🎯 Sub-quest 1.2 — "O Primeiro Octante" (Todos vértices X, Y, Z &ge; 0)</span><span class="q-badge q-done">✔ Concluída</span></div>
          <div class="q-row"><span>🎯 Quest 2 — "O Zoo dos Objetos" (Esfera, Cilindro, Cone e Malha)</span><span class="q-badge q-done">✔ Concluída</span></div>
          <div class="q-row"><span>🎯 Sub-quest 3.1 — "Diversidade" (&ge; 4 materiais distintos aplicados)</span><span class="q-badge q-done">✔ Concluída</span></div>
          <div class="q-row"><span>🎯 Sub-quest 3.2 — "A Pele Pintada" (Texturas procedurais: Asfalto, Calçada, Água, Placa 734)</span><span class="q-badge q-done">✔ Concluída</span></div>
        </div>
        <div class="act-block">
          <h4>⚔️ ATO II — As Transformações Mágicas</h4>
          <div class="q-row"><span>🎯 Quest 4 — "O Ritual da Translação" (Matrizes 4×4)</span><span class="q-badge q-done">✔ Concluída</span></div>
          <div class="q-row"><span>🎯 Sub-quest 5.1 — "O Eixo Sagrado" (Rotação X, Y e Z)</span><span class="q-badge q-done">✔ Concluída</span></div>
          <div class="q-row"><span>🎯 Sub-quest 5.2 — "O Eixo Livre" (Caminho B: Quatérnios nas copas/palmeira)</span><span class="q-badge q-done">✔ Concluída</span></div>
          <div class="q-row"><span>🎯 Quest 6 — "O Encantamento da Escala"</span><span class="q-badge q-done">✔ Concluída</span></div>
          <div class="q-row"><span>✂️ Quest Bônus 6.1 — "A Distorção" (Cisalhamento XY no Abrigo Metálico)</span><span class="q-badge q-bonus">+0.5 pts</span></div>
          <div class="q-row"><span>🪞 Quest Bônus 6.2 — "O Reflexo Perfeito" (Espelho em plano arbitrário nos Cones)</span><span class="q-badge q-bonus">+0.5 pts</span></div>
        </div>
        <div class="act-block">
          <h4>🔦 ATO III — A Luz e a Escuridão</h4>
          <div class="q-row"><span>💡 Sub-quest 7.1 — "A Chama Pontual" (4 Postes com atenuação quadrática)</span><span class="q-badge q-done">✔ Concluída</span></div>
          <div class="q-row"><span>🌫️ Sub-quest 7.2 — "O Brilho Ambiente" (Componente Ia)</span><span class="q-badge q-done">✔ Concluída</span></div>
          <div class="q-row"><span>🔦 Sub-quest Bônus 7.3 — "O Holofote" (Spot na Escultura Cinética + Spots Submersos)</span><span class="q-badge q-bonus">+1.0 pts</span></div>
          <div class="q-row"><span>☀️ Sub-quest Bônus 7.4 — "O Sol Direcional" (Iluminação Diurna com Sombras)</span><span class="q-badge q-bonus">+0.5 pts</span></div>
        </div>
        <div class="act-block">
          <h4>📷 ATOS IV & V — A Câmera do Observador & As Projeções</h4>
          <div class="q-row"><span>👁️ Quests 8 & 9 — Eye, At, Up, Distância Focal (d) e Janela (xmin..ymax)</span><span class="q-badge q-done">✔ Concluída</span></div>
          <div class="q-row"><span>🔭 Quest 10 — "A Perspectiva" (Presets Zoom Out e Zoom In)</span><span class="q-badge q-done">✔ Concluída</span></div>
          <div class="q-row"><span>1️⃣2️⃣3️⃣ Quests Bônus — "Os Pontos de Fuga" (1, 2 e 3+ Pontos de Fuga)</span><span class="q-badge q-bonus">+1.5 pts</span></div>
          <div class="q-row"><span>📏📐 Quests Bônus — "Projeções Alternativas" (Ortográfica e Oblíqua)</span><span class="q-badge q-bonus">+1.0 pts</span></div>
        </div>
        <div class="act-block">
          <h4>🌑🖱️🖼️ ATOS VI, VII & VIII — Sombras, Interatividade e Mosaico 500×500</h4>
          <div class="q-row"><span>🌑 Quest 11 — "A Sombra" (Sombras projetadas no Ray Casting)</span><span class="q-badge q-done">✔ Concluída</span></div>
          <div class="q-row"><span>🎯 Quest 12 — "O Toque" (Pick por clique no Render + Picking AR 3D)</span><span class="q-badge q-done">✔ Concluída</span></div>
          <div class="q-row"><span>🖥️ Quest Bônus — "A Interface Gráfica" (Streamlit + Motor WebGL Interativo)</span><span class="q-badge q-bonus">+1.0 pts</span></div>
          <div class="q-row"><span>🖼️ Quest 13 — "O Mosaico de 500×500" (Ray Casting >= 500×500 px)</span><span class="q-badge q-done">✔ Concluída</span></div>
          <div class="q-row"><span>🌟 Quest Lendária — "Criatividade e Beleza" (Cáusticas por Refração + Mini-Game)</span><span class="q-badge q-bonus">+1.0 pts</span></div>
        </div>
      </div>

      <div class="modal-body" id="tab-inv" style="display:none;">
        <p>Aproxime-se das <b>Mesas de Trabalho</b> (sob o Abrigo Metálico e na Praça do Bloco 734) e pressione <b>[E]</b> para coletar suas entregas acadêmicas:</p>
        <div class="inv-grid" id="inv-grid-container"></div>
      </div>

      <div class="modal-body" id="tab-help" style="display:none;">
        <div class="act-block">
          <h4>🎮 Controles do Personagem</h4>
          <p>• <b>W A S D / Setas:</b> Caminhar e correr pelo campus do CT.<br/>
             • <b>Arrastar Mouse (ou Q/E câmera):</b> Girar a câmera orbital ao redor do personagem.<br/>
             • <b>Espaço:</b> Pular (e subir pela escada lateral em X=59.35, Z=24.0).<br/>
             • <b>E (ou 🔍 O que é isso?):</b> Ativa <b>Realidade Aumentada</b> com Bounding Box 3D no objeto próximo, coleta atividades nas mesas ou prepara um <b>cafezinho</b> na Máquina de Café!<br/>
             • <b>F (ou ⚔️ Atacar):</b> Golpeia janelas de vidro fumê da fachada arqueada (3 golpes quebram a janela e <b>derramam asfalto animado</b> na parede!).<br/>
             • <b>Descer do Telhado:</b> Use a <b>Escada Lateral</b>, ou salte caindo sobre a <b>copa de uma Árvore</b> ou dentro da <b>Fonte</b> para aterrissagem amortecida.<br/>
             • <b>Q ou Tab:</b> Abre/fecha este painel de Quests e pausa o jogo.</p>
        </div>
      </div>
    </div>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
(function() {
  const MODO_NOITE = __MODO_NOITE__;
  const wrapper = document.getElementById('game-wrapper');
  const canvas = document.getElementById('c');

  // Renderer WebGL
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(wrapper.clientWidth, wrapper.clientHeight);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(MODO_NOITE ? 0x060b19 : 0x8ec5fc);
  scene.fog = new THREE.FogExp2(MODO_NOITE ? 0x060b19 : 0x93c5fd, MODO_NOITE ? 0.014 : 0.0065);

  const camera = new THREE.PerspectiveCamera(58, wrapper.clientWidth / wrapper.clientHeight, 0.1, 250);

  // Iluminação
  const hemiLight = new THREE.HemisphereLight(
    MODO_NOITE ? 0x1e293b : 0xe0f2fe,
    MODO_NOITE ? 0x0f172a : 0x475569,
    MODO_NOITE ? 0.45 : 0.85
  );
  scene.add(hemiLight);

  const dirLight = new THREE.DirectionalLight(
    MODO_NOITE ? 0x60a5fa : 0xfffbeb,
    MODO_NOITE ? 0.25 : 1.15
  );
  dirLight.position.set(12, 34, 8);
  dirLight.castShadow = true;
  dirLight.shadow.mapSize.set(2048, 2048);
  dirLight.shadow.camera.near = 2;
  dirLight.shadow.camera.far = 110;
  const dCam = 42;
  dirLight.shadow.camera.left = -dCam;
  dirLight.shadow.camera.right = dCam;
  dirLight.shadow.camera.top = dCam;
  dirLight.shadow.camera.bottom = -dCam;
  scene.add(dirLight);

  // Materiais Compartilhados (Leves)
  const matAsfalto = new THREE.MeshStandardMaterial({ color: 0x262930, roughness: 0.9 });
  const matFaixa = new THREE.MeshBasicMaterial({ color: 0xeab308 });
  const matCalcada = new THREE.MeshStandardMaterial({ color: 0xb0aea8, roughness: 0.85 });
  const matAlvenaria = new THREE.MeshStandardMaterial({ color: 0xf1f5f9, roughness: 0.65 });
  const matVidro = new THREE.MeshStandardMaterial({
    color: 0x0f172a, roughness: 0.12, metalness: 0.8, transparent: true, opacity: 0.86
  });
  const matEsquadria = new THREE.MeshStandardMaterial({ color: 0x1e293b, metalness: 0.6, roughness: 0.4 });
  const matAcoBranco = new THREE.MeshStandardMaterial({ color: 0xf8fafc, metalness: 0.35, roughness: 0.25 });
  const matConcreto = new THREE.MeshStandardMaterial({ color: 0x64748b, roughness: 0.8 });
  const matTronco = new THREE.MeshStandardMaterial({ color: 0x5c3a21, roughness: 0.9 });
  const matFolha1 = new THREE.MeshStandardMaterial({ color: 0x15803d, roughness: 0.75 });
  const matFolha2 = new THREE.MeshStandardMaterial({ color: 0x22c55e, roughness: 0.7 });
  const matAgua = new THREE.MeshStandardMaterial({
    color: 0x0ea5e9, roughness: 0.15, metalness: 0.4, transparent: true, opacity: 0.82
  });
  const matGranito = new THREE.MeshStandardMaterial({ color: 0x334155, roughness: 0.6 });
  const matCone = new THREE.MeshStandardMaterial({ color: 0xf97316, roughness: 0.45 });
  const matMetalGalv = new THREE.MeshStandardMaterial({ color: 0x94a3b8, metalness: 0.7, roughness: 0.3, side: THREE.DoubleSide });
  const matMadeira = new THREE.MeshStandardMaterial({ color: 0x854d27, roughness: 0.65 });
  const matPapel = new THREE.MeshStandardMaterial({ color: 0xf8fafc, roughness: 0.5 });
  const matTelaPC = new THREE.MeshStandardMaterial({ color: 0x0284c7, emissive: 0x0369a1, emissiveIntensity: 0.6 });

  function addBox(w, h, d, mat, x, y, z, ry = 0, cast = true) {
    const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
    m.position.set(x, y, z);
    if (ry) m.rotation.y = ry;
    m.castShadow = cast;
    m.receiveShadow = true;
    scene.add(m);
    return m;
  }

  // Listas de Interação AR, Janelas Quebráveis, Asfalto Derramado e Coletáveis
  const arObjects = [];
  function registrarAR(nome, tag, desc, meta, center, size, extra = {}) {
    arObjects.push({
      nome, tag, desc, meta,
      center: new THREE.Vector3(...center),
      size: new THREE.Vector3(...size),
      ...extra
    });
  }

  // 1. PISO (VIA ASFÁLTICA + CALÇADA)
  addBox(66, 0.1, 14, matAsfalto, 35, 0.05, 9, 0, false);
  addBox(64, 0.02, 0.14, matFaixa, 35, 0.11, 11.5, 0, false);
  addBox(64, 0.02, 0.14, matFaixa, 35, 0.11, 11.9, 0, false);
  addBox(66, 0.2, 44, matCalcada, 35, 0.10, 38, 0, false);

  // 2. BLOCO 734 (ARQUEADO R=22m + ALAS RETANGULARES + PLACA 734)
  addBox(22, 6.6, 33, matAlvenaria, 48, 3.3, 36.5);
  addBox(22, 6.6, 11, matAlvenaria, 26, 3.3, 47.5);
  const arcGeo = new THREE.CylinderGeometry(22, 22, 6.6, 48, 1, false, Math.PI, Math.PI * 0.5);
  const arcMesh = new THREE.Mesh(arcGeo, matAlvenaria);
  arcMesh.position.set(37, 3.3, 42);
  arcMesh.castShadow = true;
  arcMesh.receiveShadow = true;
  scene.add(arcMesh);

  // Placa 734 na Platibanda
  const pCanvas = document.createElement('canvas');
  pCanvas.width = 256; pCanvas.height = 96;
  const pCtx = pCanvas.getContext('2d');
  pCtx.fillStyle = '#f8fafc'; pCtx.fillRect(0, 0, 256, 96);
  pCtx.strokeStyle = '#1e293b'; pCtx.lineWidth = 6; pCtx.strokeRect(3, 3, 250, 90);
  pCtx.fillStyle = '#0f172a'; pCtx.font = 'bold 62px sans-serif';
  pCtx.textAlign = 'center'; pCtx.textBaseline = 'middle';
  pCtx.fillText('734', 128, 52);
  const placaTex = new THREE.CanvasTexture(pCanvas);
  const angPlaca = 0.14 * (Math.PI / 2);
  const placaMesh = new THREE.Mesh(
    new THREE.BoxGeometry(3.4, 1.15, 0.18),
    new THREE.MeshStandardMaterial({ map: placaTex })
  );
  placaMesh.position.set(37 - 22.08 * Math.cos(angPlaca), 5.72, 42 - 22.08 * Math.sin(angPlaca));
  placaMesh.rotation.y = Math.PI / 2 - angPlaca;
  scene.add(placaMesh);

  registrarAR(
    'Bloco 734 (CT / Asfalto)',
    'EDIFICAÇÃO PRINCIPAL • CSG ANALÍTICO',
    'Edifício sede com fachada arqueada tangencial (R=22m, H=6.6m), 9 janelas quebráveis em vidro fumê e platibanda nº 734.',
    'Primitivas: Prisma + Cilindro R=22m • Primeiro Octante',
    [37, 3.3, 36.5], [44, 6.8, 33]
  );

  // 2B. JANELAS QUEBRÁVEIS NA FACHADA ARQUEADA + SISTEMA DE ASFALTO DERRAMADO
  const breakableWindows = [];
  const asphaltSpills = [];
  const glassShards = [];
  const matAsfaltoLiquido = new THREE.MeshStandardMaterial({
    color: 0x090b10, roughness: 0.18, metalness: 0.35
  });

  function criarJanelaCurva(nome, sArco, yCentro, larg, alt, isPorta = false) {
    const ang = sArco * (Math.PI / 2);
    const r = 22.06;
    const wx = 37 - r * Math.cos(ang);
    const wz = 42 - r * Math.sin(ang);
    const rotY = Math.PI / 2 - ang;

    const grp = new THREE.Group();
    grp.position.set(wx, yCentro, wz);
    grp.rotation.y = rotY;

    // Moldura esquadria
    const frame = new THREE.Mesh(new THREE.BoxGeometry(larg + 0.18, alt + 0.18, 0.12), matEsquadria);
    grp.add(frame);

    // Vidro quebrável
    const glassMat = matVidro.clone();
    const glass = new THREE.Mesh(new THREE.BoxGeometry(larg, alt, 0.16), glassMat);
    grp.add(glass);

    // Barra de durabilidade flutuante (aparece ao bater)
    const hpBarBg = new THREE.Mesh(
      new THREE.BoxGeometry(1.4, 0.14, 0.05),
      new THREE.MeshBasicMaterial({ color: 0x0f172a })
    );
    hpBarBg.position.set(0, alt * 0.5 + 0.32, 0.12);
    hpBarBg.visible = false;
    grp.add(hpBarBg);

    const hpBarFill = new THREE.Mesh(
      new THREE.BoxGeometry(1.34, 0.10, 0.06),
      new THREE.MeshBasicMaterial({ color: 0x22c55e })
    );
    hpBarFill.position.set(0, alt * 0.5 + 0.32, 0.13);
    hpBarFill.visible = false;
    grp.add(hpBarFill);

    scene.add(grp);

    const winObj = {
      nome, grp, glass, glassMat, hpBarBg, hpBarFill,
      pos: new THREE.Vector3(wx, yCentro, wz),
      normal: new THREE.Vector3(-Math.cos(ang), 0, -Math.sin(ang)),
      rotY, larg, alt, isPorta,
      hp: 3, maxHp: 3, broken: false, hideBarTimer: 0
    };
    breakableWindows.push(winObj);
  }

  criarJanelaCurva('Porta Principal de Vidro (Bloco 734)', 0.13, 1.48, 2.5, 2.55, true);
  for (let i = 0; i < 9; i++) {
    const sMid = 0.24 + ((i + 0.5) / 9.0) * 0.72;
    criarJanelaCurva(`Janela Fachada Curva #${i + 1}`, sMid, 2.30, 1.75, 1.88, false);
  }

  function dispararDerramamentoAsfalto(winObj) {
    // Cria cascata vertical de asfalto escorrendo da janela + poça na calçada
    const basePos = winObj.pos.clone().addScaledVector(winObj.normal, 0.16);
    const streamMesh = new THREE.Mesh(
      new THREE.BoxGeometry(winObj.larg * 0.82, 0.1, 0.18),
      matAsfaltoLiquido
    );
    streamMesh.position.set(basePos.x, winObj.pos.y - winObj.alt * 0.35, basePos.z);
    streamMesh.rotation.y = winObj.rotY;
    scene.add(streamMesh);

    const puddlePos = winObj.pos.clone().addScaledVector(winObj.normal, 1.15);
    const puddleMesh = new THREE.Mesh(
      new THREE.CylinderGeometry(0.1, 0.1, 0.04, 24),
      matAsfaltoLiquido
    );
    puddleMesh.position.set(puddlePos.x, 0.22, puddlePos.z);
    scene.add(puddleMesh);

    asphaltSpills.push({
      streamMesh,
      puddleMesh,
      topY: winObj.pos.y - winObj.alt * 0.35,
      progress: 0.0
    });
  }

  function spawnEstilhacos(pos, corHex, qtd = 18) {
    const geo = new THREE.BoxGeometry(0.14, 0.14, 0.05);
    const mat = new THREE.MeshBasicMaterial({ color: corHex });
    for (let i = 0; i < qtd; i++) {
      const m = new THREE.Mesh(geo, mat);
      m.position.copy(pos).add(new THREE.Vector3((Math.random()-0.5)*1.2, (Math.random()-0.5)*1.2, (Math.random()-0.5)*0.6));
      scene.add(m);
      glassShards.push({
        mesh: m,
        vel: new THREE.Vector3((Math.random()-0.5)*5, Math.random()*4 + 1.5, (Math.random()-0.5)*5),
        life: 1.4
      });
    }
  }

  // 3. ESPELHO D'ÁGUA / FONTE FRONTAL
  addBox(15.3, 0.34, 0.28, matGranito, 27.0, 0.37, 16.8);
  addBox(15.3, 0.34, 0.28, matGranito, 27.0, 0.37, 22.8);
  addBox(0.28, 0.34, 6.0, matGranito, 19.5, 0.37, 19.8);
  addBox(0.28, 0.34, 6.0, matGranito, 34.5, 0.37, 19.8);
  const aguaMesh = addBox(14.85, 0.08, 5.85, matAgua, 27.0, 0.38, 19.8, 0, false);

  [22.0, 27.0, 32.0].forEach((sx) => {
    const sp = new THREE.Mesh(
      new THREE.SphereGeometry(0.16, 12, 12),
      new THREE.MeshBasicMaterial({ color: 0x67e8f9 })
    );
    sp.position.set(sx, 0.32, 17.12);
    scene.add(sp);
  });
  registrarAR(
    "Espelho d'Água (Fonte Cáustica)",
    "REFRAÇÃO n=1.333 • ZONA DE POUSO SEGURO",
    "Lâmina d'água com 3 spots submersos que projetam filamentos de cáustica na fachada. Também amortece saltos do telhado!",
    "Cubos + Esferas • Textura Procedural Ondulatória",
    [27.0, 0.42, 19.8], [15.4, 0.6, 6.2]
  );

  // 4. ESCULTURA CINÉTICA EM AÇO (5 LOSANGOS ROTACIONADOS A 45°)
  const escGrp = new THREE.Group();
  escGrp.position.set(15.0, 0.2, 22.0);
  escGrp.rotation.y = THREE.MathUtils.degToRad(-24);
  const baseEsc = new THREE.Mesh(new THREE.BoxGeometry(3.2, 0.28, 3.4), matConcreto);
  baseEsc.position.y = 0.14;
  baseEsc.castShadow = true;
  escGrp.add(baseEsc);

  for (let k = 0; k < 5; k++) {
    const fc = k - 2;
    const Lk = 2.35 * (1.0 - 0.055 * fc);
    const rk = Lk / Math.sqrt(2);
    const qGrp = new THREE.Group();
    qGrp.position.set(fc * 0.22, 0.28 + rk + 0.05, -1.16 + k * 0.58);
    qGrp.rotation.z = Math.PI / 4;
    const esp = 0.14, prof = 0.16;
    const b1 = new THREE.Mesh(new THREE.BoxGeometry(Lk + esp, esp, prof), matAcoBranco); b1.position.y = Lk/2;
    const b2 = new THREE.Mesh(new THREE.BoxGeometry(Lk + esp, esp, prof), matAcoBranco); b2.position.y = -Lk/2;
    const b3 = new THREE.Mesh(new THREE.BoxGeometry(esp, Lk + esp, prof), matAcoBranco); b3.position.x = Lk/2;
    const b4 = new THREE.Mesh(new THREE.BoxGeometry(esp, Lk + esp, prof), matAcoBranco); b4.position.x = -Lk/2;
    [b1, b2, b3, b4].forEach(b => { b.castShadow = true; qGrp.add(b); });
    escGrp.add(qGrp);
  }
  scene.add(escGrp);
  registrarAR(
    "Escultura Cinética em Aço",
    "COMPOSIÇÃO PARAMÉTRICA • 5 LOSANGOS",
    "Cinco quadros metálicos rotacionados a 45°, escalonados progressivamente e orientados a -24° em Y sobre base de concreto.",
    "Translação · Rotação Z(45°) & Y(-24°) · Escala",
    [15.0, 2.0, 22.0], [3.6, 3.8, 3.8]
  );

  // 5. ESCADA DE ACESSO AO TELHADO
  addBox(0.55, 6.6, 1.1, matEsquadria, 59.35, 3.4, 24.0);
  for (let deg = 0.6; deg <= 6.4; deg += 0.45) {
    addBox(0.65, 0.06, 0.95, matMetalGalv, 59.35, deg, 24.0);
  }
  registrarAR(
    "Escada de Acesso ao Telhado",
    "TRANSPORTE VERTICAL • Y=0.2m → Y=6.8m",
    "Escada marinheiro lateral (X=59.35, Z=24.0). Aproxime-se e pressione Espaço/W para subir ou S para descer do telhado.",
    "Cubos Escalonados • Acesso Cobertura Bloco 734",
    [59.35, 3.4, 24.0], [1.2, 6.8, 1.5]
  );

  // 6. ÁRVORES + AVE PREDADORA
  const arvoresCoords = [
    { nome: "Árvore Bosque Esq.", x: 10.5, z: 28.5, esc: 1.15 },
    { nome: "Árvore Fundo Esq.", x: 11.5, z: 38.0, esc: 1.25 },
    { nome: "Árvore Praça Dir.", x: 37.5, z: 17.5, esc: 0.90 }
  ];
  arvoresCoords.forEach((arv) => {
    const tMesh = new THREE.Mesh(
      new THREE.CylinderGeometry(0.28 * arv.esc, 0.42 * arv.esc, 3.2 * arv.esc, 12),
      matTronco
    );
    tMesh.position.set(arv.x, 0.2 + 1.6 * arv.esc, arv.z);
    tMesh.castShadow = true;
    scene.add(tMesh);

    const offsets = [
      [0, 3.6, 0, 1.65, matFolha1],
      [-1.1, 3.2, 0.6, 1.25, matFolha2],
      [1.1, 3.3, -0.5, 1.30, matFolha2],
      [0.3, 4.5, 0.4, 1.20, matFolha1]
    ];
    offsets.forEach(([ox, oy, oz, r, m]) => {
      const cMesh = new THREE.Mesh(new THREE.SphereGeometry(r * arv.esc, 14, 12), m);
      cMesh.position.set(arv.x + ox * arv.esc, oy * arv.esc, arv.z + oz * arv.esc);
      cMesh.castShadow = true;
      scene.add(cMesh);
    });
    registrarAR(
      arv.nome,
      "VEGETAÇÃO • QUATÉRNIOS • POUSO SEGURO",
      "Tronco cilíndrico e copa esférica composta por quatérnios. Saltar do telhado sobre sua copa amortece a queda!",
      "Cilindro + 4 Esferas • Zona Segura de Descida",
      [arv.x, 2.8 * arv.esc, arv.z], [3.4 * arv.esc, 5.4 * arv.esc, 3.4 * arv.esc]
    );
  });

  // Ave Predadora na Árvore do Fundo (X=11.5, Z=38.0)
  const aveGrp = new THREE.Group();
  const aveHome = new THREE.Vector3(11.5, 5.3, 38.0);
  aveGrp.position.copy(aveHome);
  const matAve = new THREE.MeshStandardMaterial({ color: 0x312e81, roughness: 0.5 });
  const aveCorpo = new THREE.Mesh(new THREE.ConeGeometry(0.32, 1.1, 10), matAve);
  aveCorpo.rotation.x = Math.PI / 2;
  const asaE = new THREE.Mesh(new THREE.BoxGeometry(1.3, 0.08, 0.45), matAve); asaE.position.x = -0.65;
  const asaD = new THREE.Mesh(new THREE.BoxGeometry(1.3, 0.08, 0.45), matAve); asaD.position.x = 0.65;
  aveGrp.add(aveCorpo, asaE, asaD);
  scene.add(aveGrp);
  registrarAR(
    "Ave Predadora Territorial",
    "FAUNA DO CAMPUS • ALERTA DE PROXIMIDADE",
    "Ave empoleirada na Árvore do Fundo (X=11.5, Z=38.0). Protege seu ninho e avança se você se aproximar demais!",
    "Cone + Cubos • Comportamento Reativo",
    [11.5, 5.3, 38.0], [2.4, 1.2, 1.6]
  );

  // 7. CONES DE SINALIZAÇÃO + ESPELHADOS
  [[35.3, 16.2], [35.9, 15.6], [36.5, 16.9], [37.1, 16.3]].forEach(([cx, cz]) => {
    const c = new THREE.Mesh(new THREE.ConeGeometry(0.32, 0.95, 14), matCone);
    c.position.set(cx, 0.2 + 0.475, cz);
    c.castShadow = true;
    scene.add(c);
  });
  registrarAR(
    "Cones de Sinalização & Reflexo",
    "QUEST BÔNUS 6.2 • ESPELHO EM PLANO ARBITRÁRIO",
    "Cones viários duplicados geometricamente através de matriz de reflexão especular em relação a um plano oblíquo.",
    "Primitiva Cone + Matriz de Espelho (I - 2nnᵀ)",
    [36.2, 0.7, 16.2], [2.2, 1.1, 2.0]
  );

  // 8. ABRIGO METÁLICO CISALHADO + PILARES
  const abrigoGeo = new THREE.BoxGeometry(4.5, 0.16, 3.2);
  const abrigoMesh = new THREE.Mesh(abrigoGeo, matMetalGalv);
  abrigoMesh.position.set(46.8, 2.6, 18.8);
  abrigoMesh.rotation.z = -0.09;
  abrigoMesh.castShadow = true;
  scene.add(abrigoMesh);
  [[44.7, 17.4], [48.8, 17.4], [44.7, 20.2], [48.8, 20.2]].forEach(([px, pz]) => {
    const pil = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.08, 2.4, 10), matEsquadria);
    pil.position.set(px, 1.4, pz);
    pil.castShadow = true;
    scene.add(pil);
  });
  registrarAR(
    "Abrigo Metálico (Malha Cisalhada)",
    "QUEST BÔNUS 6.1 • CISALHAMENTO XY (Sh=0.15)",
    "Cobertura em malha poligonal de aço galvanizado deformada por cisalhamento XY sobre pilares cilíndricos.",
    "Malha 12 Triângulos + Matriz Cisalhamento",
    [46.8, 1.5, 18.8], [4.6, 2.8, 3.4]
  );

  // 9. POSTES DE ILUMINAÇÃO PÚBLICA
  [[9.5, 21.0, 0.6], [18.5, 16.2, 0.4], [41.0, 16.5, 0.0], [54.5, 17.5, -0.5]].forEach(([px, pz, ox], idx) => {
    const p = new THREE.Mesh(new THREE.CylinderGeometry(0.10, 0.14, 6.2, 10), matEsquadria);
    p.position.set(px, 3.3, pz);
    p.castShadow = true;
    scene.add(p);
    const lum = new THREE.Mesh(
      new THREE.SphereGeometry(0.28, 12, 12),
      new THREE.MeshBasicMaterial({ color: MODO_NOITE ? 0xfef08a : 0xf8fafc })
    );
    lum.position.set(px + ox, 6.3, pz + 0.4);
    scene.add(lum);
    if (MODO_NOITE) {
      const pt = new THREE.PointLight(0xfde68a, 1.4, 24);
      pt.position.set(px + ox, 6.0, pz + 0.4);
      scene.add(pt);
    }
  });

  // 10. ELEMENTOS LEVES ESTILO ROBLOX: CARROS, ESTUDANTES, NPC E PALMEIRA
  function criarVeiculoRoblox(nome, x, y, z, rotDeg, corHex, desc) {
    const grp = new THREE.Group();
    grp.position.set(x, y, z);
    grp.rotation.y = THREE.MathUtils.degToRad(rotDeg);

    const matCarro = new THREE.MeshStandardMaterial({ color: corHex, metalness: 0.4, roughness: 0.25 });
    const matPneu = new THREE.MeshStandardMaterial({ color: 0x111827, roughness: 0.9 });
    const chassi = new THREE.Mesh(new THREE.BoxGeometry(2.0, 0.65, 4.3), matCarro);
    chassi.position.y = 0.55; chassi.castShadow = true;
    const cabine = new THREE.Mesh(new THREE.BoxGeometry(1.75, 0.54, 2.2), matVidro);
    cabine.position.set(0, 1.12, -0.15); cabine.castShadow = true;
    grp.add(chassi, cabine);

    [[-1.02, 0.36, 1.35], [1.02, 0.36, 1.35], [-1.02, 0.36, -1.35], [1.02, 0.36, -1.35]].forEach(([rx, ry, rz]) => {
      const roda = new THREE.Mesh(new THREE.CylinderGeometry(0.36, 0.36, 0.22, 14), matPneu);
      roda.rotation.z = Math.PI / 2;
      roda.position.set(rx, ry, rz);
      grp.add(roda);
    });
    [-0.65, 0.65].forEach((fx) => {
      const f = new THREE.Mesh(new THREE.SphereGeometry(0.14, 8, 8), new THREE.MeshBasicMaterial({ color: 0xfef9c3 }));
      f.position.set(fx, 0.62, 2.15);
      const l = new THREE.Mesh(new THREE.SphereGeometry(0.14, 8, 8), new THREE.MeshBasicMaterial({ color: 0xef4444 }));
      l.position.set(fx, 0.62, -2.15);
      grp.add(f, l);
    });
    scene.add(grp);
    registrarAR(nome, "VEÍCULO ESTILO ROBLOX • PRIMITIVAS PURAS", desc, "Cubos + Cilindros + Esferas (Zero Triângulos Pesados)", [x, y + 0.8, z], [4.5, 1.7, 2.3]);
  }

  criarVeiculoRoblox("SUV Prata na Via CT", 23.0, 0.1, 7.2, 90, 0xcbd5e1, "Veículo utilitário modelado com primitivas analíticas leves para máximo desempenho.");
  criarVeiculoRoblox("Carro Esportivo Vermelho", 39.0, 0.1, 10.6, -90, 0xdc2626, "Esportivo vermelho na via frontal do CT construído com geometria pura.");

  function criarAvatarRoblox(nome, x, y, z, rotDeg, corTorso, corCabeca, corCalca, esc = 1.0, desc = "") {
    const grp = new THREE.Group();
    grp.position.set(x, y, z);
    grp.rotation.y = THREE.MathUtils.degToRad(rotDeg);
    grp.scale.setScalar(esc);

    const mTorso = new THREE.MeshStandardMaterial({ color: corTorso, roughness: 0.6 });
    const mCab = new THREE.MeshStandardMaterial({ color: corCabeca, roughness: 0.5 });
    const mCalca = new THREE.MeshStandardMaterial({ color: corCalca, roughness: 0.8 });

    const pE = new THREE.Mesh(new THREE.BoxGeometry(0.32, 0.76, 0.34), mCalca); pE.position.set(-0.20, 0.38, 0);
    const pD = new THREE.Mesh(new THREE.BoxGeometry(0.32, 0.76, 0.34), mCalca); pD.position.set(0.20, 0.38, 0);
    const torso = new THREE.Mesh(new THREE.BoxGeometry(0.76, 0.84, 0.40), mTorso); torso.position.set(0, 1.16, 0);
    const cab = new THREE.Mesh(new THREE.BoxGeometry(0.48, 0.48, 0.48), mCab); cab.position.set(0, 1.85, 0);
    const bE = new THREE.Mesh(new THREE.BoxGeometry(0.26, 0.78, 0.28), mCab); bE.position.set(-0.54, 1.15, 0);
    const bD = new THREE.Mesh(new THREE.BoxGeometry(0.26, 0.78, 0.28), mCab); bD.position.set(0.54, 1.15, 0);
    [pE, pD, torso, cab, bE, bD].forEach(part => { part.castShadow = true; grp.add(part); });
    scene.add(grp);
    if (desc) {
      registrarAR(nome, "PERSONAGEM ARTICULADO • HIERARQUIA 3D", desc, "6 Cubos Hierárquicos · Transformações Locais", [x, y + 1.0, z], [1.1, 2.2, 1.1]);
    }
    return { grp, bE, bD, pE, pD };
  }

  criarAvatarRoblox("Estudante Casual", 18.6, 0.2, 18.8, -35, 0x2563eb, 0xfacc15, 0x1e293b, 0.95, "Estudante de Engenharia observando a Escultura Cinética e a Fonte.");
  criarAvatarRoblox("Estudante Vestido", 19.6, 0.2, 18.2, -55, 0xea580c, 0xfacc15, 0x1e293b, 0.92, "Estudante na praça do Bloco 734 conversando sobre Computação Gráfica.");
  const npcMisterioso = criarAvatarRoblox("NPC Misterioso Parado", 43.0, 0.2, 19.5, 180, 0x0f172a, 0xe2e8f0, 0x020617, 1.02, "Figura silenciosa próxima ao abrigo metálico. Melhor não encostar!");

  // Palmeira Estilizada (X=42.5, Z=17.5)
  [[0, 0.2, 0, 0.24, 1.6, 0], [0.05, 1.75, 0, 0.20, 1.6, -0.07], [0.15, 3.3, 0, 0.16, 1.5, -0.12]].forEach(([ox, oy, oz, r, h, rz]) => {
    const seg = new THREE.Mesh(new THREE.CylinderGeometry(r * 0.85, r, h, 10), matTronco);
    seg.position.set(42.5 + ox, oy + h * 0.5, 17.5 + oz);
    seg.rotation.z = rz;
    seg.castShadow = true;
    scene.add(seg);
  });
  for (let k = 0; k < 6; k++) {
    const ang = k * (Math.PI * 2 / 6);
    const palma = new THREE.Mesh(new THREE.ConeGeometry(0.42, 2.2, 8), matFolha2);
    palma.position.set(42.74 + Math.cos(ang) * 0.9, 4.65, 17.5 + Math.sin(ang) * 0.9);
    palma.rotation.z = -Math.cos(ang) * 1.15;
    palma.rotation.x = Math.sin(ang) * 1.15;
    palma.castShadow = true;
    scene.add(palma);
  }
  registrarAR(
    "Palmeira Tropical CT",
    "VEGETAÇÃO LEVE • CILINDROS & CONES",
    "Palmeira estilizada criada com tronco segmentado cilíndrico e 6 folhas cônicas orientadas por quatérnios.",
    "3 Cilindros + 6 Cones (Zero Malha .OBJ)",
    [42.6, 2.6, 17.5], [2.8, 5.2, 2.8]
  );

  // 11. MESAS DE TRABALHO (PCs e PAPÉIS) + ITENS COLETÁVEIS DE INVENTÁRIO
  const mesasConfig = [
    { nome: "Mesa Trabalho 1 (PC • Atividades 1–3)", x: 45.6, z: 18.2, tipo: "pc", desc: "Bancada com computador para realizar e coletar as Atividades 1, 2 e 3." },
    { nome: "Mesa Trabalho 2 (Papéis • Atividades 4–5)", x: 47.8, z: 18.2, tipo: "papel", desc: "Mesa de estudo com folhas técnicas das Atividades 4 e 5." },
    { nome: "Mesa Avaliações (Papéis • AP1 e AP2)", x: 45.6, z: 19.6, tipo: "papel", desc: "Bancada de entrega das Avaliações Parciais AP1 e AP2." },
    { nome: "Estação Trabalho Final (PC • Bloco 734)", x: 39.2, z: 18.6, tipo: "pc", desc: "Workstation principal onde o Trabalho Final de Ray Casting é compilado." }
  ];
  mesasConfig.forEach((m) => {
    addBox(1.45, 0.08, 0.85, matMadeira, m.x, 0.96, m.z);
    [[-0.64, -0.34], [0.64, -0.34], [-0.64, 0.34], [0.64, 0.34]].forEach(([dx, dz]) => {
      const pe = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 0.74, 8), matEsquadria);
      pe.position.set(m.x + dx, 0.57, m.z + dz);
      scene.add(pe);
    });
    if (m.tipo === "pc") {
      addBox(0.62, 0.40, 0.06, matTelaPC, m.x, 1.25, m.z - 0.12);
      addBox(0.20, 0.36, 0.38, matEsquadria, m.x + 0.46, 1.16, m.z - 0.08);
    } else {
      addBox(0.34, 0.06, 0.44, matPapel, m.x - 0.12, 1.03, m.z + 0.05, 0.24);
    }
    registrarAR(m.nome, "BANCADA ACADÊMICA • COLETA DE QUESTS", m.desc, "Cubos + Cilindros · Pressione [E] para coletar itens", [m.x, 0.8, m.z], [1.6, 1.4, 1.0]);
  });

  // 8 Itens Coletáveis do Inventário
  const collectibles = [
    { id: "atv1", nome: "Atividade 1", x: 45.15, y: 1.42, z: 18.2, cor: 0x38bdf8, coletado: false },
    { id: "atv2", nome: "Atividade 2", x: 45.60, y: 1.42, z: 18.2, cor: 0x38bdf8, coletado: false },
    { id: "atv3", nome: "Atividade 3", x: 46.05, y: 1.42, z: 18.2, cor: 0x38bdf8, coletado: false },
    { id: "atv4", nome: "Atividade 4", x: 47.55, y: 1.42, z: 18.2, cor: 0x4ade80, coletado: false },
    { id: "atv5", nome: "Atividade 5", x: 48.05, y: 1.42, z: 18.2, cor: 0x4ade80, coletado: false },
    { id: "ap1",  nome: "AP1",         x: 45.35, y: 1.42, z: 19.6, cor: 0xfacc15, coletado: false },
    { id: "ap2",  nome: "AP2",         x: 45.85, y: 1.42, z: 19.6, cor: 0xfacc15, coletado: false },
    { id: "tf",   nome: "Trabalho Final", x: 39.20, y: 1.50, z: 18.6, cor: 0xf472b6, coletado: false }
  ];
  collectibles.forEach((item) => {
    const geom = item.id === "tf" ? new THREE.OctahedronGeometry(0.22, 0) : new THREE.BoxGeometry(0.22, 0.22, 0.22);
    const matI = new THREE.MeshStandardMaterial({ color: item.cor, emissive: item.cor, emissiveIntensity: 0.45 });
    const mesh = new THREE.Mesh(geom, matI);
    mesh.position.set(item.x, item.y, item.z);
    scene.add(mesh);
    item.mesh = mesh;
  });

  function atualizarUIInventario() {
    const count = collectibles.filter(c => c.coletado).length;
    document.getElementById('inv-count').textContent = `${count}/${collectibles.length}`;
    const grid = document.getElementById('inv-grid-container');
    grid.innerHTML = collectibles.map(c => `
      <div class="inv-slot ${c.coletado ? 'collected' : ''}">
        <div style="font-size:20px; margin-bottom:4px;">${c.coletado ? '✅' : '📦'}</div>
        <div style="font-weight:700; color:${c.coletado ? '#4ade80' : '#f8fafc'};">${c.nome}</div>
        <div style="font-size:11px; color:#94a3b8; margin-top:2px;">
          ${c.coletado ? 'Coletado no Inventário!' : `Mesa em X=${c.x.toFixed(1)}, Z=${c.z.toFixed(1)}`}
        </div>
      </div>
    `).join('');
  }
  atualizarUIInventario();

  // 12. MÁQUINA DE CAFÉ INTERATIVA (X=48.3, Z=19.6)
  const cafePos = new THREE.Vector3(48.3, 0.2, 19.6);
  addBox(0.85, 0.90, 0.75, matMadeira, cafePos.x, 0.65, cafePos.z);
  addBox(0.46, 0.56, 0.44, new THREE.MeshStandardMaterial({ color: 0x334155, metalness: 0.75, roughness: 0.25 }), cafePos.x, 1.38, cafePos.z);
  const xicaraMesa = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.05, 0.11, 12), matPapel);
  xicaraMesa.position.set(cafePos.x, 1.15, cafePos.z - 0.14);
  scene.add(xicaraMesa);
  const ledCafe = new THREE.Mesh(new THREE.SphereGeometry(0.04, 8, 8), new THREE.MeshBasicMaterial({ color: 0x22c55e }));
  ledCafe.position.set(cafePos.x + 0.14, 1.54, cafePos.z - 0.23);
  scene.add(ledCafe);

  registrarAR(
    "Máquina de Café Expresso CT",
    "INTERAÇÃO ESPECIAL • CUTSCENE DO CAFEZINHO",
    "Cafeteira automática do Centro de Tecnologia. Pressione [E] para assistir à cutscene do cafezinho e ganhar Energia + Velocidade!",
    "Cubos + Cilindro + Esfera LED · [E] Tomar Café",
    [cafePos.x, 1.0, cafePos.z], [1.1, 1.7, 1.0],
    { isCoffeeMachine: true }
  );

  // 13. JOGADOR CONTROLÁVEL (ESTILO ROBLOX) + XÍCARA NA MÃO + ARCO DE ATAQUE
  const player = criarAvatarRoblox("Jogador", 24.0, 0.2, 13.5, 0, 0x0284c7, 0xfacc15, 0x0f172a, 1.0, "");
  const xicaraMao = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.055, 0.13, 12), matPapel);
  xicaraMao.position.set(0, -0.38, 0.14);
  xicaraMao.visible = false;
  player.bD.add(xicaraMao);

  // Partículas de vapor do café
  const steamParticles = [];
  for (let i = 0; i < 12; i++) {
    const sp = new THREE.Mesh(
      new THREE.SphereGeometry(0.05, 6, 6),
      new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.65 })
    );
    sp.visible = false;
    scene.add(sp);
    steamParticles.push({ mesh: sp, t: Math.random() });
  }

  // Wireframe Bounding Box Temporária de Realidade Aumentada
  const arBoxHelper = new THREE.Box3Helper(new THREE.Box3(), 0x22d3ee);
  arBoxHelper.visible = false;
  scene.add(arBoxHelper);
  let arTimer = 0;

  // Estado do Jogo
  let isPaused = false;
  let inCoffeeCutscene = false;
  let coffeeTimer = 0;
  let coffeeBuffTimer = 0;
  let energy = 100;
  let windowsBrokenCount = 0;
  let attackTimer = 0;
  let shakeTimer = 0;

  const pState = {
    pos: new THREE.Vector3(24.0, 0.2, 13.5),
    velY: 0,
    yaw: Math.PI,
    camYaw: Math.PI,
    camPitch: 0.34,
    camDist: 7.2,
    onGround: true,
    wasOnRoof: false
  };

  // Toast helper
  let toastTimeout = null;
  function showToast(msg, duration = 2600) {
    const el = document.getElementById('toast-msg');
    el.textContent = msg;
    el.style.display = 'block';
    clearTimeout(toastTimeout);
    toastTimeout = setTimeout(() => { el.style.display = 'none'; }, duration);
  }

  // Fullscreen toggle (com fallback para iframes restritos)
  document.getElementById('btn-fullscreen').addEventListener('click', () => {
    if (!document.fullscreenElement && !wrapper.classList.contains('pseudo-fullscreen')) {
      if (wrapper.requestFullscreen) {
        wrapper.requestFullscreen().catch(() => wrapper.classList.add('pseudo-fullscreen'));
      } else {
        wrapper.classList.add('pseudo-fullscreen');
      }
    } else {
      if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
      wrapper.classList.remove('pseudo-fullscreen');
    }
  });

  // Painel Retrátil de Quests & Info (Pausa o Jogo)
  const pauseModal = document.getElementById('pause-modal');
  function togglePauseModal(forceState = null) {
    if (inCoffeeCutscene) return;
    isPaused = forceState !== null ? forceState : !isPaused;
    pauseModal.style.display = isPaused ? 'flex' : 'none';
    if (isPaused) atualizarUIInventario();
  }
  document.getElementById('btn-quests').addEventListener('click', () => togglePauseModal(true));
  document.getElementById('btn-resume').addEventListener('click', () => togglePauseModal(false));

  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const target = btn.getAttribute('data-tab');
      ['tab-quests', 'tab-inv', 'tab-help'].forEach(id => {
        document.getElementById(id).style.display = (id === target) ? 'block' : 'none';
      });
    });
  });

  // Controles de Teclado e Mouse
  const keys = {};
  window.addEventListener('keydown', (e) => {
    const k = e.key.toLowerCase();
    if (k === 'q' || k === 'tab') {
      e.preventDefault();
      togglePauseModal();
      return;
    }
    if (k === 'escape' && isPaused) {
      togglePauseModal(false);
      return;
    }
    keys[k] = true;
    if (isPaused || inCoffeeCutscene) return;

    if (e.code === 'Space') {
      e.preventDefault();
      tentarPular();
    } else if (k === 'e') {
      executarInteracaoAR();
    } else if (k === 'f') {
      executarAtaque();
    }
  });
  window.addEventListener('keyup', (e) => { keys[e.key.toLowerCase()] = false; });

  let isDragging = false, lastMX = 0, lastMY = 0;
  canvas.addEventListener('mousedown', (e) => { isDragging = true; lastMX = e.clientX; lastMY = e.clientY; });
  window.addEventListener('mouseup', () => { isDragging = false; });
  window.addEventListener('mousemove', (e) => {
    if (!isDragging || isPaused || inCoffeeCutscene) return;
    const dx = e.clientX - lastMX;
    const dy = e.clientY - lastMY;
    lastMX = e.clientX; lastMY = e.clientY;
    pState.camYaw -= dx * 0.0065;
    pState.camPitch = Math.max(0.08, Math.min(1.25, pState.camPitch + dy * 0.0055));
  });
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    pState.camDist = Math.max(3.2, Math.min(16.0, pState.camDist + e.deltaY * 0.005));
  }, { passive: false });

  function tentarPular() {
    if (pState.onGround) {
      pState.velY = coffeeBuffTimer > 0 ? 8.6 : 7.4;
      pState.onGround = false;
    }
  }

  // Ação de Ataque [F]: Quebrar Janelas + Derramar Asfalto
  function executarAtaque() {
    if (attackTimer > 0 || isPaused || inCoffeeCutscene) return;
    attackTimer = 0.36;

    const forward = new THREE.Vector3(Math.sin(pState.yaw), 0, Math.cos(pState.yaw));
    const hitPoint = pState.pos.clone().add(new THREE.Vector3(0, 1.3, 0)).addScaledVector(forward, 1.5);

    let acertouJanela = false;
    breakableWindows.forEach((w) => {
      if (w.broken) return;
      const distXZ = Math.hypot(hitPoint.x - w.pos.x, hitPoint.z - w.pos.z);
      if (distXZ < 2.35 && Math.abs(hitPoint.y - w.pos.y) < 2.2) {
        acertouJanela = true;
        w.hp -= 1;
        w.hideBarTimer = 3.0;
        w.hpBarBg.visible = true;
        w.hpBarFill.visible = true;
        const ratio = Math.max(0, w.hp / w.maxHp);
        w.hpBarFill.scale.x = Math.max(0.01, ratio);
        w.hpBarFill.material.color.setHex(ratio > 0.5 ? 0x22c55e : (ratio > 0.2 ? 0xfacc15 : 0xef4444));

        spawnEstilhacos(w.pos, 0x93c5fd, 8);
        shakeTimer = 0.14;

        if (w.hp <= 0) {
          w.broken = true;
          w.glass.visible = false;
          w.hpBarBg.visible = false;
          w.hpBarFill.visible = false;
          windowsBrokenCount++;
          document.getElementById('win-broken').textContent = `${windowsBrokenCount}/${breakableWindows.length}`;
          spawnEstilhacos(w.pos, 0xe0f2fe, 24);
          dispararDerramamentoAsfalto(w);
          showToast(`💥 ${w.nome} quebrada! Derramando asfalto CBUQ na fachada...`, 3200);
        } else {
          w.glassMat.color.setHex(w.hp === 2 ? 0x1e3a5f : 0x3b82f6);
          showToast(`⚔️ Impacto em ${w.nome}! Durabilidade: ${w.hp}/${w.maxHp}`, 1800);
        }
      }
    });

    if (!acertouJanela) {
      showToast("⚔️ Golpe no ar! Aproxime-se das janelas de vidro do Bloco 734 para quebrá-las.", 1600);
    }
  }

  // Botão de Interação AR ("O que é isso?") + Coleta de Itens + Cutscene Café
  function encontrarObjetoProximo() {
    // 1. Prioridade: Itens coletáveis próximos
    for (const item of collectibles) {
      if (!item.coletado && pState.pos.distanceTo(new THREE.Vector3(item.x, pState.pos.y, item.z)) < 2.2) {
        return { tipo: 'collectible', item };
      }
    }
    // 2. Objetos AR (incluindo Máquina de Café)
    let melhor = null;
    let menorDist = 5.2;
    for (const obj of arObjects) {
      const d = Math.hypot(pState.pos.x - obj.center.x, pState.pos.z - obj.center.z);
      const raioEfetivo = Math.max(obj.size.x, obj.size.z) * 0.5 + 2.6;
      if (d < raioEfetivo && d < menorDist) {
        menorDist = d;
        melhor = { tipo: 'ar', obj };
      }
    }
    return melhor;
  }

  function iniciarCutsceneCafe() {
    inCoffeeCutscene = true;
    coffeeTimer = 4.6;
    xicaraMao.visible = true;
    document.getElementById('lb-top').style.height = '54px';
    document.getElementById('lb-bottom').style.height = '54px';
    document.getElementById('cutscene-caption').style.display = 'block';
    document.getElementById('ar-card').style.display = 'none';
  }

  function encerrarCutsceneCafe() {
    inCoffeeCutscene = false;
    xicaraMao.visible = false;
    steamParticles.forEach(s => s.mesh.visible = false);
    document.getElementById('lb-top').style.height = '0';
    document.getElementById('lb-bottom').style.height = '0';
    document.getElementById('cutscene-caption').style.display = 'none';
    energy = 100;
    coffeeBuffTimer = 30.0;
    document.getElementById('energy-fill').style.width = '100%';
    document.getElementById('buff-tag').style.display = 'inline';
    showToast("☕ Cafezinho concluído! Energia 100% + Buff de Velocidade por 30s!", 3200);
  }

  function executarInteracaoAR() {
    if (isPaused || inCoffeeCutscene) return;
    const alvo = encontrarObjetoProximo();
    if (!alvo) {
      showToast("📡 Nenhum objeto interativo por perto. Aproxime-se de uma estrutura, mesa ou cafeteira!", 2200);
      return;
    }

    if (alvo.tipo === 'collectible') {
      alvo.item.coletado = true;
      alvo.item.mesh.visible = false;
      atualizarUIInventario();
      showToast(`🎒 Coletado: ${alvo.item.nome}! (${collectibles.filter(c => c.coletado).length}/8 no inventário)`, 2800);
      return;
    }

    const obj = alvo.obj;
    if (obj.isCoffeeMachine) {
      iniciarCutsceneCafe();
      return;
    }

    // Ativa Bounding Box Temporária de Realidade Aumentada
    const minV = obj.center.clone().sub(obj.size.clone().multiplyScalar(0.5));
    const maxV = obj.center.clone().add(obj.size.clone().multiplyScalar(0.5));
    arBoxHelper.box.set(minV, maxV);
    arBoxHelper.visible = true;
    arTimer = 6.0;

    document.getElementById('ar-title').textContent = obj.nome;
    document.getElementById('ar-desc').textContent = obj.desc;
    document.getElementById('ar-meta').textContent = obj.meta;
    document.getElementById('ar-card').style.display = 'block';
  }

  document.getElementById('btn-interact').addEventListener('click', executarInteracaoAR);
  document.getElementById('btn-attack').addEventListener('click', executarAtaque);
  document.getElementById('btn-jump').addEventListener('click', tentarPular);

  // Auxiliar de Colisão & Altura do Solo (Incluindo Telhado, Escada, Árvores e Fonte)
  function dentroBloco734(x, z) {
    if (x < 15.0 || x > 59.0 || z < 20.0 || z > 53.0) return false;
    if (x < 37.0 && z < 42.0) {
      return ((x - 37.0) ** 2 + (z - 42.0) ** 2) <= (22.0 ** 2);
    }
    return true;
  }

  function getGroundY(x, z, curY) {
    if (dentroBloco734(x, z) && curY >= 6.1) return 6.60;
    if (x >= 19.5 && x <= 34.5 && z >= 16.8 && z <= 22.8) return 0.42; // Dentro da fonte
    return z >= 16.0 ? 0.20 : 0.10;
  }

  // Loop Principal de Simulação e Renderização
  const clock = new THREE.Clock();

  function animate() {
    requestAnimationFrame(animate);
    const dt = Math.min(clock.getDelta(), 0.05);
    const elapsed = clock.getElapsedTime();

    // Se pausado, apenas renderiza o frame congelado
    if (isPaused) {
      renderer.render(scene, camera);
      return;
    }

    // Animação da água da fonte e itens coletáveis
    aguaMesh.position.y = 0.38 + Math.sin(elapsed * 2.8) * 0.015;
    collectibles.forEach((c, idx) => {
      if (!c.coletado) {
        c.mesh.rotation.y += dt * 2.2;
        c.mesh.position.y = c.y + Math.sin(elapsed * 3.0 + idx) * 0.07;
      }
    });

    // Animação de derramamento de asfalto das janelas quebradas
    asphaltSpills.forEach((sp) => {
      if (sp.progress < 1.0) {
        sp.progress = Math.min(1.0, sp.progress + dt * 0.45);
        const h = (sp.topY - 0.20) * sp.progress;
        sp.streamMesh.scale.y = Math.max(0.1, h / 0.1);
        sp.streamMesh.position.y = sp.topY - h * 0.5;
        const r = 0.1 + 1.35 * sp.progress;
        sp.puddleMesh.scale.set(r / 0.1, 1, r / 0.1);
      }
    });

    // Estilhaços de vidro / partículas
    for (let i = glassShards.length - 1; i >= 0; i--) {
      const s = glassShards[i];
      s.life -= dt;
      s.vel.y -= 14.0 * dt;
      s.mesh.position.addScaledVector(s.vel, dt);
      s.mesh.rotation.x += dt * 6;
      s.mesh.rotation.y += dt * 5;
      if (s.life <= 0 || s.mesh.position.y < 0.15) {
        scene.remove(s.mesh);
        glassShards.splice(i, 1);
      }
    }

    // Barras de HP das janelas
    breakableWindows.forEach((w) => {
      if (w.hideBarTimer > 0) {
        w.hideBarTimer -= dt;
        if (w.hideBarTimer <= 0) {
          w.hpBarBg.visible = false;
          w.hpBarFill.visible = false;
        }
      }
    });

    // Bounding Box AR Timer
    if (arTimer > 0) {
      arTimer -= dt;
      arBoxHelper.color.setHex(Math.sin(elapsed * 10) > 0 ? 0x22d3ee : 0x38bdf8);
      if (arTimer <= 0) {
        arBoxHelper.visible = false;
        document.getElementById('ar-card').style.display = 'none';
      }
    }

    // Cutscene do Cafezinho
    if (inCoffeeCutscene) {
      coffeeTimer -= dt;
      player.bD.rotation.x = -1.35 + Math.sin(elapsed * 6.0) * 0.22;
      steamParticles.forEach((sp, i) => {
        sp.mesh.visible = true;
        sp.t = (sp.t + dt * 0.7) % 1.0;
        sp.mesh.position.set(
          pState.pos.x + Math.sin(elapsed * 4 + i) * 0.12,
          pState.pos.y + 1.55 + sp.t * 0.75,
          pState.pos.z + Math.cos(elapsed * 4 + i) * 0.12
        );
        sp.mesh.material.opacity = (1.0 - sp.t) * 0.7;
      });
      const angCam = elapsed * 0.9;
      camera.position.set(
        pState.pos.x + Math.cos(angCam) * 2.4,
        pState.pos.y + 1.85,
        pState.pos.z + Math.sin(angCam) * 2.4
      );
      camera.lookAt(pState.pos.x, pState.pos.y + 1.35, pState.pos.z);
      if (coffeeTimer <= 0) {
        player.bD.rotation.x = 0;
        encerrarCutsceneCafe();
      }
      renderer.render(scene, camera);
      return;
    }

    if (coffeeBuffTimer > 0) {
      coffeeBuffTimer -= dt;
      if (coffeeBuffTimer <= 0) {
        document.getElementById('buff-tag').style.display = 'none';
      }
    }

    // Ave Predadora (Ataca se invadir a zona da Árvore do Fundo X=11.5, Z=38.0)
    const distAve = Math.hypot(pState.pos.x - aveHome.x, pState.pos.z - aveHome.z);
    if (distAve < 6.2 && pState.pos.y < 6.0) {
      const targetAve = pState.pos.clone().add(new THREE.Vector3(0, 1.4, 0));
      aveGrp.position.lerp(targetAve, dt * 2.4);
      aveGrp.lookAt(targetAve);
      if (distAve < 1.8) {
        energy = Math.max(10, energy - 18 * dt);
        document.getElementById('energy-fill').style.width = `${energy}%`;
      }
    } else {
      aveGrp.position.lerp(aveHome.clone().add(new THREE.Vector3(0, Math.sin(elapsed * 3) * 0.18, 0)), dt * 2.0);
    }
    asaE.rotation.z = Math.sin(elapsed * 12) * 0.35;
    asaD.rotation.z = -Math.sin(elapsed * 12) * 0.35;

    // Movimento do Jogador
    const speed = coffeeBuffTimer > 0 ? 9.5 : 6.8;
    let moveF = 0, moveR = 0;
    if (keys['w'] || keys['arrowup']) moveF += 1;
    if (keys['s'] || keys['arrowdown']) moveF -= 1;
    if (keys['d'] || keys['arrowright']) moveR += 1;
    if (keys['a'] || keys['arrowleft']) moveR -= 1;

    const fwd = new THREE.Vector3(-Math.sin(pState.camYaw), 0, -Math.cos(pState.camYaw));
    const right = new THREE.Vector3(Math.cos(pState.camYaw), 0, -Math.sin(pState.camYaw));
    const moveVec = new THREE.Vector3().addScaledVector(fwd, moveF).addScaledVector(right, moveR);

    const isMoving = moveVec.lengthSq() > 0.001;
    if (isMoving) {
      moveVec.normalize();
      pState.yaw = Math.atan2(moveVec.x, moveVec.z);
      const nextX = Math.max(3.0, Math.min(66.0, pState.pos.x + moveVec.x * speed * dt));
      const nextZ = Math.max(3.0, Math.min(58.0, pState.pos.z + moveVec.z * speed * dt));

      // Bloqueia atravessar paredes do Bloco 734 se estiver abaixo do telhado
      if (!(dentroBloco734(nextX, nextZ) && pState.pos.y < 6.1)) {
        pState.pos.x = nextX;
        pState.pos.z = nextZ;
      }
    }

    // Escada Lateral (X=59.35, Z=24.0) — Subir e Descer do Telhado
    const distEscada = Math.hypot(pState.pos.x - 59.35, pState.pos.z - 24.0);
    if (distEscada < 1.35) {
      if (keys['w'] || keys['arrowup'] || keys[' ']) {
        pState.pos.y = Math.min(6.75, pState.pos.y + 4.8 * dt);
        pState.velY = 0;
        pState.onGround = true;
      } else if (keys['s'] || keys['arrowdown']) {
        pState.pos.y = Math.max(0.20, pState.pos.y - 4.8 * dt);
        pState.velY = 0;
        pState.onGround = true;
      }
    }

    // Gravidade e Pouso (Verificando Descida do Telhado em Árvore ou Fonte!)
    const groundY = getGroundY(pState.pos.x, pState.pos.z, pState.pos.y);
    if (pState.pos.y > groundY || pState.velY > 0) {
      pState.velY -= 19.5 * dt;
      pState.pos.y += pState.velY * dt;
      pState.onGround = false;
    }

    if (pState.pos.y <= groundY) {
      // Detecta aterrissagem vindo do telhado
      if (pState.wasOnRoof && groundY < 1.5) {
        const naFonte = (pState.pos.x >= 19.5 && pState.pos.x <= 34.5 && pState.pos.z >= 16.8 && pState.pos.z <= 22.8);
        const naArvore = arvoresCoords.some(a => Math.hypot(pState.pos.x - a.x, pState.pos.z - a.z) < 3.4) ||
                         (Math.hypot(pState.pos.x - 42.5, pState.pos.z - 17.5) < 3.0);
        const naEscada = distEscada < 2.2;

        if (naFonte) {
          spawnEstilhacos(pState.pos.clone().add(new THREE.Vector3(0, 0.4, 0)), 0x38bdf8, 28);
          showToast("💦 Descida perfeita na Fonte do CT! Queda 100% amortecida na água.", 3200);
        } else if (naArvore) {
          spawnEstilhacos(pState.pos.clone().add(new THREE.Vector3(0, 1.2, 0)), 0x22c55e, 24);
          showToast("🌿 Pouso amortecido pela copa da Árvore! Descida segura do telhado.", 3200);
        } else if (naEscada) {
          showToast("🪜 Descida segura pela Escada Lateral!", 2200);
        } else {
          shakeTimer = 0.25;
          energy = Math.max(15, energy - 20);
          document.getElementById('energy-fill').style.width = `${energy}%`;
          showToast("⚠️ Impacto no concreto! Dica: desça pela Escada, pela Árvore ou dentro da Fonte.", 3400);
        }
      }

      pState.pos.y = groundY;
      pState.velY = 0;
      pState.onGround = true;
      pState.wasOnRoof = (groundY >= 6.4);
    }

    // Atualiza Transformação & Animação de Membros do Jogador
    player.grp.position.copy(pState.pos);
    player.grp.rotation.y = pState.yaw;

    if (isMoving && pState.onGround) {
      const swing = Math.sin(elapsed * 11.0) * 0.55;
      player.pE.rotation.x = swing;
      player.pD.rotation.x = -swing;
      player.bE.rotation.x = -swing * 0.7;
      if (attackTimer <= 0) player.bD.rotation.x = swing * 0.7;
    } else {
      player.pE.rotation.x *= 0.8;
      player.pD.rotation.x *= 0.8;
      player.bE.rotation.x *= 0.8;
      if (attackTimer <= 0) player.bD.rotation.x *= 0.8;
    }

    if (attackTimer > 0) {
      attackTimer -= dt;
      player.bD.rotation.x = -Math.sin((1.0 - attackTimer / 0.36) * Math.PI) * 2.1;
    }

    // Prompt Contextual de Proximidade
    const prox = encontrarObjetoProximo();
    const promptEl = document.getElementById('interact-prompt');
    if (prox) {
      promptEl.style.display = 'block';
      if (prox.tipo === 'collectible') {
        promptEl.textContent = `🎒 [E] Coletar "${prox.item.nome}" para o Inventário`;
      } else if (prox.obj.isCoffeeMachine) {
        promptEl.textContent = `☕ [E] Interagir com Máquina de Café (Assistir Cutscene)`;
      } else {
        promptEl.textContent = `🔍 [E] O que é isso? — ${prox.obj.nome}`;
      }
    } else {
      promptEl.style.display = 'none';
    }

    // Câmera Orbital Suave
    const targetCam = new THREE.Vector3(
      pState.pos.x + Math.sin(pState.camYaw) * Math.cos(pState.camPitch) * pState.camDist,
      pState.pos.y + 1.5 + Math.sin(pState.camPitch) * pState.camDist,
      pState.pos.z + Math.cos(pState.camYaw) * Math.cos(pState.camPitch) * pState.camDist
    );
    if (shakeTimer > 0) {
      shakeTimer -= dt;
      targetCam.x += (Math.random() - 0.5) * 0.25;
      targetCam.y += (Math.random() - 0.5) * 0.25;
    }
    camera.position.lerp(targetCam, 0.22);
    camera.lookAt(pState.pos.x, pState.pos.y + 1.35, pState.pos.z);

    renderer.render(scene, camera);
  }

  window.addEventListener('resize', () => {
    camera.aspect = wrapper.clientWidth / wrapper.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(wrapper.clientWidth, wrapper.clientHeight);
  });

  animate();
})();
</script>
</body>
</html>
"""
  return template.replace("__MODO_NOITE__", "true" if modo_noite else "false")


# ==============================================================================
# BARRA LATERAL — PARÂMETROS DO MOTOR RAY CASTING & CÂMERA (ATOS I–VIII)
# ==============================================================================
renderer = st.session_state.renderer

with st.sidebar:
  st.header("🎛️ Painel de Controle — Bloco 734")
  preset_key = st.selectbox(
      "Preset de Câmera / Projeção",
      options=list(renderer.presets.keys()),
      format_func=lambda k: renderer.presets[k]["titulo"],
      index=0,
  )
  preset_base = renderer.presets[preset_key]

  col_res, col_turno = st.columns(2)
  with col_res:
    nova_res = st.selectbox(
        "Resolução (px)",
        options=[250, 360, 500, 600],
        index=[250, 360, 500, 600].index(st.session_state.resolucao),
    )
    if nova_res != st.session_state.resolucao:
      st.session_state.resolucao = nova_res
      st.session_state.renderer = RenderizadorCT(
          largura=nova_res, altura=nova_res
      )
      st.session_state.ultima_img_arr = None
      st.session_state.ultimo_pick_info = None
      st.rerun()

  with col_turno:
    turno = st.radio("Iluminação", ["Dia", "Noite"], index=0)
    modo_noite = turno == "Noite"

  st.subheader("✨ Efeitos Ópticos")
  c1, c2 = st.columns(2)
  with c1:
    ativar_sombras = st.checkbox("Sombras", value=True)
    ativar_caustics = st.checkbox("Cáusticas", value=True)
  with c2:
    ativar_reflexos = st.checkbox("Reflexos", value=True)
    ssaa_4x = st.checkbox("SSAA 4x", value=False)

  with st.expander("📷 Ajuste Fino da Câmera (Eye, At, d, Janela)", expanded=False):
    ex = st.number_input("Eye X", value=float(preset_base["eye"][0]), step=0.5)
    ey = st.number_input("Eye Y", value=float(preset_base["eye"][1]), step=0.5)
    ez = st.number_input("Eye Z", value=float(preset_base["eye"][2]), step=0.5)

    ax = st.number_input("At X", value=float(preset_base["at"][0]), step=0.5)
    ay = st.number_input("At Y", value=float(preset_base["at"][1]), step=0.5)
    az = st.number_input("At Z", value=float(preset_base["at"][2]), step=0.5)

    d_focal = st.slider(
        "Distância Focal (d)", 0.5, 4.0, float(preset_base["d"]), 0.05
    )
    w_escala = st.slider("Escala da Janela (FOV)", 0.5, 2.0, 1.0, 0.05)

  w0, w1, w2, w3 = preset_base["win"]
  cfg_atual = {
      "titulo": preset_base["titulo"],
      "eye": [ex, ey, ez],
      "at": [ax, ay, az],
      "up": preset_base["up"],
      "d": d_focal,
      "win": (w0 * w_escala, w1 * w_escala, w2 * w_escala, w3 * w_escala),
      "proj": preset_base["proj"],
  }

  btn_renderizar = st.button(
      "🚀 Renderizar Cena (Ray Casting)", type="primary", use_container_width=True
  )


# ==============================================================================
# ABAS PRINCIPAIS DA APLICAÇÃO
# ==============================================================================
st.title("🏛️ Bloco 734 (CT/UFC) — Motor Ray Casting & Walkthrough 3D")

aba_rc, aba_webgl, aba_quests = st.tabs([
    "🖼️ Renderizador Ray Casting (Picking & Alta Qualidade)",
    "🎮 Modo WebGL / Estilo Roblox (Walkthrough, Quests & Café)",
    "📜 Painel de Quests (Atos I–VIII) & Matrizes",
])


# ------------------------------------------------------------------------------
# ABA 1: RENDERIZADOR RAY CASTING + PICKING COM BANNER NO TOPO/BASE
# ------------------------------------------------------------------------------
with aba_rc:
  if btn_renderizar or st.session_state.ultima_img_arr is None:
    with st.spinner(
        f"Traçando raios analíticos em {renderer.W}×{renderer.H}px..."
    ):
      img_arr, legenda = renderer.renderizar_config(
          cfg_atual,
          modo_noite=modo_noite,
          ativar_caustics=ativar_caustics,
          ativar_reflexos=ativar_reflexos,
          ativar_sombras=ativar_sombras,
          ssaa=ssaa_4x,
      )
      st.session_state.ultima_img_arr = img_arr
      st.session_state.ultima_legenda = legenda
      st.session_state.cfg_renderizada = cfg_atual
      st.session_state.ultimo_pick_info = None

  img_uint8 = (np.clip(st.session_state.ultima_img_arr, 0.0, 1.0) * 255).astype(
      np.uint8
  )
  pil_img = Image.fromarray(img_uint8)

  col_img, col_ctrl = st.columns([2.2, 1.0])

  with col_img:
    st.caption(
        f"📌 **{st.session_state.ultima_legenda}** — Clique sobre qualquer ponto da imagem para inspecionar o objeto atingido."
    )

    # Banner superior caso já exista um pixel inspecionado
    if st.session_state.ultimo_pick_info is not None:
      px_c, py_c = st.session_state.ultima_coord_clicada or (None, None)
      renderizar_banner_pick(st.session_state.ultimo_pick_info, px_c, py_c)

    if TEM_IMG_COORDS:
      coords = streamlit_image_coordinates(pil_img, key="rc_pick_canvas")
      if coords is not None:
        px, py = int(coords["x"]), int(coords["y"])
        if st.session_state.ultima_coord_clicada != (px, py):
          st.session_state.ultima_coord_clicada = (px, py)
          st.session_state.ultimo_pick_info = renderer.consultar_pick_buffer(
              px, py
          )
          st.rerun()
    else:
      st.image(pil_img, use_container_width=True)

  with col_ctrl:
    st.subheader("🎯 Inspetor de Pixel (Picking)")
    st.write(
        "Clique diretamente na imagem ao lado ou informe coordenadas `(X, Y)` para consultar o buffer geométrico em tempo real:"
    )
    p_x = st.number_input("Pixel X", 0, renderer.W - 1, renderer.W // 2)
    p_y = st.number_input("Pixel Y", 0, renderer.H - 1, renderer.H // 2)
    if st.button("🔍 Inspecionar Coordenada (X, Y)", use_container_width=True):
      st.session_state.ultima_coord_clicada = (int(p_x), int(p_y))
      st.session_state.ultimo_pick_info = renderer.consultar_pick_buffer(
          int(p_x), int(p_y)
      )
      st.rerun()

    st.divider()
    buf_png = io.BytesIO()
    pil_img.save(buf_png, format="PNG")
    st.download_button(
        label=f"⬇️ Baixar Imagem PNG ({renderer.W}×{renderer.H})",
        data=buf_png.getvalue(),
        file_name=f"bloco734_raycasting_{renderer.W}x{renderer.H}.png",
        mime="image/png",
        use_container_width=True,
    )


# ------------------------------------------------------------------------------
# ABA 2: MODO WEBGL / ESTILO ROBLOX COMPLETO & IMERSIVO
# ------------------------------------------------------------------------------
with aba_webgl:
  components.html(
      gerar_html_jogo_webgl(modo_noite=modo_noite), height=695, scrolling=False
  )


# ------------------------------------------------------------------------------
# ABA 3: PAINEL DE QUESTS (ATOS I–VIII) & AUDITORIA MATEMÁTICA
# ------------------------------------------------------------------------------
with aba_quests:
  st.subheader("📊 Auditoria Automática — Quests da Apresentação Final (CG I)")
  status_quests = renderer.obter_status_quests(
      cfg_atual=cfg_atual,
      resolucao=renderer.W,
      ativar_sombras=ativar_sombras,
  )

  for ato_key, ato_data in status_quests.items():
    with st.expander(ato_data["titulo"], expanded=True):
      for desc_q, ok_q, tag_q in ato_data["items"]:
        c_q1, c_q2 = st.columns([4.5, 1.0])
        with c_q1:
          st.markdown(f"{'✅' if ok_q else '⚠️'} {desc_q}")
        with c_q2:
          st.caption(f"**{tag_q}**")

  st.subheader("🧮 Matriz de Transformação Mundo → Câmera ($M_{wc}$)")
  M_wc_atual = construir_Mwc(
      cfg_atual["eye"], cfg_atual["at"], cfg_atual["up"]
  )
  st.code(np.array2string(M_wc_atual, precision=4, suppress_small=True))
