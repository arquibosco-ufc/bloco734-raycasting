import io
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
        padding-top: 1.0rem;
        padding-bottom: 1.4rem;
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
        grid-template-columns: repeat(auto-fit, minmax(195px, 1fr));
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

  coord_txt = (
      f"Pixel ({px}, {py}) • " if (px is not None and py is not None) else ""
  )
  st.markdown(
      f"""
      <div class="pick-banner">
          <div class="pick-banner-title">
              🎯 {coord_txt}{info['nome']} <span style="font-size:0.8rem; color:#94a3b8;">[{info['tipo_primitiva']}]</span>
          </div>
          <div class="pick-banner-grid">
              <div><b>Grupo:</b> {info['grupo']}</div>
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
# HTML / THREE.JS / WEB AUDIO — MODO WEBGL COMPLETO & INTERATIVO
# ==============================================================================
def gerar_html_jogo_webgl(
    modo_noite=False, ativar_sol=True, ativar_spots=True
):
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
    position: relative; width: 100%; height: 700px;
    background: #0b1120; border-radius: 12px; overflow: hidden;
    border: 1px solid #1e293b;
  }
  #game-wrapper.pseudo-fullscreen {
    position: fixed !important; inset: 0 !important;
    width: 100vw !important; height: 100vh !important;
    z-index: 999999 !important; border-radius: 0 !important;
  }
  canvas { display: block; width: 100%; height: 100%; cursor: grab; }
  canvas:active { cursor: grabbing; }

  /* HUD MINIMALISTA */
  #hud-top-left {
    position: absolute; top: 12px; left: 12px;
    display: flex; flex-direction: column; gap: 6px;
    pointer-events: none; z-index: 10;
  }
  .hud-pill {
    background: rgba(15, 23, 42, 0.78);
    backdrop-filter: blur(6px);
    border: 1px solid rgba(148, 163, 184, 0.30);
    color: #f8fafc; padding: 6px 12px; border-radius: 999px;
    font-size: 12.5px; font-weight: 600;
    display: flex; align-items: center; gap: 8px;
    width: fit-content;
  }
  .bar-bg {
    width: 104px; height: 9px; background: rgba(30, 41, 59, 0.95);
    border-radius: 999px; overflow: hidden; border: 1px solid rgba(255,255,255,0.12);
  }
  #energy-fill {
    width: 100%; height: 100%;
    background: linear-gradient(90deg, #22c55e, #38bdf8);
    transition: width 0.18s, background 0.2s;
  }

  #hud-top-right {
    position: absolute; top: 12px; right: 12px;
    display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 7px; z-index: 15;
  }
  .hud-btn {
    background: rgba(15, 23, 42, 0.84);
    backdrop-filter: blur(6px);
    border: 1px solid rgba(56, 189, 248, 0.45);
    color: #f8fafc; padding: 7px 11px; border-radius: 8px;
    font-size: 12px; font-weight: 600; cursor: pointer;
    transition: all 0.15s ease;
  }
  .hud-btn:hover { background: rgba(30, 41, 59, 0.96); border-color: #38bdf8; transform: translateY(-1px); }
  .hud-btn.surv-on {
    background: rgba(127, 29, 29, 0.88);
    border-color: #ef4444; color: #fecaca;
    box-shadow: 0 0 14px rgba(239, 68, 68, 0.35);
  }

  /* PROMPT DE PROXIMIDADE */
  #interact-prompt {
    position: absolute; bottom: 84px; left: 50%; transform: translateX(-50%);
    background: rgba(15, 23, 42, 0.88);
    border: 1px solid #38bdf8; color: #f8fafc;
    padding: 7px 16px; border-radius: 999px;
    font-size: 13px; font-weight: 600;
    display: none; z-index: 12; pointer-events: none;
    box-shadow: 0 4px 16px rgba(0,0,0,0.45);
  }

  /* BOTÕES DE AÇÃO INFERIORES */
  #hud-bottom-left {
    position: absolute; bottom: 14px; left: 14px;
    display: flex; gap: 6px; z-index: 14;
  }
  #hud-bottom-right {
    position: absolute; bottom: 14px; right: 14px;
    display: flex; gap: 7px; z-index: 14;
  }
  .action-btn {
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid rgba(148, 163, 184, 0.42);
    color: #f8fafc; padding: 9px 12px; border-radius: 10px;
    font-size: 12px; font-weight: 700; cursor: pointer;
    transition: 0.15s;
  }
  .action-btn:hover { background: #1e293b; border-color: #38bdf8; }
  .action-btn.ar { border-color: #38bdf8; color: #7dd3fc; }
  .action-btn.atk { border-color: #f97316; color: #fdba74; }
  .action-btn.dance { border-color: #ec4899; color: #f9a8d4; }

  /* CARTÃO DE REALIDADE AUMENTADA */
  #ar-card {
    position: absolute; top: 64px; right: 14px; width: 315px;
    background: rgba(9, 14, 28, 0.90);
    backdrop-filter: blur(10px);
    border: 1px solid #22d3ee; border-left: 4px solid #22d3ee;
    border-radius: 10px; padding: 12px 14px; color: #e2e8f0;
    display: none; z-index: 18;
    box-shadow: 0 10px 30px rgba(34, 211, 238, 0.25);
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

  /* FLASH DE DANO / SUSTO */
  #damage-flash {
    position: absolute; inset: 0; pointer-events: none; z-index: 22;
    background: radial-gradient(circle, transparent 45%, rgba(220, 38, 38, 0.65) 100%);
    opacity: 0; transition: opacity 0.22s ease;
  }

  /* TOAST DE FEEDBACK */
  #toast-msg {
    position: absolute; top: 58px; left: 50%; transform: translateX(-50%);
    background: rgba(15, 23, 42, 0.92); border: 1px solid #38bdf8;
    color: #f8fafc; padding: 7px 16px; border-radius: 8px;
    font-size: 12.5px; font-weight: 600; display: none; z-index: 25;
    text-align: center; max-width: 82%;
  }

  /* LETTERBOX CUTSCENE (CAFÉ & DANÇA) */
  .letterbox {
    position: absolute; left: 0; width: 100%; height: 0;
    background: #020617; z-index: 30; transition: height 0.4s ease;
  }
  #lb-top { top: 0; }
  #lb-bottom { bottom: 0; }
  #cutscene-caption {
    position: absolute; bottom: 18px; left: 50%; transform: translateX(-50%);
    color: #fef3c7; font-size: 15px; font-weight: 700;
    text-shadow: 0 2px 10px #000; z-index: 31; display: none;
    text-align: center; width: 88%;
  }

  /* TELA GAME OVER (MODO SOBREVIVÊNCIA — DUMB WAYS TO DIE) */
  #gameover-modal {
    position: absolute; inset: 0;
    background: rgba(69, 10, 10, 0.88);
    backdrop-filter: blur(8px);
    display: none; flex-direction: column; align-items: center; justify-content: center;
    z-index: 45; color: #fef2f2; text-align: center; padding: 24px;
  }
  #gameover-modal h2 { font-size: 32px; color: #fca5a5; margin-bottom: 8px; letter-spacing: 1px; }
  #gameover-reason { font-size: 16px; color: #fecaca; margin-bottom: 20px; max-width: 520px; }

  /* PAINEL RETRÁTIL DE QUESTS & INFORMAÇÕES (PAUSA O JOGO) */
  #pause-modal {
    position: absolute; inset: 0;
    background: rgba(2, 6, 23, 0.85);
    backdrop-filter: blur(8px);
    display: none; align-items: center; justify-content: center;
    z-index: 40; padding: 18px;
  }
  .modal-box {
    width: 100%; max-width: 880px; max-height: 92%;
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
  <div id="damage-flash"></div>

  <!-- HUD MINIMALISTA -->
  <div id="hud-top-left">
    <div class="hud-pill">
      <span>❤️ Life</span>
      <div class="bar-bg"><div id="energy-fill"></div></div>
      <span id="hp-text" style="font-size:11.5px; color:#cbd5e1;">100%</span>
      <span id="buff-tag" style="color:#facc15; display:none;">☕ Turbo!</span>
    </div>
    <div class="hud-pill">
      <span>🎒 Inventário: <b id="inv-count" style="color:#38bdf8;">0/8</b></span>
      <span>•</span>
      <span>🪟 Vidros: <b id="win-broken" style="color:#fb923c;">0/10</b></span>
      <span>•</span>
      <span id="proj-badge" style="color:#a5f3fc;">📷 Perspectiva (2 PF)</span>
    </div>
  </div>

  <div id="hud-top-right">
    <button class="hud-btn" id="btn-survival">💀 Sobrevivência: OFF</button>
    <button class="hud-btn" id="btn-audio">🔊 Som: ON</button>
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

  <!-- ATALHOS RÁPIDOS DE PROJEÇÃO / CÂMERA -->
  <div id="hud-bottom-left">
    <button class="action-btn" id="btn-vp" title="Alternar 1, 2 e 3 Pontos de Fuga">📐 Pontos Fuga</button>
    <button class="action-btn" id="btn-iso" title="Alternar Projeção Isométrica / Ortográfica">🧊 Isométrica</button>
    <button class="action-btn" id="btn-obl" title="Alternar Projeção Oblíqua Gabinete">📏 Oblíqua</button>
  </div>

  <!-- BOTÕES DE AÇÃO DO PERSONAGEM -->
  <div id="hud-bottom-right">
    <button class="action-btn dance" id="btn-dance">💃 Comemorar [C]</button>
    <button class="action-btn ar" id="btn-interact">🔍 Interagir / AR [E]</button>
    <button class="action-btn atk" id="btn-attack">⚔️ Atacar [F]</button>
    <button class="action-btn" id="btn-jump">⬆️ Pular [Espaço]</button>
  </div>

  <!-- LETTERBOX CUTSCENE -->
  <div class="letterbox" id="lb-top"></div>
  <div class="letterbox" id="lb-bottom"></div>
  <div id="cutscene-caption">☕ Preparando um cafezinho expresso no Centro de Tecnologia...</div>

  <!-- MODAL GAME OVER (MODO SOBREVIVÊNCIA) -->
  <div id="gameover-modal">
    <h2>💀 DUMB WAYS TO DIE — CT EDITION</h2>
    <p id="gameover-reason">Você sofreu um acidente no campus do Bloco 734!</p>
    <button class="hud-btn" id="btn-respawn" style="font-size:15px; padding:11px 24px; background:#dc2626; border-color:#fca5a5;">
      🔄 Renascer na Praça do Bloco 734
    </button>
  </div>

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
        <button class="tab-btn" data-tab="tab-help">🧭 Controles, Totens & Áudio</button>
      </div>
      <div class="modal-body" id="tab-quests">
        <div class="act-block">
          <h4>🏰 ATO I — A Construção do Mundo</h4>
          <div class="q-row"><span>🎯 Sub-quest 1.1 — "Tudo Tem Um Propósito" (Bloco 734 CT Coerente)</span><span class="q-badge q-done">✔ Concluída</span></div>
          <div class="q-row"><span>🎯 Sub-quest 1.2 — "O Primeiro Octante" (Todos vértices X, Y, Z &ge; 0)</span><span class="q-badge q-done">✔ Concluída</span></div>
          <div class="q-row"><span>🎯 Quest 2 — "O Zoo dos Objetos" (Esfera, Cilindro, Cone, Caixa, Arco e Malha)</span><span class="q-badge q-done">✔ Concluída</span></div>
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
          <h4>🔦 ATO III — A Luz e a Escuridão (Totens Interativos na Praça)</h4>
          <div class="q-row"><span>💡 Sub-quest 7.1 — "A Chama Pontual" (4 Postes com atenuação quadrática)</span><span class="q-badge q-done">✔ Concluída</span></div>
          <div class="q-row"><span>🌫️ Sub-quest 7.2 — "O Brilho Ambiente" (Componente Ia)</span><span class="q-badge q-done">✔ Concluída</span></div>
          <div class="q-row"><span>🔦 Sub-quest Bônus 7.3 — "O Holofote" (Totem Liga/Desliga Spots)</span><span class="q-badge q-bonus" id="q-spot-badge">+1.0 pts</span></div>
          <div class="q-row"><span>☀️ Sub-quest Bônus 7.4 — "O Sol Direcional" (Totem Liga/Desliga Sol)</span><span class="q-badge q-bonus" id="q-sun-badge">+0.5 pts</span></div>
        </div>
        <div class="act-block">
          <h4>📷 ATOS IV & V — Câmera, Pontos de Fuga & Projeções Alternativas</h4>
          <div class="q-row"><span>👁️ Quests 8 & 9 — Eye, At, Up, Distância Focal (d) e Janela (xmin..ymax)</span><span class="q-badge q-done">✔ Concluída</span></div>
          <div class="q-row"><span>🔭 Quest 10 — "A Perspectiva" (Zoom In / Zoom Out)</span><span class="q-badge q-done">✔ Concluída</span></div>
          <div class="q-row"><span>1️⃣2️⃣3️⃣ Quests Bônus — "Os Pontos de Fuga" (Totem/Botão 1, 2 e 3 PF)</span><span class="q-badge q-bonus" id="q-vp-badge">+1.5 pts</span></div>
          <div class="q-row"><span>🧊📏 Quests Bônus — "Projeções Alternativas" (Totem/Botão Isométrica & Oblíqua)</span><span class="q-badge q-bonus" id="q-proj-badge">+1.0 pts</span></div>
        </div>
        <div class="act-block">
          <h4>🌑🖱️🖼️ ATOS VI, VII & VIII — Sombras, Interatividade e Mosaico 500×500</h4>
          <div class="q-row"><span>🌑 Quest 11 — "A Sombra" (Sombras projetadas)</span><span class="q-badge q-done">✔ Concluída</span></div>
          <div class="q-row"><span>🎯 Quest 12 — "O Toque" (Pick por clique no Render + Totens + AR 3D)</span><span class="q-badge q-done">✔ Concluída</span></div>
          <div class="q-row"><span>🖥️ Quest Bônus — "A Interface Gráfica" (Streamlit + Motor WebGL Interativo)</span><span class="q-badge q-bonus">+1.0 pts</span></div>
          <div class="q-row"><span>🖼️ Quest 13 — "O Mosaico de 500×500" (Ray Casting &ge; 500×500 px)</span><span class="q-badge q-done">✔ Concluída</span></div>
          <div class="q-row"><span>🌟 Quest Lendária — "Criatividade e Beleza" (Cáusticas + Sobrevivência + Dança)</span><span class="q-badge q-bonus">+1.0 pts</span></div>
        </div>
      </div>

      <div class="modal-body" id="tab-inv" style="display:none;">
        <p>Aproxime-se das <b>Mesas de Trabalho</b> (sob o Abrigo Metálico e na Praça) e pressione <b>[E]</b> para coletar suas entregas. Coletar tudo ativa a <b>Cutscene de Comemoração</b>!</p>
        <div class="inv-grid" id="inv-grid-container"></div>
      </div>

      <div class="modal-body" id="tab-help" style="display:none;">
        <div class="act-block">
          <h4>🎮 Controles, Totens Interativos & Mecânicas</h4>
          <p>• <b>W A S D / Setas:</b> Caminhar (com passos estilo Roblox).<br/>
             • <b>Arrastar Mouse / Roda:</b> Girar câmera orbital e Zoom.<br/>
             • <b>Clique nos Totens 3D (ou [E] perto deles):</b> Liga/Desliga <b>Spots</b>, Liga/Desliga <b>Sol</b>, alterna <b>1/2/3 Pontos de Fuga</b>, <b>Projeção Isométrica</b> e <b>Projeção Oblíqua</b>.<br/>
             • <b>[E] na Cafeteira (Ilha da Praça X=35.8, Z=18.8):</b> Cutscene do cafezinho em 360° livre, recupera 100% de Life e dá Turbo.<br/>
             • <b>[F] Atacar:</b> Quebra as janelas curvas do Bloco 734 e derrama asfalto animado.<br/>
             • <b>[C] Comemorar:</b> Executa dança comemorativa estilo Fortnite com música.<br/>
             • <b>💀 Sobrevivência (Dumb Ways to Die):</b> Quando ativo, carros na via, ave territorial, NPC sombrio e quedas do telhado podem eliminar o personagem!</p>
        </div>
      </div>
    </div>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
(function() {
  const MODO_NOITE = __MODO_NOITE__;
  let sunEnabled = __ATIVAR_SOL__;
  let spotsEnabled = __ATIVAR_SPOTS__;

  const wrapper = document.getElementById('game-wrapper');
  const canvas = document.getElementById('c');

  // ============================================================================
  // 1. MOTOR DE ÁUDIO PROCEDURAL (WEB AUDIO API — SEM ARQUIVOS EXTERNOS)
  // ============================================================================
  const SoundEngine = {
    ctx: null,
    enabled: true,
    ambientTimer: null,
    init() {
      if (!this.ctx) {
        const AC = window.AudioContext || window.webkitAudioContext;
        if (AC) this.ctx = new AC();
      }
      if (this.ctx && this.ctx.state === 'suspended') {
        this.ctx.resume().catch(() => {});
      }
      if (!this.ambientTimer) {
        this.ambientTimer = setInterval(() => this.playAmbientTick(), 3400);
      }
    },
    tone(freq, dur, type = 'sine', vol = 0.08, delay = 0, slideTo = null) {
      if (!this.enabled) return;
      this.init();
      if (!this.ctx) return;
      const t0 = this.ctx.currentTime + delay;
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();
      osc.type = type;
      osc.frequency.setValueAtTime(freq, t0);
      if (slideTo) osc.frequency.exponentialRampToValueAtTime(Math.max(20, slideTo), t0 + dur);
      gain.gain.setValueAtTime(0.0001, t0);
      gain.gain.linearRampToValueAtTime(vol, t0 + 0.012);
      gain.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
      osc.connect(gain);
      gain.connect(this.ctx.destination);
      osc.start(t0);
      osc.stop(t0 + dur + 0.02);
    },
    playStep() {
      // Passo estilo plástico/bloco Roblox
      const pitch = 155 + Math.random() * 35;
      this.tone(pitch, 0.055, 'triangle', 0.075, 0, 75);
    },
    playJump() {
      this.tone(220, 0.18, 'sine', 0.09, 0, 520);
    },
    playSwitch() {
      this.tone(580, 0.07, 'square', 0.06, 0);
      this.tone(860, 0.09, 'square', 0.06, 0.07);
    },
    playCollect() {
      [523.25, 659.25, 783.99, 1046.5].forEach((f, i) => {
        this.tone(f, 0.14, 'triangle', 0.09, i * 0.065);
      });
    },
    playGlass() {
      [1400, 1950, 2400, 1100, 1750].forEach((f, i) => {
        this.tone(f + Math.random()*300, 0.12, 'sawtooth', 0.06, i * 0.025, 320);
      });
      // Som grave de asfalto escorrendo
      this.tone(110, 0.65, 'sine', 0.08, 0.1, 55);
    },
    playHonk() {
      this.tone(415, 0.22, 'sawtooth', 0.07, 0);
      this.tone(523, 0.22, 'sawtooth', 0.06, 0);
    },
    playScare() {
      // Susto agudo + impacto grave
      this.tone(920, 0.35, 'sawtooth', 0.12, 0, 240);
      this.tone(95, 0.45, 'square', 0.12, 0, 38);
    },
    playCoffee() {
      [330, 392, 494, 587, 659].forEach((f, i) => {
        this.tone(f, 0.25, 'sine', 0.07, i * 0.22);
      });
    },
    playDanceBeat() {
      // Melodia rítmica inspirada em emote comemorativo
      const notes = [
        [329.6, 0.0], [329.6, 0.22], [392.0, 0.44], [440.0, 0.66],
        [392.0, 0.96], [329.6, 1.22], [293.7, 1.48], [329.6, 1.75],
        [440.0, 2.15], [392.0, 2.40], [329.6, 2.65], [392.0, 2.95], [523.3, 3.25]
      ];
      notes.forEach(([f, d]) => this.tone(f, 0.18, 'square', 0.065, d));
    },
    playAmbientTick() {
      if (!this.enabled || !this.ctx || this.ctx.state !== 'running') return;
      if (MODO_NOITE || !sunEnabled) {
        // Grilo noturno suave
        this.tone(3600, 0.04, 'sine', 0.012, 0);
        this.tone(3650, 0.04, 'sine', 0.012, 0.08);
      } else {
        // Pássaro distante de dia
        const base = 1800 + Math.random() * 500;
        this.tone(base, 0.08, 'sine', 0.015, 0, base * 1.25);
        this.tone(base * 1.15, 0.09, 'sine', 0.015, 0.14, base * 0.9);
      }
    }
  };

  document.getElementById('btn-audio').addEventListener('click', (e) => {
    SoundEngine.enabled = !SoundEngine.enabled;
    e.target.textContent = SoundEngine.enabled ? '🔊 Som: ON' : '🔇 Som: OFF';
    if (SoundEngine.enabled) SoundEngine.playSwitch();
  });

  // ============================================================================
  // 2. RENDERER WEBGL & CÂMERAS (PERSPECTIVA, ISOMÉTRICA E OBLÍQUA)
  // ============================================================================
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(wrapper.clientWidth, wrapper.clientHeight);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  const scene = new THREE.Scene();
  const perspCamera = new THREE.PerspectiveCamera(58, wrapper.clientWidth / wrapper.clientHeight, 0.1, 260);
  const frustumSize = 26;
  const aspect0 = wrapper.clientWidth / wrapper.clientHeight;
  const orthoCamera = new THREE.OrthographicCamera(
    -frustumSize * aspect0 * 0.5, frustumSize * aspect0 * 0.5,
    frustumSize * 0.5, -frustumSize * 0.5, -60, 220
  );

  let projMode = 'perspectiva'; // 'perspectiva' | 'isometrica' | 'obliqua'
  let vanishingMode = 2; // 1, 2, 3

  // ============================================================================
  // 3. ILUMINAÇÃO DINÂMICA (SOL DIRECIONAL & HOLOFOTES / SPOTS)
  // ============================================================================
  const hemiLight = new THREE.HemisphereLight(0xe0f2fe, 0x475569, 0.85);
  scene.add(hemiLight);

  const dirLight = new THREE.DirectionalLight(0xfffbeb, 1.15);
  dirLight.position.set(12, 34, 8);
  dirLight.castShadow = true;
  dirLight.shadow.mapSize.set(2048, 2048);
  dirLight.shadow.camera.near = 2;
  dirLight.shadow.camera.far = 110;
  const dCam = 44;
  dirLight.shadow.camera.left = -dCam;
  dirLight.shadow.camera.right = dCam;
  dirLight.shadow.camera.top = dCam;
  dirLight.shadow.camera.bottom = -dCam;
  scene.add(dirLight);

  // Spotlights (Escultura Cinética + 3 Spots Submersos da Fonte)
  const spotLightsList = [];
  const spotEsc = new THREE.SpotLight(0xbae6fd, 2.6, 32, Math.PI / 5, 0.35, 1);
  spotEsc.position.set(12.5, 5.2, 18.0);
  spotEsc.target.position.set(15.0, 1.8, 22.0);
  scene.add(spotEsc, spotEsc.target);
  spotLightsList.push(spotEsc);

  const spotBulbs = [];
  [22.0, 27.0, 32.0].forEach((sx) => {
    const sp = new THREE.Mesh(
      new THREE.SphereGeometry(0.17, 12, 12),
      new THREE.MeshBasicMaterial({ color: 0x67e8f9 })
    );
    sp.position.set(sx, 0.34, 17.12);
    scene.add(sp);
    spotBulbs.push(sp);

    const pSub = new THREE.PointLight(0x38bdf8, 1.3, 12);
    pSub.position.set(sx, 0.85, 17.4);
    scene.add(pSub);
    spotLightsList.push(pSub);
  });

  const posteLights = [];
  [[9.5, 21.0, 0.6], [18.5, 16.2, 0.4], [41.0, 16.5, 0.0], [54.5, 17.5, -0.5]].forEach(([px, pz, ox]) => {
    const pt = new THREE.PointLight(0xfde68a, 1.3, 24);
    pt.position.set(px + ox, 6.0, pz + 0.4);
    scene.add(pt);
    posteLights.push(pt);
  });

  function atualizarIluminacaoCena() {
    const isDark = MODO_NOITE || !sunEnabled;
    scene.background = new THREE.Color(isDark ? 0x060b19 : 0x8ec5fc);
    scene.fog = new THREE.FogExp2(isDark ? 0x060b19 : 0x93c5fd, isDark ? 0.013 : 0.006);
    hemiLight.color.setHex(isDark ? 0x1e293b : 0xe0f2fe);
    hemiLight.intensity = isDark ? 0.48 : 0.88;
    dirLight.visible = sunEnabled;
    dirLight.intensity = MODO_NOITE ? 0.25 : 1.15;

    spotLightsList.forEach(l => { l.visible = spotsEnabled; });
    spotBulbs.forEach(b => { b.material.color.setHex(spotsEnabled ? 0x67e8f9 : 0x334155); });
    posteLights.forEach(p => { p.intensity = isDark ? 1.45 : 0.25; });

    document.getElementById('q-sun-badge').textContent = sunEnabled ? '✔ Sol ON (+0.5 pts)' : '⚠️ Sol OFF';
    document.getElementById('q-spot-badge').textContent = spotsEnabled ? '✔ Spots ON (+1.0 pts)' : '⚠️ Spots OFF';
  }

  // ============================================================================
  // 4. MATERIAIS & CONSTRUÇÃO DO CENÁRIO BLOCO 734
  // ============================================================================
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

  const arObjects = [];
  const clickableTotems = [];
  function registrarAR(nome, tag, desc, meta, center, size, extra = {}) {
    arObjects.push({
      nome, tag, desc, meta,
      center: new THREE.Vector3(...center),
      size: new THREE.Vector3(...size),
      ...extra
    });
  }

  // Piso Urbano
  addBox(66, 0.1, 14, matAsfalto, 35, 0.05, 9, 0, false);
  addBox(64, 0.02, 0.14, matFaixa, 35, 0.11, 11.5, 0, false);
  addBox(64, 0.02, 0.14, matFaixa, 35, 0.11, 11.9, 0, false);
  addBox(66, 0.2, 44, matCalcada, 35, 0.10, 38, 0, false);

  // Bloco 734
  addBox(22, 6.6, 33, matAlvenaria, 48, 3.3, 36.5);
  addBox(22, 6.6, 11, matAlvenaria, 26, 3.3, 47.5);
  const arcMesh = new THREE.Mesh(
    new THREE.CylinderGeometry(22, 22, 6.6, 48, 1, false, Math.PI, Math.PI * 0.5),
    matAlvenaria
  );
  arcMesh.position.set(37, 3.3, 42);
  arcMesh.castShadow = true;
  arcMesh.receiveShadow = true;
  scene.add(arcMesh);

  // Placa 734
  const pCanvas = document.createElement('canvas');
  pCanvas.width = 256; pCanvas.height = 96;
  const pCtx = pCanvas.getContext('2d');
  pCtx.fillStyle = '#f8fafc'; pCtx.fillRect(0, 0, 256, 96);
  pCtx.strokeStyle = '#1e293b'; pCtx.lineWidth = 6; pCtx.strokeRect(3, 3, 250, 90);
  pCtx.fillStyle = '#0f172a'; pCtx.font = 'bold 62px sans-serif';
  pCtx.textAlign = 'center'; pCtx.textBaseline = 'middle';
  pCtx.fillText('734', 128, 52);
  const angPlaca = 0.14 * (Math.PI / 2);
  const placaMesh = new THREE.Mesh(
    new THREE.BoxGeometry(3.4, 1.15, 0.18),
    new THREE.MeshStandardMaterial({ map: new THREE.CanvasTexture(pCanvas) })
  );
  placaMesh.position.set(37 - 22.08 * Math.cos(angPlaca), 5.72, 42 - 22.08 * Math.sin(angPlaca));
  placaMesh.rotation.y = Math.PI / 2 - angPlaca;
  scene.add(placaMesh);

  registrarAR(
    'Bloco 734 (CT / UFC)',
    'EDIFICAÇÃO PRINCIPAL • CSG ANALÍTICO',
    'Edifício sede com fachada arqueada tangencial (R=22m, H=6.6m), 9 janelas quebráveis e platibanda nº 734.',
    'Primitivas: Prisma + Cilindro R=22m • Primeiro Octante',
    [37, 3.3, 36.5], [44, 6.8, 33]
  );

  // Janelas Quebráveis + Derramamento de Asfalto
  const breakableWindows = [];
  const asphaltSpills = [];
  const particles = [];
  const matAsfaltoLiquido = new THREE.MeshStandardMaterial({ color: 0x090b10, roughness: 0.18, metalness: 0.35 });

  function criarJanelaCurva(nome, sArco, yCentro, larg, alt) {
    const ang = sArco * (Math.PI / 2);
    const r = 22.06;
    const wx = 37 - r * Math.cos(ang);
    const wz = 42 - r * Math.sin(ang);
    const rotY = Math.PI / 2 - ang;

    const grp = new THREE.Group();
    grp.position.set(wx, yCentro, wz);
    grp.rotation.y = rotY;

    const frame = new THREE.Mesh(new THREE.BoxGeometry(larg + 0.18, alt + 0.18, 0.12), matEsquadria);
    const glassMat = matVidro.clone();
    const glass = new THREE.Mesh(new THREE.BoxGeometry(larg, alt, 0.16), glassMat);
    const hpBarBg = new THREE.Mesh(new THREE.BoxGeometry(1.4, 0.14, 0.05), new THREE.MeshBasicMaterial({ color: 0x0f172a }));
    hpBarBg.position.set(0, alt * 0.5 + 0.32, 0.12);
    hpBarBg.visible = false;
    const hpBarFill = new THREE.Mesh(new THREE.BoxGeometry(1.34, 0.10, 0.06), new THREE.MeshBasicMaterial({ color: 0x22c55e }));
    hpBarFill.position.set(0, alt * 0.5 + 0.32, 0.13);
    hpBarFill.visible = false;

    grp.add(frame, glass, hpBarBg, hpBarFill);
    scene.add(grp);

    breakableWindows.push({
      nome, grp, glass, glassMat, hpBarBg, hpBarFill,
      pos: new THREE.Vector3(wx, yCentro, wz),
      normal: new THREE.Vector3(-Math.cos(ang), 0, -Math.sin(ang)),
      rotY, larg, alt, hp: 3, maxHp: 3, broken: false, hideBarTimer: 0
    });
  }

  criarJanelaCurva('Porta Principal de Vidro (Bloco 734)', 0.13, 1.48, 2.5, 2.55);
  for (let i = 0; i < 9; i++) {
    const sMid = 0.24 + ((i + 0.5) / 9.0) * 0.72;
    criarJanelaCurva(`Janela Fachada Curva #${i + 1}`, sMid, 2.30, 1.75, 1.88);
  }

  function dispararDerramamentoAsfalto(winObj) {
    const basePos = winObj.pos.clone().addScaledVector(winObj.normal, 0.16);
    const streamMesh = new THREE.Mesh(new THREE.BoxGeometry(winObj.larg * 0.82, 0.1, 0.18), matAsfaltoLiquido);
    streamMesh.position.set(basePos.x, winObj.pos.y - winObj.alt * 0.35, basePos.z);
    streamMesh.rotation.y = winObj.rotY;
    scene.add(streamMesh);

    const puddlePos = winObj.pos.clone().addScaledVector(winObj.normal, 1.15);
    const puddleMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.1, 0.04, 24), matAsfaltoLiquido);
    puddleMesh.position.set(puddlePos.x, 0.22, puddlePos.z);
    scene.add(puddleMesh);

    asphaltSpills.push({ streamMesh, puddleMesh, topY: winObj.pos.y - winObj.alt * 0.35, progress: 0.0 });
  }

  function spawnParticulas(pos, coresHex, qtd = 20, spread = 5.0) {
    const geo = new THREE.BoxGeometry(0.14, 0.14, 0.06);
    for (let i = 0; i < qtd; i++) {
      const cor = Array.isArray(coresHex) ? coresHex[i % coresHex.length] : coresHex;
      const m = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: cor }));
      m.position.copy(pos).add(new THREE.Vector3((Math.random()-0.5)*1.2, (Math.random()-0.5)*1.0, (Math.random()-0.5)*0.8));
      scene.add(m);
      particles.push({
        mesh: m,
        vel: new THREE.Vector3((Math.random()-0.5)*spread, Math.random()*4.5 + 1.8, (Math.random()-0.5)*spread),
        life: 1.6
      });
    }
  }

  // Fonte Frontal
  addBox(15.3, 0.34, 0.28, matGranito, 27.0, 0.37, 16.8);
  addBox(15.3, 0.34, 0.28, matGranito, 27.0, 0.37, 22.8);
  addBox(0.28, 0.34, 6.0, matGranito, 19.5, 0.37, 19.8);
  addBox(0.28, 0.34, 6.0, matGranito, 34.5, 0.37, 19.8);
  const aguaMesh = addBox(14.85, 0.08, 5.85, matAgua, 27.0, 0.38, 19.8, 0, false);

  registrarAR(
    "Espelho d'Água (Fonte Cáustica)",
    "REFRAÇÃO n=1.333 • ZONA DE POUSO SEGURO",
    "Lâmina d'água com 3 spots submersos. Saltar do telhado dentro da fonte amortece 100% da queda!",
    "Cubos + Esferas • Textura Ondulatória",
    [27.0, 0.42, 19.8], [15.4, 0.6, 6.2]
  );

  // Escultura Cinética (5 Losangos)
  const escGrp = new THREE.Group();
  escGrp.position.set(15.0, 0.2, 22.0);
  escGrp.rotation.y = THREE.MathUtils.degToRad(-24);
  const baseEsc = new THREE.Mesh(new THREE.BoxGeometry(3.2, 0.28, 3.4), matConcreto);
  baseEsc.position.y = 0.14;
  escGrp.add(baseEsc);
  for (let k = 0; k < 5; k++) {
    const fc = k - 2;
    const Lk = 2.35 * (1.0 - 0.055 * fc);
    const rk = Lk / Math.sqrt(2);
    const qGrp = new THREE.Group();
    qGrp.position.set(fc * 0.22, 0.28 + rk + 0.05, -1.16 + k * 0.58);
    qGrp.rotation.z = Math.PI / 4;
    const esp = 0.14, prof = 0.16;
    [
      [Lk + esp, esp, 0, Lk/2], [Lk + esp, esp, 0, -Lk/2],
      [esp, Lk + esp, Lk/2, 0], [esp, Lk + esp, -Lk/2, 0]
    ].forEach(([w, h, bx, by]) => {
      const b = new THREE.Mesh(new THREE.BoxGeometry(w, h, prof), matAcoBranco);
      b.position.set(bx, by, 0);
      b.castShadow = true;
      qGrp.add(b);
    });
    escGrp.add(qGrp);
  }
  scene.add(escGrp);
  registrarAR(
    "Escultura Cinética em Aço",
    "COMPOSIÇÃO PARAMÉTRICA • 5 LOSANGOS",
    "Cinco quadros metálicos rotacionados a 45°, escalonados progressivamente e orientados a -24° em Y.",
    "Translação · Rotação Z(45°) & Y(-24°) · Escala",
    [15.0, 2.0, 22.0], [3.6, 3.8, 3.8]
  );

  // Escada Telhado
  addBox(0.55, 6.6, 1.1, matEsquadria, 59.35, 3.4, 24.0);
  for (let deg = 0.6; deg <= 6.4; deg += 0.45) {
    addBox(0.65, 0.06, 0.95, matMetalGalv, 59.35, deg, 24.0);
  }
  registrarAR(
    "Escada de Acesso ao Telhado",
    "TRANSPORTE VERTICAL • Y=0.2m → Y=6.8m",
    "Escada lateral (X=59.35, Z=24.0). Use W/Espaço para subir e S para descer.",
    "Cubos Escalonados",
    [59.35, 3.4, 24.0], [1.2, 6.8, 1.5]
  );

  // Árvores + Ave Predadora
  const arvoresCoords = [
    { nome: "Árvore Bosque Esq.", x: 10.5, z: 28.5, esc: 1.15 },
    { nome: "Árvore Fundo Esq.", x: 11.5, z: 38.0, esc: 1.25 },
    { nome: "Árvore Praça Dir.", x: 37.5, z: 17.5, esc: 0.90 }
  ];
  arvoresCoords.forEach((arv) => {
    const tMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.28 * arv.esc, 0.42 * arv.esc, 3.2 * arv.esc, 12), matTronco);
    tMesh.position.set(arv.x, 0.2 + 1.6 * arv.esc, arv.z);
    tMesh.castShadow = true;
    scene.add(tMesh);
    [[0, 3.6, 0, 1.65, matFolha1], [-1.1, 3.2, 0.6, 1.25, matFolha2], [1.1, 3.3, -0.5, 1.30, matFolha2], [0.3, 4.5, 0.4, 1.20, matFolha1]].forEach(([ox, oy, oz, r, m]) => {
      const cMesh = new THREE.Mesh(new THREE.SphereGeometry(r * arv.esc, 14, 12), m);
      cMesh.position.set(arv.x + ox * arv.esc, oy * arv.esc, arv.z + oz * arv.esc);
      cMesh.castShadow = true;
      scene.add(cMesh);
    });
    registrarAR(arv.nome, "VEGETAÇÃO • POUSO SEGURO DO TELHADO", "Saltar do telhado sobre a copa amortece a queda!", "Cilindro + Esferas (Quatérnios)", [arv.x, 2.8 * arv.esc, arv.z], [3.4 * arv.esc, 5.4 * arv.esc, 3.4 * arv.esc]);
  });

  // Ave Predadora (X=11.5, Z=38.0)
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
  let birdScareCooldown = 0;

  // Cones + Abrigo Cisalhado + Postes
  [[35.3, 16.2], [35.9, 15.6], [36.5, 16.9], [37.1, 16.3]].forEach(([cx, cz]) => {
    const c = new THREE.Mesh(new THREE.ConeGeometry(0.32, 0.95, 14), matCone);
    c.position.set(cx, 0.2 + 0.475, cz);
    c.castShadow = true;
    scene.add(c);
  });

  const abrigoMesh = new THREE.Mesh(new THREE.BoxGeometry(4.5, 0.16, 3.2), matMetalGalv);
  abrigoMesh.position.set(46.8, 2.6, 18.8);
  abrigoMesh.rotation.z = -0.09;
  abrigoMesh.castShadow = true;
  scene.add(abrigoMesh);
  [[44.7, 17.4], [48.8, 17.4], [44.7, 20.2], [48.8, 20.2]].forEach(([px, pz]) => {
    const pil = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.08, 2.4, 10), matEsquadria);
    pil.position.set(px, 1.4, pz);
    scene.add(pil);
  });

  [[9.5, 21.0, 0.6], [18.5, 16.2, 0.4], [41.0, 16.5, 0.0], [54.5, 17.5, -0.5]].forEach(([px, pz, ox]) => {
    const p = new THREE.Mesh(new THREE.CylinderGeometry(0.10, 0.14, 6.2, 10), matEsquadria);
    p.position.set(px, 3.3, pz);
    scene.add(p);
    const lum = new THREE.Mesh(new THREE.SphereGeometry(0.28, 12, 12), new THREE.MeshBasicMaterial({ color: 0xfef08a }));
    lum.position.set(px + ox, 6.3, pz + 0.4);
    scene.add(lum);
  });

  // ============================================================================
  // 5. CARROS ANIMADOS NA VIA CT (SUV PRATA & ESPORTIVO VERMELHO)
  // ============================================================================
  const animatedCars = [];
  function criarVeiculoAnimado(nome, xInit, zLane, dirX, speed, corHex) {
    const grp = new THREE.Group();
    grp.position.set(xInit, 0.1, zLane);
    grp.rotation.y = dirX > 0 ? Math.PI / 2 : -Math.PI / 2;

    const matCarro = new THREE.MeshStandardMaterial({ color: corHex, metalness: 0.4, roughness: 0.25 });
    const matPneu = new THREE.MeshStandardMaterial({ color: 0x111827, roughness: 0.9 });
    const chassi = new THREE.Mesh(new THREE.BoxGeometry(2.0, 0.65, 4.3), matCarro);
    chassi.position.y = 0.55; chassi.castShadow = true;
    const cabine = new THREE.Mesh(new THREE.BoxGeometry(1.75, 0.54, 2.2), matVidro);
    cabine.position.set(0, 1.12, -0.15);
    grp.add(chassi, cabine);

    const rodas = [];
    [[-1.02, 0.36, 1.35], [1.02, 0.36, 1.35], [-1.02, 0.36, -1.35], [1.02, 0.36, -1.35]].forEach(([rx, ry, rz]) => {
      const roda = new THREE.Mesh(new THREE.CylinderGeometry(0.36, 0.36, 0.22, 12), matPneu);
      roda.rotation.z = Math.PI / 2;
      roda.position.set(rx, ry, rz);
      grp.add(roda);
      rodas.push(roda);
    });
    [-0.65, 0.65].forEach((fx) => {
      const f = new THREE.Mesh(new THREE.SphereGeometry(0.14, 8, 8), new THREE.MeshBasicMaterial({ color: 0xfef9c3 }));
      f.position.set(fx, 0.62, 2.15);
      const l = new THREE.Mesh(new THREE.SphereGeometry(0.14, 8, 8), new THREE.MeshBasicMaterial({ color: 0xef4444 }));
      l.position.set(fx, 0.62, -2.15);
      grp.add(f, l);
    });
    scene.add(grp);
    animatedCars.push({ nome, grp, rodas, dirX, speed, honkTimer: 0 });
  }

  criarVeiculoAnimado("SUV Prata na Via CT", 12.0, 7.2, 1, 9.5, 0xcbd5e1);
  criarVeiculoAnimado("Carro Esportivo Vermelho", 58.0, 10.6, -1, 13.0, 0xdc2626);

  // Personagens & Palmeira
  function criarAvatarRoblox(x, y, z, rotDeg, corTorso, corCabeca, corCalca, esc = 1.0) {
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
    return { grp, bE, bD, pE, pD, torso, cab };
  }

  criarAvatarRoblox(18.6, 0.2, 18.8, -35, 0x2563eb, 0xfacc15, 0x1e293b, 0.95);
  criarAvatarRoblox(19.6, 0.2, 18.2, -55, 0xea580c, 0xfacc15, 0x1e293b, 0.92);
  const npcMisterioso = criarAvatarRoblox(43.0, 0.2, 19.5, 180, 0x0f172a, 0xe2e8f0, 0x020617, 1.02);
  let npcScareCooldown = 0;

  // Palmeira
  [[0, 0.2, 0, 0.24, 1.6, 0], [0.05, 1.75, 0, 0.20, 1.6, -0.07], [0.15, 3.3, 0, 0.16, 1.5, -0.12]].forEach(([ox, oy, oz, r, h, rz]) => {
    const seg = new THREE.Mesh(new THREE.CylinderGeometry(r * 0.85, r, h, 10), matTronco);
    seg.position.set(42.5 + ox, oy + h * 0.5, 17.5 + oz);
    seg.rotation.z = rz;
    scene.add(seg);
  });
  for (let k = 0; k < 6; k++) {
    const ang = k * (Math.PI * 2 / 6);
    const palma = new THREE.Mesh(new THREE.ConeGeometry(0.42, 2.2, 8), matFolha2);
    palma.position.set(42.74 + Math.cos(ang) * 0.9, 4.65, 17.5 + Math.sin(ang) * 0.9);
    palma.rotation.z = -Math.cos(ang) * 1.15;
    palma.rotation.x = Math.sin(ang) * 1.15;
    scene.add(palma);
  }

  // ============================================================================
  // 6. TOTENS INTERATIVOS 3D NA PRAÇA (CLIQUE DO MOUSE OU [E])
  // ============================================================================
  function criarTotemInterativo(id, titulo, x, z, corHex, acaoFn) {
    const grp = new THREE.Group();
    grp.position.set(x, 0.2, z);

    const pedestal = new THREE.Mesh(new THREE.CylinderGeometry(0.28, 0.34, 1.05, 14), matEsquadria);
    pedestal.position.y = 0.525;
    pedestal.castShadow = true;

    const btnMat = new THREE.MeshStandardMaterial({
      color: corHex, emissive: corHex, emissiveIntensity: 0.45, roughness: 0.3
    });
    const botaoTopo = new THREE.Mesh(new THREE.BoxGeometry(0.52, 0.26, 0.52), btnMat);
    botaoTopo.position.y = 1.18;
    botaoTopo.castShadow = true;

    grp.add(pedestal, botaoTopo);
    scene.add(grp);

    const totemData = { id, titulo, x, z, grp, botaoTopo, acaoFn };
    botaoTopo.userData.totem = totemData;
    pedestal.userData.totem = totemData;
    clickableTotems.push(botaoTopo, pedestal);

    registrarAR(
      titulo,
      "TOTEM INTERATIVO • CLIQUE OU [E]",
      "Painel de controle físico na praça do CT. Clique com o mouse sobre ele ou pressione [E] ao se aproximar!",
      "Cilindro + Cubo Emissivo",
      [x, 0.8, z], [1.0, 1.6, 1.0],
      { totem: totemData }
    );
    return totemData;
  }

  function alternarPontosDeFuga() {
    SoundEngine.playSwitch();
    vanishingMode = (vanishingMode % 3) + 1;
    projMode = 'perspectiva';
    if (vanishingMode === 1) {
      pState.pos.set(27.0, 0.2, 13.5);
      pState.yaw = 0;
      pState.camYaw = Math.PI;
      pState.camPitch = 0.08;
      pState.camDist = 7.5;
      showToast("1️⃣ 1 Ponto de Fuga ativado (Eixo Frontal Z alinhado com a Fonte e Bloco 734)", 2800);
    } else if (vanishingMode === 2) {
      pState.camYaw = Math.PI * 0.78;
      pState.camPitch = 0.14;
      pState.camDist = 7.8;
      showToast("2️⃣ 2 Pontos de Fuga ativados (Câmera nivelada em ângulo oblíquo)", 2800);
    } else {
      pState.camYaw = Math.PI * 0.76;
      pState.camPitch = 0.72;
      pState.camDist = 11.5;
      showToast("3️⃣ 3+ Pontos de Fuga ativados (Visada aérea inclinada)", 2800);
    }
    document.getElementById('proj-badge').textContent = `📷 Perspectiva (${vanishingMode} PF)`;
  }

  function alternarIsometrica() {
    SoundEngine.playSwitch();
    if (projMode === 'isometrica') {
      projMode = 'perspectiva';
      document.getElementById('proj-badge').textContent = `📷 Perspectiva (${vanishingMode} PF)`;
      showToast("📷 Retornou para Projeção Perspectiva", 2000);
    } else {
      projMode = 'isometrica';
      pState.camYaw = Math.PI * 0.75;
      pState.camPitch = 0.615; // arctan(1/sqrt(2)) ≈ 35.26°
      document.getElementById('proj-badge').textContent = "🧊 Projeção Isométrica (Ortográfica)";
      showToast("🧊 Projeção Isométrica / Ortográfica ativada (Raios paralelos sem fuga)", 2800);
    }
  }

  function alternarObliqua() {
    SoundEngine.playSwitch();
    if (projMode === 'obliqua') {
      projMode = 'perspectiva';
      document.getElementById('proj-badge').textContent = `📷 Perspectiva (${vanishingMode} PF)`;
      showToast("📷 Retornou para Projeção Perspectiva", 2000);
    } else {
      projMode = 'obliqua';
      pState.camYaw = Math.PI;
      pState.camPitch = 0.05;
      document.getElementById('proj-badge').textContent = "📏 Projeção Oblíqua (Gabinete 45°)";
      showToast("📏 Projeção Oblíqua (Gabinete 45°) ativada via cisalhamento na matriz de projeção!", 2800);
    }
  }

  criarTotemInterativo("totem_spots", "Totem: Liga/Desliga Holofotes (Spots)", 21.5, 15.2, 0x38bdf8, () => {
    spotsEnabled = !spotsEnabled;
    SoundEngine.playSwitch();
    atualizarIluminacaoCena();
    showToast(`🔦 Holofotes / Spots: ${spotsEnabled ? 'LIGADOS (Quest Bônus 7.3)' : 'DESLIGADOS'}`, 2400);
  });

  criarTotemInterativo("totem_sol", "Totem: Liga/Desliga Sol Direcional", 23.5, 15.2, 0xfacc15, () => {
    sunEnabled = !sunEnabled;
    SoundEngine.playSwitch();
    atualizarIluminacaoCena();
    showToast(`☀️ Sol Direcional: ${sunEnabled ? 'LIGADO (Quest Bônus 7.4)' : 'DESLIGADO'}`, 2400);
  });

  criarTotemInterativo("totem_vp", "Totem: Alternar Pontos de Fuga (1/2/3 PF)", 25.5, 15.2, 0xa855f7, alternarPontosDeFuga);
  criarTotemInterativo("totem_iso", "Totem: Projeção Isométrica (Ortográfica)", 27.5, 15.2, 0x22c55e, alternarIsometrica);
  criarTotemInterativo("totem_obl", "Totem: Projeção Oblíqua (Gabinete)", 29.5, 15.2, 0xf97316, alternarObliqua);

  document.getElementById('btn-vp').addEventListener('click', alternarPontosDeFuga);
  document.getElementById('btn-iso').addEventListener('click', alternarIsometrica);
  document.getElementById('btn-obl').addEventListener('click', alternarObliqua);

  // ============================================================================
  // 7. MESAS DE TRABALHO, INVENTÁRIO (8 ITENS) & CAFETEIRA EM ILHA ABERTA
  // ============================================================================
  const mesasConfig = [
    { nome: "Mesa Trabalho 1 (PC • Atividades 1–3)", x: 45.6, z: 18.2, tipo: "pc" },
    { nome: "Mesa Trabalho 2 (Papéis • Atividades 4–5)", x: 47.8, z: 18.2, tipo: "papel" },
    { nome: "Mesa Avaliações (Papéis • AP1 e AP2)", x: 45.6, z: 19.6, tipo: "papel" },
    { nome: "Estação Trabalho Final (PC • Bloco 734)", x: 39.2, z: 18.6, tipo: "pc" }
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
  });

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
          ${c.coletado ? 'Concluído!' : `Mesa X=${c.x.toFixed(1)}, Z=${c.z.toFixed(1)}`}
        </div>
      </div>
    `).join('');
  }
  atualizarUIInventario();

  // Cafeteira reposicionada em ILHA ABERTA na praça (X=35.8, Z=18.8 — zero parede próxima!)
  const cafePos = new THREE.Vector3(35.8, 0.2, 18.8);
  addBox(0.85, 0.90, 0.75, matMadeira, cafePos.x, 0.65, cafePos.z);
  addBox(0.46, 0.56, 0.44, new THREE.MeshStandardMaterial({ color: 0x334155, metalness: 0.75, roughness: 0.25 }), cafePos.x, 1.38, cafePos.z);
  const xicaraMesa = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.05, 0.11, 12), matPapel);
  xicaraMesa.position.set(cafePos.x, 1.15, cafePos.z - 0.14);
  scene.add(xicaraMesa);
  const ledCafe = new THREE.Mesh(new THREE.SphereGeometry(0.045, 8, 8), new THREE.MeshBasicMaterial({ color: 0x22c55e }));
  ledCafe.position.set(cafePos.x + 0.14, 1.54, cafePos.z - 0.23);
  scene.add(ledCafe);

  registrarAR(
    "Máquina de Café Expresso CT",
    "ILHA ABERTA • RECUPERA 100% LIFE",
    "Cafeteira automática reposicionada no centro da praça (X=35.8, Z=18.8) para câmera livre 360°. Recupera toda a vida!",
    "Cubos + Cilindro + Esfera LED · [E] Tomar Café",
    [cafePos.x, 1.0, cafePos.z], [1.1, 1.7, 1.0],
    { isCoffeeMachine: true }
  );

  // ============================================================================
  // 8. JOGADOR, VIDA / SOBREVIVÊNCIA & CUTSCENES (CAFÉ E DANÇA FORTNITE)
  // ============================================================================
  const player = criarAvatarRoblox(24.0, 0.2, 14.2, 0, 0x0284c7, 0xfacc15, 0x0f172a, 1.0);
  const xicaraMao = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.055, 0.13, 12), matPapel);
  xicaraMao.position.set(0, -0.38, 0.14);
  xicaraMao.visible = false;
  player.bD.add(xicaraMao);

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

  const arBoxHelper = new THREE.Box3Helper(new THREE.Box3(), 0x22d3ee);
  arBoxHelper.visible = false;
  scene.add(arBoxHelper);
  let arTimer = 0;

  let isPaused = false;
  let survivalMode = false;
  let isDead = false;
  let inCoffeeCutscene = false;
  let coffeeTimer = 0;
  let inDanceCutscene = false;
  let danceTimer = 0;
  let coffeeBuffTimer = 0;
  let energy = 100;
  let windowsBrokenCount = 0;
  let attackTimer = 0;
  let shakeTimer = 0;
  let stepAccumulator = 0;

  const pState = {
    pos: new THREE.Vector3(24.0, 0.2, 14.2),
    velY: 0,
    yaw: Math.PI,
    camYaw: Math.PI,
    camPitch: 0.34,
    camDist: 7.4,
    onGround: true,
    wasOnRoof: false
  };

  function atualizarBarraLife() {
    const pct = Math.max(0, Math.min(100, Math.round(energy)));
    const fill = document.getElementById('energy-fill');
    fill.style.width = `${pct}%`;
    fill.style.background = pct > 50
      ? 'linear-gradient(90deg, #22c55e, #38bdf8)'
      : (pct > 25 ? 'linear-gradient(90deg, #f59e0b, #facc15)' : '#ef4444');
    document.getElementById('hp-text').textContent = `${pct}%`;
  }

  function aplicarDano(qtd, motivo, isScare = false) {
    if (isDead || inCoffeeCutscene) return;
    if (isScare) SoundEngine.playScare();
    shakeTimer = 0.28;
    const flash = document.getElementById('damage-flash');
    flash.style.opacity = '1';
    setTimeout(() => { flash.style.opacity = '0'; }, 260);

    if (survivalMode) {
      energy = Math.max(0, energy - qtd);
      atualizarBarraLife();
      if (energy <= 0) {
        isDead = true;
        document.getElementById('gameover-reason').textContent = motivo;
        document.getElementById('gameover-modal').style.display = 'flex';
        return;
      }
      showToast(`💔 -${qtd} Life! ${motivo}`, 2400);
    } else {
      // Modo exploração: perde life mas nunca morre (mínimo 10%)
      energy = Math.max(10, energy - qtd);
      atualizarBarraLife();
      showToast(`⚠️ -${qtd} Life! ${motivo} (Tome Café em X=35.8, Z=18.8 para recuperar!)`, 2600);
    }
  }

  document.getElementById('btn-survival').addEventListener('click', (e) => {
    survivalMode = !survivalMode;
    SoundEngine.playSwitch();
    e.target.textContent = survivalMode ? '💀 Sobrevivência: ON' : '💀 Sobrevivência: OFF';
    e.target.classList.toggle('surv-on', survivalMode);
    showToast(
      survivalMode
        ? '💀 Modo Sobrevivência (Dumb Ways to Die) ATIVADO! Cuidado com carros, ave, NPC e telhado!'
        : '🛡️ Modo Sobrevivência DESATIVADO (Você perde Life, mas não morre).',
      3000
    );
  });

  document.getElementById('btn-respawn').addEventListener('click', () => {
    isDead = false;
    energy = 100;
    atualizarBarraLife();
    pState.pos.set(24.0, 0.2, 14.2);
    pState.velY = 0;
    pState.wasOnRoof = false;
    document.getElementById('gameover-modal').style.display = 'none';
    SoundEngine.playCollect();
    showToast("🔄 Você renasceu na Praça do Bloco 734!", 2200);
  });

  // Cutscene de Comemoração (Dança Estilo Fortnite)
  function iniciarCutsceneDanca(mensagem = "💃 VITÓRIA NO BLOCO 734! Comemorando conquista!") {
    if (inCoffeeCutscene || isDead) return;
    inDanceCutscene = true;
    danceTimer = 3.8;
    SoundEngine.playDanceBeat();
    document.getElementById('lb-top').style.height = '48px';
    document.getElementById('lb-bottom').style.height = '48px';
    const cap = document.getElementById('cutscene-caption');
    cap.textContent = mensagem;
    cap.style.display = 'block';
    spawnParticulas(pState.pos.clone().add(new THREE.Vector3(0, 2.2, 0)), [0x38bdf8, 0xfacc15, 0xf472b6, 0x4ade80], 36, 6.5);
  }

  function encerrarCutsceneDanca() {
    inDanceCutscene = false;
    player.bE.rotation.set(0, 0, 0);
    player.bD.rotation.set(0, 0, 0);
    player.pE.rotation.set(0, 0, 0);
    player.pD.rotation.set(0, 0, 0);
    player.torso.rotation.z = 0;
    document.getElementById('lb-top').style.height = '0';
    document.getElementById('lb-bottom').style.height = '0';
    document.getElementById('cutscene-caption').style.display = 'none';
  }

  document.getElementById('btn-dance').addEventListener('click', () => {
    iniciarCutsceneDanca("💃 MODO COMEMORAÇÃO — Dança da Vitória no Centro de Tecnologia!");
  });

  // Cutscene do Cafezinho (Ilha Aberta X=35.8, Z=18.8 — sem parede!)
  function iniciarCutsceneCafe() {
    inCoffeeCutscene = true;
    coffeeTimer = 4.2;
    xicaraMao.visible = true;
    SoundEngine.playCoffee();
    document.getElementById('lb-top').style.height = '48px';
    document.getElementById('lb-bottom').style.height = '48px';
    const cap = document.getElementById('cutscene-caption');
    cap.textContent = "☕ Tomando um cafezinho expresso na Praça do CT... Life 100% restaurado!";
    cap.style.display = 'block';
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
    atualizarBarraLife();
    coffeeBuffTimer = 30.0;
    document.getElementById('buff-tag').style.display = 'inline';
    showToast("☕ Cafezinho concluído! Life 100% + Turbo de Velocidade por 30s!", 3000);
  }

  let toastTimeout = null;
  function showToast(msg, duration = 2600) {
    const el = document.getElementById('toast-msg');
    el.textContent = msg;
    el.style.display = 'block';
    clearTimeout(toastTimeout);
    toastTimeout = setTimeout(() => { el.style.display = 'none'; }, duration);
  }

  // Fullscreen
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

  // Painel de Quests (Pausa o jogo)
  const pauseModal = document.getElementById('pause-modal');
  function togglePauseModal(forceState = null) {
    if (inCoffeeCutscene || inDanceCutscene || isDead) return;
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

  // Raycaster para clicar diretamente nos Totens 3D com o mouse
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  let downX = 0, downY = 0;

  canvas.addEventListener('mousedown', (e) => {
    SoundEngine.init();
    downX = e.clientX; downY = e.clientY;
    isDragging = true; lastMX = e.clientX; lastMY = e.clientY;
  });

  window.addEventListener('mouseup', (e) => {
    isDragging = false;
    if (Math.hypot(e.clientX - downX, e.clientY - downY) < 5 && !isPaused && !isDead) {
      const rect = canvas.getBoundingClientRect();
      pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      const activeCam = (projMode === 'perspectiva') ? perspCamera : orthoCamera;
      raycaster.setFromCamera(pointer, activeCam);
      const hits = raycaster.intersectObjects(clickableTotems, false);
      if (hits.length > 0 && hits[0].object.userData.totem) {
        hits[0].object.userData.totem.acaoFn();
      }
    }
  });

  // Controles de Teclado
  const keys = {};
  window.addEventListener('keydown', (e) => {
    SoundEngine.init();
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
    if (isPaused || inCoffeeCutscene || inDanceCutscene || isDead) return;

    if (e.code === 'Space') {
      e.preventDefault();
      tentarPular();
    } else if (k === 'e') {
      executarInteracaoAR();
    } else if (k === 'f') {
      executarAtaque();
    } else if (k === 'c') {
      iniciarCutsceneDanca("💃 Comemorando no Bloco 734!");
    }
  });
  window.addEventListener('keyup', (e) => { keys[e.key.toLowerCase()] = false; });

  let isDragging = false, lastMX = 0, lastMY = 0;
  window.addEventListener('mousemove', (e) => {
    if (!isDragging || isPaused || inCoffeeCutscene || inDanceCutscene || isDead) return;
    const dx = e.clientX - lastMX;
    const dy = e.clientY - lastMY;
    lastMX = e.clientX; lastMY = e.clientY;
    pState.camYaw -= dx * 0.0065;
    pState.camPitch = Math.max(0.05, Math.min(1.25, pState.camPitch + dy * 0.0055));
  });
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    pState.camDist = Math.max(3.4, Math.min(18.0, pState.camDist + e.deltaY * 0.005));
  }, { passive: false });

  function tentarPular() {
    if (pState.onGround) {
      pState.velY = coffeeBuffTimer > 0 ? 8.8 : 7.5;
      pState.onGround = false;
      SoundEngine.playJump();
    }
  }

  function executarAtaque() {
    if (attackTimer > 0 || isPaused || inCoffeeCutscene || inDanceCutscene || isDead) return;
    attackTimer = 0.36;
    const forward = new THREE.Vector3(Math.sin(pState.yaw), 0, Math.cos(pState.yaw));
    const hitPoint = pState.pos.clone().add(new THREE.Vector3(0, 1.3, 0)).addScaledVector(forward, 1.5);

    let acertou = false;
    breakableWindows.forEach((w) => {
      if (w.broken) return;
      const distXZ = Math.hypot(hitPoint.x - w.pos.x, hitPoint.z - w.pos.z);
      if (distXZ < 2.35 && Math.abs(hitPoint.y - w.pos.y) < 2.2) {
        acertou = true;
        w.hp -= 1;
        w.hideBarTimer = 3.0;
        w.hpBarBg.visible = true;
        w.hpBarFill.visible = true;
        const ratio = Math.max(0, w.hp / w.maxHp);
        w.hpBarFill.scale.x = Math.max(0.01, ratio);
        w.hpBarFill.material.color.setHex(ratio > 0.5 ? 0x22c55e : (ratio > 0.2 ? 0xfacc15 : 0xef4444));
        spawnParticulas(w.pos, 0x93c5fd, 8);
        shakeTimer = 0.14;

        if (w.hp <= 0) {
          w.broken = true;
          w.glass.visible = false;
          w.hpBarBg.visible = false;
          w.hpBarFill.visible = false;
          windowsBrokenCount++;
          document.getElementById('win-broken').textContent = `${windowsBrokenCount}/${breakableWindows.length}`;
          SoundEngine.playGlass();
          spawnParticulas(w.pos, 0xe0f2fe, 24);
          dispararDerramamentoAsfalto(w);
          showToast(`💥 ${w.nome} quebrada! Derramando asfalto CBUQ na fachada...`, 3200);
        } else {
          SoundEngine.tone(420, 0.09, 'triangle', 0.08);
          w.glassMat.color.setHex(w.hp === 2 ? 0x1e3a5f : 0x3b82f6);
          showToast(`⚔️ Impacto em ${w.nome}! Durabilidade: ${w.hp}/${w.maxHp}`, 1800);
        }
      }
    });
    if (!acertou) SoundEngine.tone(180, 0.08, 'sine', 0.05, 0, 90);
  }

  function encontrarObjetoProximo() {
    for (const item of collectibles) {
      if (!item.coletado && pState.pos.distanceTo(new THREE.Vector3(item.x, pState.pos.y, item.z)) < 2.2) {
        return { tipo: 'collectible', item };
      }
    }
    let melhor = null;
    let menorDist = 5.2;
    for (const obj of arObjects) {
      const d = Math.hypot(pState.pos.x - obj.center.x, pState.pos.z - obj.center.z);
      const raioEfetivo = Math.max(obj.size.x, obj.size.z) * 0.5 + 2.4;
      if (d < raioEfetivo && d < menorDist) {
        menorDist = d;
        melhor = { tipo: 'ar', obj };
      }
    }
    return melhor;
  }

  function executarInteracaoAR() {
    if (isPaused || inCoffeeCutscene || inDanceCutscene || isDead) return;
    const alvo = encontrarObjetoProximo();
    if (!alvo) {
      showToast("📡 Aproxime-se de uma mesa, totem, estrutura ou da cafeteira e pressione [E]!", 2200);
      return;
    }

    if (alvo.tipo === 'collectible') {
      alvo.item.coletado = true;
      alvo.item.mesh.visible = false;
      atualizarUIInventario();
      const totalCol = collectibles.filter(c => c.coletado).length;
      if (totalCol === collectibles.length) {
        iniciarCutsceneDanca("🏆 TODAS AS 8 ATIVIDADES E PROVAS CONCLUÍDAS! Parabéns!");
      } else {
        SoundEngine.playCollect();
        iniciarCutsceneDanca(`🎒 Coletado: ${alvo.item.nome} (${totalCol}/8)!`);
      }
      return;
    }

    const obj = alvo.obj;
    if (obj.totem) {
      obj.totem.acaoFn();
      return;
    }
    if (obj.isCoffeeMachine) {
      iniciarCutsceneCafe();
      return;
    }

    SoundEngine.playSwitch();
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

  function dentroBloco734(x, z) {
    if (x < 15.0 || x > 59.0 || z < 20.0 || z > 53.0) return false;
    if (x < 37.0 && z < 42.0) return ((x - 37.0) ** 2 + (z - 42.0) ** 2) <= (22.0 ** 2);
    return true;
  }

  function getGroundY(x, z, curY) {
    if (dentroBloco734(x, z) && curY >= 6.1) return 6.60;
    if (x >= 19.5 && x <= 34.5 && z >= 16.8 && z <= 22.8) return 0.42;
    return z >= 16.0 ? 0.20 : 0.10;
  }

  // ============================================================================
  // 9. LOOP PRINCIPAL DE ANIMAÇÃO E FÍSICA
  // ============================================================================
  const clock = new THREE.Clock();
  atualizarIluminacaoCena();

  function renderActiveCamera() {
    if (projMode === 'perspectiva') {
      renderer.render(scene, perspCamera);
      return;
    }
    // Câmera Ortográfica (Isométrica ou Oblíqua Gabinete)
    const target = new THREE.Vector3(pState.pos.x, pState.pos.y + 1.4, pState.pos.z);
    const dist = 28;
    orthoCamera.position.set(
      target.x + Math.sin(pState.camYaw) * Math.cos(pState.camPitch) * dist,
      target.y + Math.sin(pState.camPitch) * dist,
      target.z + Math.cos(pState.camYaw) * Math.cos(pState.camPitch) * dist
    );
    orthoCamera.lookAt(target);
    orthoCamera.updateProjectionMatrix();

    if (projMode === 'obliqua') {
      // Aplica cisalhamento oblíquo Gabinete 45° na matriz de projeção
      const e = orthoCamera.projectionMatrix.elements;
      e[8] = -0.35;
      e[9] = -0.35;
    }
    renderer.render(scene, orthoCamera);
  }

  function animate() {
    requestAnimationFrame(animate);
    const dt = Math.min(clock.getDelta(), 0.05);
    const elapsed = clock.getElapsedTime();

    if (isPaused || isDead) {
      renderActiveCamera();
      return;
    }

    // Água & Coletáveis
    aguaMesh.position.y = 0.38 + Math.sin(elapsed * 2.8) * 0.015;
    collectibles.forEach((c, idx) => {
      if (!c.coletado) {
        c.mesh.rotation.y += dt * 2.2;
        c.mesh.position.y = c.y + Math.sin(elapsed * 3.0 + idx) * 0.07;
      }
    });

    // Derramamento de Asfalto
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

    // Partículas (Estilhaços / Confetes / Água / Folhas)
    for (let i = particles.length - 1; i >= 0; i--) {
      const s = particles[i];
      s.life -= dt;
      s.vel.y -= 13.5 * dt;
      s.mesh.position.addScaledVector(s.vel, dt);
      s.mesh.rotation.x += dt * 6;
      s.mesh.rotation.y += dt * 5;
      if (s.life <= 0 || s.mesh.position.y < 0.12) {
        scene.remove(s.mesh);
        particles.splice(i, 1);
      }
    }

    // Carros Animados na Via CT + Colisão / Dano / Buzina
    animatedCars.forEach((car) => {
      car.grp.position.x += car.dirX * car.speed * dt;
      if (car.dirX > 0 && car.grp.position.x > 65.0) car.grp.position.x = 5.0;
      if (car.dirX < 0 && car.grp.position.x < 5.0) car.grp.position.x = 65.0;
      car.rodas.forEach(r => { r.rotation.x += car.dirX * car.speed * dt * 2.5; });

      if (car.honkTimer > 0) car.honkTimer -= dt;
      const dx = Math.abs(pState.pos.x - car.grp.position.x);
      const dz = Math.abs(pState.pos.z - car.grp.position.z);
      if (dx < 5.5 && dz < 2.2 && car.honkTimer <= 0 && pState.pos.y < 1.5) {
        SoundEngine.playHonk();
        car.honkTimer = 2.5;
      }
      if (dx < 2.35 && dz < 1.25 && pState.pos.y < 1.5) {
        pState.pos.z = 14.5; // Empurra de volta para a calçada
        aplicarDano(45, `Atropelado pelo ${car.nome} na pista do CT!`, true);
      }
    });

    // Ave Predadora (Árvore do Fundo X=11.5, Z=38.0)
    if (birdScareCooldown > 0) birdScareCooldown -= dt;
    const distAve = Math.hypot(pState.pos.x - aveHome.x, pState.pos.z - aveHome.z);
    if (distAve < 6.2 && pState.pos.y < 6.0) {
      const targetAve = pState.pos.clone().add(new THREE.Vector3(0, 1.4, 0));
      aveGrp.position.lerp(targetAve, dt * 2.6);
      aveGrp.lookAt(targetAve);
      if (distAve < 1.7 && birdScareCooldown <= 0) {
        birdScareCooldown = 2.4;
        aplicarDano(28, "Susto e bicada da Ave Predadora Territorial!", true);
      }
    } else {
      aveGrp.position.lerp(aveHome.clone().add(new THREE.Vector3(0, Math.sin(elapsed * 3) * 0.18, 0)), dt * 2.0);
    }
    asaE.rotation.z = Math.sin(elapsed * 12) * 0.35;
    asaD.rotation.z = -Math.sin(elapsed * 12) * 0.35;

    // NPC Misterioso Parado (X=43.0, Z=19.5) — Susto!
    if (npcScareCooldown > 0) npcScareCooldown -= dt;
    const distNpc = Math.hypot(pState.pos.x - 43.0, pState.pos.z - 19.5);
    if (distNpc < 1.35 && pState.pos.y < 2.0 && npcScareCooldown <= 0) {
      npcScareCooldown = 3.0;
      npcMisterioso.grp.lookAt(pState.pos.x, 0.2, pState.pos.z);
      aplicarDano(25, "👻 SUSTO! Você encostou no NPC Misterioso do Bloco 734!", true);
    }

    // Cutscene de Comemoração (Dança Estilo Fortnite)
    if (inDanceCutscene) {
      danceTimer -= dt;
      const beat = elapsed * 8.5;
      // Palmas e balanço rítmico
      player.bE.rotation.x = -1.2;
      player.bD.rotation.x = -1.2;
      player.bE.rotation.z = -0.35 + Math.abs(Math.sin(beat)) * 0.45;
      player.bD.rotation.z = 0.35 - Math.abs(Math.sin(beat)) * 0.45;
      player.torso.rotation.z = Math.sin(beat * 0.5) * 0.16;
      player.pE.rotation.z = Math.max(0, Math.sin(beat * 0.5)) * 0.25;
      player.pD.rotation.z = Math.min(0, Math.sin(beat * 0.5)) * 0.25;
      player.grp.position.y = pState.pos.y + Math.abs(Math.sin(beat)) * 0.14;

      const angD = elapsed * 1.1;
      perspCamera.position.set(
        pState.pos.x + Math.cos(angD) * 3.1,
        pState.pos.y + 1.9,
        pState.pos.z + Math.sin(angD) * 3.1
      );
      perspCamera.lookAt(pState.pos.x, pState.pos.y + 1.3, pState.pos.z);
      if (danceTimer <= 0) encerrarCutsceneDanca();
      renderActiveCamera();
      return;
    }

    // Cutscene do Cafezinho (Ilha Aberta X=35.8, Z=18.8 — sem parede!)
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
      const angCam = elapsed * 0.95;
      perspCamera.position.set(
        pState.pos.x + Math.cos(angCam) * 2.6,
        pState.pos.y + 1.85,
        pState.pos.z + Math.sin(angCam) * 2.6
      );
      perspCamera.lookAt(pState.pos.x, pState.pos.y + 1.35, pState.pos.z);
      if (coffeeTimer <= 0) {
        player.bD.rotation.x = 0;
        encerrarCutsceneCafe();
      }
      renderActiveCamera();
      return;
    }

    if (coffeeBuffTimer > 0) {
      coffeeBuffTimer -= dt;
      if (coffeeBuffTimer <= 0) document.getElementById('buff-tag').style.display = 'none';
    }

    if (arTimer > 0) {
      arTimer -= dt;
      if (arTimer <= 0) {
        arBoxHelper.visible = false;
        document.getElementById('ar-card').style.display = 'none';
      }
    }

    // Movimento do Jogador + Passos Estilo Roblox
    const speed = coffeeBuffTimer > 0 ? 9.8 : 7.0;
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

      if (!(dentroBloco734(nextX, nextZ) && pState.pos.y < 6.1)) {
        pState.pos.x = nextX;
        pState.pos.z = nextZ;
      }
      if (pState.onGround) {
        stepAccumulator += speed * dt;
        if (stepAccumulator >= 2.2) {
          stepAccumulator = 0;
          SoundEngine.playStep();
        }
      }
    }

    // Escada Lateral (X=59.35, Z=24.0)
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

    // Gravidade e Pouso do Telhado (Fonte / Árvores / Escada)
    const groundY = getGroundY(pState.pos.x, pState.pos.z, pState.pos.y);
    if (pState.pos.y > groundY || pState.velY > 0) {
      pState.velY -= 19.5 * dt;
      pState.pos.y += pState.velY * dt;
      pState.onGround = false;
    }

    if (pState.pos.y <= groundY) {
      if (pState.wasOnRoof && groundY < 1.5) {
        const naFonte = (pState.pos.x >= 19.5 && pState.pos.x <= 34.5 && pState.pos.z >= 16.8 && pState.pos.z <= 22.8);
        const naArvore = arvoresCoords.some(a => Math.hypot(pState.pos.x - a.x, pState.pos.z - a.z) < 3.4) ||
                         (Math.hypot(pState.pos.x - 42.5, pState.pos.z - 17.5) < 3.0);
        const naEscada = distEscada < 2.2;

        if (naFonte) {
          SoundEngine.playCollect();
          spawnParticulas(pState.pos.clone().add(new THREE.Vector3(0, 0.4, 0)), 0x38bdf8, 28);
          showToast("💦 Pouso perfeito na Fonte do CT! Queda 100% amortecida na água.", 3200);
        } else if (naArvore) {
          SoundEngine.playCollect();
          spawnParticulas(pState.pos.clone().add(new THREE.Vector3(0, 1.2, 0)), 0x22c55e, 24);
          showToast("🌿 Pouso amortecido pela copa da Árvore! Descida segura do telhado.", 3200);
        } else if (!naEscada) {
          aplicarDano(55, "Queda livre do telhado do Bloco 734! Use a Escada, a Árvore ou a Fonte.", true);
        }
      }
      pState.pos.y = groundY;
      pState.velY = 0;
      pState.onGround = true;
      pState.wasOnRoof = (groundY >= 6.4);
    }

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

    const prox = encontrarObjetoProximo();
    const promptEl = document.getElementById('interact-prompt');
    if (prox) {
      promptEl.style.display = 'block';
      if (prox.tipo === 'collectible') {
        promptEl.textContent = `🎒 [E] Coletar "${prox.item.nome}" para o Inventário`;
      } else if (prox.obj.totem) {
        promptEl.textContent = `🎛️ [E] ou Clique — ${prox.obj.titulo}`;
      } else if (prox.obj.isCoffeeMachine) {
        promptEl.textContent = `☕ [E] Tomar Cafezinho Expresso (Recuperar 100% Life)`;
      } else {
        promptEl.textContent = `🔍 [E] O que é isso? — ${prox.obj.nome}`;
      }
    } else {
      promptEl.style.display = 'none';
    }

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
    perspCamera.position.lerp(targetCam, 0.22);
    perspCamera.lookAt(pState.pos.x, pState.pos.y + 1.35, pState.pos.z);

    renderActiveCamera();
  }

  window.addEventListener('resize', () => {
    const asp = wrapper.clientWidth / wrapper.clientHeight;
    perspCamera.aspect = asp;
    perspCamera.updateProjectionMatrix();
    orthoCamera.left = -frustumSize * asp * 0.5;
    orthoCamera.right = frustumSize * asp * 0.5;
    orthoCamera.updateProjectionMatrix();
    renderer.setSize(wrapper.clientWidth, wrapper.clientHeight);
  });

  animate();
})();
</script>
</body>
</html>
"""
  return (
      template.replace("__MODO_NOITE__", "true" if modo_noite else "false")
      .replace("__ATIVAR_SOL__", "true" if ativar_sol else "false")
      .replace("__ATIVAR_SPOTS__", "true" if ativar_spots else "false")
  )


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
    turno = st.radio("Turno", ["Dia", "Noite"], index=0)
    modo_noite = turno == "Noite"

  st.subheader("🔦 Luzes & Efeitos Ópticos")
  c1, c2 = st.columns(2)
  with c1:
    ativar_sol = st.checkbox("☀️ Sol Direcional", value=True)
    ativar_sombras = st.checkbox("🌑 Sombras", value=True)
    ativar_reflexos = st.checkbox("🪞 Reflexos", value=True)
  with c2:
    ativar_spots = st.checkbox("🔦 Holofotes/Spots", value=True)
    ativar_caustics = st.checkbox("🌊 Cáusticas", value=True)
    ssaa_2x = st.checkbox("✨ SSAA 2x", value=False)

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
    "🎮 Modo WebGL / Estilo Roblox (Sobrevivência, Totens, Áudio & Dança)",
    "📜 Painel de Quests (Atos I–VIII) & Matrizes",
])


# ------------------------------------------------------------------------------
# ABA 1: RENDERIZADOR RAY CASTING + PICKING COM BANNER NO TOPO
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
          ativar_sol=ativar_sol,
          ativar_spots=ativar_spots,
          ssaa=ssaa_2x,
      )
      st.session_state.ultima_img_arr = img_arr
      st.session_state.ultima_legenda = legenda
      st.session_state.ultimo_pick_info = None

  img_uint8 = (np.clip(st.session_state.ultima_img_arr, 0.0, 1.0) * 255).astype(
      np.uint8
  )
  pil_img = Image.fromarray(img_uint8)

  col_img, col_ctrl = st.columns([2.2, 1.0])

  with col_img:
    st.caption(
        f"📌 **{st.session_state.ultima_legenda}** — Clique sobre qualquer ponto da imagem para inspecionar o objeto."
    )

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
        "Clique na imagem ao lado ou consulte manualmente uma coordenada `(X, Y)`:"
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
# ABA 2: MODO WEBGL / ESTILO ROBLOX COMPLETO & INTERATIVO
# ------------------------------------------------------------------------------
with aba_webgl:
  components.html(
      gerar_html_jogo_webgl(
          modo_noite=modo_noite,
          ativar_sol=ativar_sol,
          ativar_spots=ativar_spots,
      ),
      height=715,
      scrolling=False,
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
      ativar_sol=ativar_sol,
      ativar_spots=ativar_spots,
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
