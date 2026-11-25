import time
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from motor_ct import RenderizadorCT

st.set_page_config(
    page_title="Bloco 734 CT/UFC — Simulador 3D & Ray Casting",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.0rem; padding-bottom: 1.0rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #111827;
        color: #e5e7eb;
        border-radius: 8px 8px 0 0;
        padding: 8px 18px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563eb !important;
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def obter_motor(largura: int, altura: int) -> RenderizadorCT:
    return RenderizadorCT(largura=largura, altura=altura)


# ==============================================================================
# JOGO INTERATIVO 3D (THREE.JS + MINI-QUESTS + IOT + VIGILÂNCIA + QUIZ CRETO)
# ==============================================================================
JOGO_3D_HTML = r"""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8" />
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; user-select: none; }
  body {
    background: #090d16;
    color: #f8fafc;
    font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    overflow: hidden;
    width: 100vw;
    height: 100vh;
  }
  #canvas-container {
    position: relative;
    width: 100%;
    height: 100%;
  }
  canvas { display: block; width: 100%; height: 100%; }

  /* HUD Superior */
  #hud-top {
    position: absolute;
    top: 12px; left: 12px; right: 12px;
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 12px;
    pointer-events: none;
    z-index: 10;
  }
  .hud-card {
    background: rgba(15, 23, 42, 0.84);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(148, 163, 184, 0.25);
    border-radius: 12px;
    padding: 10px 14px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.45);
    pointer-events: auto;
  }
  .hp-bar-wrap {
    width: 210px;
    height: 14px;
    background: #1e293b;
    border-radius: 999px;
    overflow: hidden;
    margin-top: 6px;
    border: 1px solid rgba(255,255,255,0.15);
  }
  #hp-bar {
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, #22c55e, #4ade80);
    transition: width 0.2s ease, background 0.2s ease;
  }
  .stats-row {
    display: flex;
    gap: 12px;
    margin-top: 8px;
    font-size: 12px;
    color: #cbd5e1;
  }
  .badge {
    background: rgba(30, 41, 59, 0.9);
    border: 1px solid rgba(148, 163, 184, 0.25);
    padding: 3px 8px;
    border-radius: 6px;
    font-weight: 600;
  }

  /* Painel de Mini-Quests */
  #quest-list {
    font-size: 12px;
    line-height: 1.55;
    max-width: 340px;
  }
  .q-done { color: #4ade80; text-decoration: line-through; opacity: 0.88; }
  .q-todo { color: #e2e8f0; }

  /* Alertas IoT e Patrimoniais */
  #alert-stack {
    position: absolute;
    top: 108px;
    left: 50%;
    transform: translateX(-50%);
    display: flex;
    flex-direction: column;
    gap: 8px;
    align-items: center;
    z-index: 15;
    pointer-events: none;
    width: min(640px, 92vw);
  }
  .alert-banner {
    padding: 10px 18px;
    border-radius: 10px;
    font-weight: 700;
    font-size: 13.5px;
    text-align: center;
    letter-spacing: 0.2px;
    box-shadow: 0 8px 28px rgba(0,0,0,0.55);
    animation: pulseAlert 0.85s infinite alternate;
  }
  .alert-iot {
    background: rgba(234, 88, 12, 0.94);
    border: 2px solid #fed7aa;
    color: #fff;
  }
  .alert-danger {
    background: rgba(220, 38, 38, 0.95);
    border: 2px solid #fecaca;
    color: #fff;
  }
  .alert-info {
    background: rgba(37, 99, 235, 0.92);
    border: 2px solid #bfdbfe;
    color: #fff;
    animation: none;
  }
  .alert-success {
    background: rgba(22, 163, 74, 0.94);
    border: 2px solid #bbf7d0;
    color: #fff;
    animation: none;
  }
  @keyframes pulseAlert {
    from { transform: scale(0.99); }
    to { transform: scale(1.015); }
  }

  /* Barra de Equilíbrio na Escalada */
  #climb-hud {
    position: absolute;
    bottom: 92px;
    left: 50%;
    transform: translateX(-50%);
    background: rgba(15, 23, 42, 0.9);
    border: 1px solid #38bdf8;
    border-radius: 12px;
    padding: 10px 18px;
    display: none;
    flex-direction: column;
    align-items: center;
    gap: 6px;
    z-index: 16;
  }
  .balance-track {
    position: relative;
    width: 240px;
    height: 14px;
    background: linear-gradient(90deg, #ef4444 0%, #22c55e 35%, #22c55e 65%, #ef4444 100%);
    border-radius: 999px;
    border: 1px solid #fff;
  }
  #balance-needle {
    position: absolute;
    top: -3px;
    left: 50%;
    width: 8px;
    height: 20px;
    background: #ffffff;
    border-radius: 3px;
    transform: translateX(-50%);
    box-shadow: 0 0 8px #000;
  }

  /* Controles Inferiores */
  #hud-bottom {
    position: absolute;
    bottom: 12px; left: 12px; right: 12px;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    gap: 10px;
    z-index: 12;
    pointer-events: none;
  }
  .btn-bar {
    display: flex;
    flex-wrap: wrap;
    gap: 7px;
    pointer-events: auto;
  }
  button.game-btn {
    background: rgba(30, 41, 59, 0.92);
    color: #f8fafc;
    border: 1px solid rgba(148, 163, 184, 0.35);
    border-radius: 8px;
    padding: 7px 11px;
    font-size: 12.5px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s ease;
  }
  button.game-btn:hover { background: #2563eb; border-color: #93c5fd; }
  button.game-btn.active { background: #1d4ed8; border-color: #60a5fa; }

  /* Visor Fotográfico & Flash */
  #photo-overlay {
    position: absolute;
    inset: 0;
    pointer-events: none;
    display: none;
    z-index: 25;
  }
  .viewfinder-box {
    position: absolute;
    inset: 38px;
    border: 2px dashed rgba(255,255,255,0.75);
    border-radius: 14px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    padding: 16px;
  }
  #flash-screen {
    position: absolute;
    inset: 0;
    background: #ffffff;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.35s ease-out;
    z-index: 30;
  }

  /* Cutscene Jumpscare */
  #jumpscare-overlay {
    position: absolute;
    inset: 0;
    background: radial-gradient(circle, #450a0a 0%, #000000 85%);
    display: none;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    z-index: 50;
    animation: shakeScreen 0.1s infinite;
  }
  @keyframes shakeScreen {
    0% { transform: translate(0, 0) rotate(0deg); }
    25% { transform: translate(-10px, 8px) rotate(-1.5deg); }
    50% { transform: translate(10px, -8px) rotate(1.5deg); }
    75% { transform: translate(-8px, -6px) rotate(-1deg); }
    100% { transform: translate(8px, 6px) rotate(1deg); }
  }

  /* Modal Quiz Prof. Creto */
  #creto-modal {
    position: absolute;
    inset: 0;
    background: rgba(2, 6, 23, 0.78);
    backdrop-filter: blur(6px);
    display: none;
    justify-content: center;
    align-items: center;
    z-index: 40;
  }
  .modal-card {
    background: #0f172a;
    border: 2px solid #3b82f6;
    border-radius: 16px;
    padding: 22px 26px;
    width: min(560px, 92vw);
    box-shadow: 0 20px 50px rgba(0,0,0,0.75);
  }
  .quiz-opt {
    display: block;
    width: 100%;
    text-align: left;
    background: #1e293b;
    color: #f8fafc;
    border: 1px solid #475569;
    border-radius: 10px;
    padding: 11px 14px;
    margin-top: 9px;
    font-size: 13.5px;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.15s;
  }
  .quiz-opt:hover { background: #2563eb; border-color: #93c5fd; }
</style>
</head>
<body>
<div id="canvas-container">
  <div id="hud-top">
    <!-- Status do Jogador -->
    <div class="hud-card">
      <div style="display:flex; justify-content:space-between; align-items:center; gap:14px;">
        <span style="font-weight:700; font-size:14px;">🎓 Explorador CT — Bloco 734</span>
        <span id="hp-text" style="font-weight:700; font-size:13px; color:#4ade80;">100 / 100 HP</span>
      </div>
      <div class="hp-bar-wrap"><div id="hp-bar"></div></div>
      <div class="stats-row">
        <span class="badge" id="pos-badge">Pos: (27.0, 0.2, 26.0)</span>
        <span class="badge" id="cam-badge">Câm: 3ª Pessoa</span>
        <span class="badge" id="vigilante-badge" style="color:#93c5fd;">👮 Vigilância: Calma</span>
      </div>
    </div>

    <!-- Checklist de Mini-Quests -->
    <div class="hud-card" id="quest-list">
      <div style="font-weight:700; font-size:13px; margin-bottom:4px; color:#38bdf8;">📋 Mini-Quests & Camadas CT</div>
      <div id="q-foto" class="q-todo">⬜ 📸 Álbum Fotográfico & Solar (<span id="foto-count">0</span>/5)</div>
      <div id="q-arvore" class="q-todo">⬜ 🌳 Subir na Árvore sem Cair</div>
      <div id="q-poste" class="q-todo">⬜ 💡 Subir no Poste sem Cair</div>
      <div id="q-predio" class="q-todo">⬜ ☀️ Telhado & Foto Painéis Fotovoltaicos</div>
      <div id="q-rua" class="q-todo">⬜ 🚸 Atravessar na Faixa e Voltar</div>
      <div id="q-creto" class="q-todo">⬜ 🎓 Quiz CG Rep. Prof. Creto (<span id="creto-count">0</span>/4)</div>
    </div>
  </div>

  <!-- Pilha de Alertas Dinâmicos -->
  <div id="alert-stack"></div>

  <!-- Medidor de Equilíbrio na Escalada -->
  <div id="climb-hud">
    <div style="font-size:12.5px; font-weight:700; color:#7dd3fc;" id="climb-label">🧗 Escalando! Use [W/S] subir/descer e [A/D] equilibrar</div>
    <div class="balance-track"><div id="balance-needle"></div></div>
  </div>

  <!-- Visor Fotográfico -->
  <div id="photo-overlay">
    <div class="viewfinder-box">
      <div style="display:flex; justify-content:space-between; font-weight:700; font-size:14px; text-shadow:0 2px 6px #000;">
        <span>[REC ●] MODO FOTO TÉCNICA CT</span>
        <span id="photo-target-hint">Alvo: Aponte para um ponto de interesse e pressione [P]</span>
      </div>
      <div style="text-align:center; font-size:28px; opacity:0.7;">✛</div>
      <div style="text-align:center; font-size:12px; color:#e2e8f0; text-shadow:0 1px 4px #000;">
        Pontos: Painéis Fotovoltaicos (Telhado) • Escultura • Fonte • Placa 734 • Cachorro Guardião
      </div>
    </div>
  </div>
  <div id="flash-screen"></div>

  <!-- Cutscene Jumpscare -->
  <div id="jumpscare-overlay">
    <div style="font-size:112px; line-height:1;">🦉👁️👁️</div>
    <div style="font-size:32px; font-weight:900; color:#ef4444; margin-top:12px; text-shadow:0 0 20px #000;">
      SUSTO NO CAMPUS CT!
    </div>
    <div style="font-size:15px; color:#fecaca; margin-top:8px;">
      A Coruja Guardiã das Copas te pegou desprevenido!
    </div>
  </div>

  <!-- Modal Quiz Prof. Creto -->
  <div id="creto-modal">
    <div class="modal-card">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <span style="font-weight:800; font-size:16px; color:#60a5fa;">🎓 Representante do Prof. Creto — Computação Gráfica</span>
        <span id="creto-progress" class="badge" style="background:#1d4ed8;">Pergunta 1 / 4</span>
      </div>
      <p id="creto-question" style="margin-top:14px; font-size:14.5px; line-height:1.5; color:#f1f5f9;"></p>
      <div id="creto-options"></div>
      <div id="creto-feedback" style="margin-top:12px; font-size:13px; font-weight:700; min-height:20px;"></div>
      <div style="margin-top:14px; display:flex; justify-content:flex-end;">
        <button class="game-btn" onclick="fecharModalCreto()">Fechar [ESC]</button>
      </div>
    </div>
  </div>

  <!-- Barra Inferior de Ações -->
  <div id="hud-bottom">
    <div class="hud-card" style="font-size:12px; color:#cbd5e1; max-width:440px;">
      <b>Controles:</b> <kbd>WASD</kbd> Mover • <kbd>Shift</kbd> Correr • <kbd>Espaço</kbd> Pular •
      <kbd>E</kbd> Escalar/Descer • <kbd>F</kbd> Falar (Creto) • <kbd>P</kbd> Foto • <kbd>C</kbd> Câmera
    </div>
    <div class="btn-bar">
      <button class="game-btn" onclick="alternarVisorFoto()">📸 Foto [P]</button>
      <button class="game-btn" onclick="interagirEscalada()">🧗 Escalar [E]</button>
      <button class="game-btn" onclick="abrirQuizCreto()">🎓 Rep. Creto [F]</button>
      <button class="game-btn" id="btn-cam-obliqua" onclick="ativarCameraObliqua()">📐 Câm. Oblíqua</button>
      <button class="game-btn" onclick="ciclarCamera()">🎥 Trocar Câm. [C]</button>
      <button class="game-btn" onclick="dispararJumpscare()">👻 Susto</button>
      <button class="game-btn" onclick="resetarJogador()">🔄 Respawn</button>
    </div>
  </div>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
// ============================================================================
// ÁUDIO PROCEDURAL (WEB AUDIO API — SEM DEPENDÊNCIAS EXTERNAS)
// ============================================================================
let audioCtx = null;
function ensureAudio() {
  if (!audioCtx) {
    const AC = window.AudioContext || window.webkitAudioContext;
    if (AC) audioCtx = new AC();
  }
  if (audioCtx && audioCtx.state === 'suspended') audioCtx.resume();
}

function playTone(freq, type, duration, vol = 0.14, slideTo = null) {
  ensureAudio();
  if (!audioCtx) return;
  const osc = audioCtx.createOscillator();
  const gain = audioCtx.createGain();
  osc.type = type;
  osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
  if (slideTo) {
    osc.frequency.exponentialRampToValueAtTime(slideTo, audioCtx.currentTime + duration);
  }
  gain.gain.setValueAtTime(vol, audioCtx.currentTime);
  gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + duration);
  osc.connect(gain);
  gain.connect(audioCtx.destination);
  osc.start();
  osc.stop(audioCtx.currentTime + duration);
}

function somAlertaIoT() { playTone(880, 'sine', 0.16, 0.10, 1180); }
function somApitoVigilante() {
  playTone(2600, 'triangle', 0.14, 0.14, 2900);
  setTimeout(() => playTone(2750, 'triangle', 0.22, 0.14, 2400), 160);
}
function somLatidoCao() {
  playTone(190, 'sawtooth', 0.14, 0.18, 95);
  setTimeout(() => playTone(220, 'sawtooth', 0.16, 0.18, 85), 170);
}
function somFoto() {
  playTone(1200, 'square', 0.05, 0.12, 400);
  setTimeout(() => playTone(1600, 'sine', 0.07, 0.10, 900), 55);
}
function somJumpscare() {
  playTone(110, 'sawtooth', 0.95, 0.28, 520);
  playTone(147, 'square', 0.95, 0.24, 690);
  playTone(740, 'sawtooth', 0.85, 0.22, 180);
}
function somAcerto() {
  playTone(523.25, 'sine', 0.14, 0.16);
  setTimeout(() => playTone(659.25, 'sine', 0.14, 0.16), 120);
  setTimeout(() => playTone(783.99, 'sine', 0.26, 0.18), 240);
}
function somErro() {
  playTone(185, 'sawtooth', 0.35, 0.18, 110);
}

// ============================================================================
// CENA THREE.JS — BLOCO 734 (ALINHADA COM O MOTOR_CT.PY NO 1º OCTANTE)
// ============================================================================
const container = document.getElementById('canvas-container');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x87ceeb);
scene.fog = new THREE.FogExp2(0x9ecae1, 0.0065);

const camera = new THREE.PerspectiveCamera(58, container.clientWidth / container.clientHeight, 0.15, 260);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
container.insertBefore(renderer.domElement, container.firstChild);

// Iluminação
const hemiLight = new THREE.HemisphereLight(0xe0f2fe, 0x475569, 0.72);
scene.add(hemiLight);

const sunLight = new THREE.DirectionalLight(0xfffbeb, 1.05);
sunLight.position.set(12, 38, 8);
sunLight.castShadow = true;
sunLight.shadow.mapSize.set(2048, 2048);
sunLight.shadow.camera.left = -45;
sunLight.shadow.camera.right = 45;
sunLight.shadow.camera.top = 45;
sunLight.shadow.camera.bottom = -45;
scene.add(sunLight);

// Materiais
const matAsfalto = new THREE.MeshStandardMaterial({ color: 0x262930, roughness: 0.88 });
const matFaixa = new THREE.MeshStandardMaterial({ color: 0xf8fafc, roughness: 0.5 });
const matCalcada = new THREE.MeshStandardMaterial({ color: 0xb4b0a8, roughness: 0.85 });
const matAlvenaria = new THREE.MeshStandardMaterial({ color: 0xf1f5f9, roughness: 0.55 });
const matVidro = new THREE.MeshStandardMaterial({ color: 0x0f172a, roughness: 0.15, metalness: 0.75 });
const matGranito = new THREE.MeshStandardMaterial({ color: 0x334155, roughness: 0.4 });
const matAgua = new THREE.MeshStandardMaterial({ color: 0x0284c7, roughness: 0.12, metalness: 0.45, transparent: true, opacity: 0.86 });
const matAco = new THREE.MeshStandardMaterial({ color: 0xe2e8f0, roughness: 0.2, metalness: 0.88 });
const matTronco = new THREE.MeshStandardMaterial({ color: 0x5c3a21, roughness: 0.9 });
const matFolha = new THREE.MeshStandardMaterial({ color: 0x15803d, roughness: 0.75 });
const matSolar = new THREE.MeshStandardMaterial({ color: 0x1e3a8a, roughness: 0.22, metalness: 0.65 });
const matAsfaltoDerramado = new THREE.MeshStandardMaterial({ color: 0x09090b, roughness: 0.25, metalness: 0.3 });

// 1. Piso Urbano (Via + Faixa de Pedestres + Calçada)
const viaMesh = new THREE.Mesh(new THREE.BoxGeometry(66, 0.10, 14), matAsfalto);
viaMesh.position.set(35, 0.05, 9);
viaMesh.receiveShadow = true;
scene.add(viaMesh);

// Faixa de Pedestres em X = 25.5..28.5
for (let zf = 3.2; zf <= 14.8; zf += 1.2) {
  const zebra = new THREE.Mesh(new THREE.BoxGeometry(3.0, 0.02, 0.58), matFaixa);
  zebra.position.set(27.0, 0.11, zf);
  scene.add(zebra);
}

const calcadaMesh = new THREE.Mesh(new THREE.BoxGeometry(66, 0.20, 44), matCalcada);
calcadaMesh.position.set(35, 0.10, 38);
calcadaMesh.receiveShadow = true;
scene.add(calcadaMesh);

// 2. Bloco 734 (Ala Leste + Ala Sul + Fachada Arqueada R=22m)
const alaLeste = new THREE.Mesh(new THREE.BoxGeometry(22, 6.4, 33), matAlvenaria);
alaLeste.position.set(48, 3.4, 36.5);
alaLeste.castShadow = true;
alaLeste.receiveShadow = true;
scene.add(alaLeste);

const alaSul = new THREE.Mesh(new THREE.BoxGeometry(22, 6.4, 11), matAlvenaria);
alaSul.position.set(26, 3.4, 47.5);
alaSul.castShadow = true;
alaSul.receiveShadow = true;
scene.add(alaSul);

const arcoGeo = new THREE.CylinderGeometry(22, 22, 6.4, 48, 1, false, Math.PI, Math.PI * 0.5);
const arcoMesh = new THREE.Mesh(arcoGeo, matAlvenaria);
arcoMesh.position.set(37, 3.4, 42);
arcoMesh.castShadow = true;
arcoMesh.receiveShadow = true;
scene.add(arcoMesh);

// Janelas na curva + Poças de Asfalto Derramado sob janelas
const pocasAsfalto = [];
for (let i = 0; i < 9; i++) {
  const sMid = 0.24 + ((i + 0.5) / 9.0) * 0.72;
  const ang = sMid * (Math.PI / 2.0);
  const wx = 37.0 - 22.06 * Math.cos(ang);
  const wz = 42.0 - 22.06 * Math.sin(ang);
  const jan = new THREE.Mesh(new THREE.BoxGeometry(1.35, 1.8, 1.35), matVidro);
  jan.position.set(wx, 2.3, wz);
  jan.rotation.y = -ang;
  scene.add(jan);

  // Asfalto derramado nas janelas #2, #5 e #8
  if (i === 1 || i === 4 || i === 7) {
    const px = 37.0 - 23.1 * Math.cos(ang);
    const pz = 42.0 - 23.1 * Math.sin(ang);
    const poca = new THREE.Mesh(new THREE.CylinderGeometry(1.15, 1.25, 0.06, 20), matAsfaltoDerramado);
    poca.position.set(px, 0.23, pz);
    scene.add(poca);
    pocasAsfalto.push({ x: px, z: pz, r: 1.25 });
  }
}

// 2B. Painéis Fotovoltaicos no Telhado (Y = 6.60m) + Balizador IoT de Borda
const solarGroup = new THREE.Group();
[29.0, 34.0].forEach((zf) => {
  const painel = new THREE.Mesh(new THREE.BoxGeometry(11.0, 0.12, 2.2), matSolar);
  painel.position.set(48.0, 7.05, zf);
  painel.rotation.x = THREE.MathUtils.degToRad(22);
  painel.castShadow = true;
  solarGroup.add(painel);

  const moldura = new THREE.Mesh(new THREE.BoxGeometry(11.15, 0.08, 2.32), matAco);
  moldura.position.set(48.0, 6.98, zf);
  moldura.rotation.x = THREE.MathUtils.degToRad(22);
  solarGroup.add(moldura);
});
scene.add(solarGroup);

// Balizador IoT laranja piscante na beirada do prédio
const iotBeacon = new THREE.Mesh(
  new THREE.SphereGeometry(0.28, 16, 16),
  new THREE.MeshStandardMaterial({ color: 0xf97316, emissive: 0xea580c, emissiveIntensity: 0.9 })
);
iotBeacon.position.set(38.5, 6.88, 21.2);
scene.add(iotBeacon);

// Escada Lateral de Acesso ao Telhado (X = 59.35, Z = 24.0)
const escadaMesh = new THREE.Mesh(
  new THREE.BoxGeometry(0.55, 6.6, 1.1),
  new THREE.MeshStandardMaterial({ color: 0x334155, metalness: 0.7, roughness: 0.3 })
);
escadaMesh.position.set(59.35, 3.5, 24.0);
scene.add(escadaMesh);

// 3. Fonte / Espelho d'Água
const bordaFonte = new THREE.Mesh(new THREE.BoxGeometry(15.3, 0.34, 6.3), matGranito);
bordaFonte.position.set(27.0, 0.37, 19.8);
scene.add(bordaFonte);

const aguaMesh = new THREE.Mesh(new THREE.BoxGeometry(14.7, 0.36, 5.7), matAgua);
aguaMesh.position.set(27.0, 0.39, 19.8);
scene.add(aguaMesh);

// 4. Escultura Cinética (5 Losangos)
const baseEsc = new THREE.Mesh(new THREE.BoxGeometry(3.4, 0.28, 3.6), matGranito);
baseEsc.position.set(15.0, 0.34, 22.0);
scene.add(baseEsc);

const escGroup = new THREE.Group();
escGroup.position.set(15.0, 0.48, 22.0);
escGroup.rotation.y = THREE.MathUtils.degToRad(-24);
for (let k = 0; k < 5; k++) {
  const fc = k - 2;
  const Lk = 2.35 * (1.0 - 0.055 * fc);
  const frame = new THREE.Mesh(new THREE.TorusGeometry(Lk * 0.58, 0.09, 8, 4), matAco);
  frame.position.set(fc * 0.22, Lk * 0.65 + 0.1, -1.16 + k * 0.58);
  escGroup.add(frame);
}
scene.add(escGroup);

// 5. Árvores Escaláveis + Coruja do Jumpscare
const arvoresData = [
  { nome: 'Árvore Bosque Esquerdo', x: 10.5, z: 28.5, h: 4.5, esc: 1.15, temCoruja: false },
  { nome: 'Árvore Fundo Esquerdo', x: 11.5, z: 38.0, h: 4.9, esc: 1.25, temCoruja: true },
  { nome: 'Árvore Praça Direita', x: 37.5, z: 17.5, h: 3.8, esc: 0.90, temCoruja: false }
];
arvoresData.forEach((a) => {
  const tronco = new THREE.Mesh(new THREE.CylinderGeometry(0.32 * a.esc, 0.38 * a.esc, 3.4 * a.esc, 14), matTronco);
  tronco.position.set(a.x, 0.2 + 1.7 * a.esc, a.z);
  tronco.castShadow = true;
  scene.add(tronco);

  const copa = new THREE.Mesh(new THREE.SphereGeometry(1.75 * a.esc, 16, 16), matFolha);
  copa.position.set(a.x, 0.2 + 3.7 * a.esc, a.z);
  copa.castShadow = true;
  scene.add(copa);

  if (a.temCoruja) {
    const coruja = new THREE.Mesh(
      new THREE.ConeGeometry(0.35, 1.0, 12),
      new THREE.MeshStandardMaterial({ color: 0x312e81, emissive: 0x1e1b4b })
    );
    coruja.position.set(a.x, 0.2 + 5.1 * a.esc, a.z);
    scene.add(coruja);
  }
});

// 6. Postes de Iluminação Escaláveis
const postesData = [
  { nome: 'Poste #1', x: 9.5, z: 21.0, h: 5.8 },
  { nome: 'Poste #2', x: 18.5, z: 16.2, h: 5.8 },
  { nome: 'Poste #3', x: 41.0, z: 16.5, h: 5.8 },
  { nome: 'Poste #4', x: 54.5, z: 17.5, h: 5.8 }
];
postesData.forEach((p) => {
  const mastro = new THREE.Mesh(
    new THREE.CylinderGeometry(0.11, 0.13, 6.0, 12),
    new THREE.MeshStandardMaterial({ color: 0x1e293b, metalness: 0.8 })
  );
  mastro.position.set(p.x, 3.2, p.z);
  mastro.castShadow = true;
  scene.add(mastro);

  const lamp = new THREE.Mesh(
    new THREE.SphereGeometry(0.28, 12, 12),
    new THREE.MeshStandardMaterial({ color: 0xfef08a, emissive: 0xfde047, emissiveIntensity: 0.9 })
  );
  lamp.position.set(p.x, 6.3, p.z + 0.25);
  scene.add(lamp);
});

// 7. Veículos Dinâmicos na Via CT
const carros = [];
function criarCarro(corHex, x0, z0, dir) {
  const g = new THREE.Group();
  const corpo = new THREE.Mesh(
    new THREE.BoxGeometry(4.3, 1.15, 2.0),
    new THREE.MeshStandardMaterial({ color: corHex, metalness: 0.6, roughness: 0.25 })
  );
  corpo.position.y = 0.72;
  corpo.castShadow = true;
  g.add(corpo);

  const cabine = new THREE.Mesh(new THREE.BoxGeometry(2.3, 0.75, 1.75), matVidro);
  cabine.position.set(-0.2 * dir, 1.55, 0);
  g.add(cabine);

  g.position.set(x0, 0.1, z0);
  scene.add(g);
  carros.push({ mesh: g, dir, speed: 6.5 });
}
criarCarro(0xcbd5e1, 22.0, 7.2, 1);
criarCarro(0xdc2626, 48.0, 10.6, -1);

// 8. Cachorro Guardião Interativo (X = 13.5, Z = 17.5)
const caoGroup = new THREE.Group();
const matCao = new THREE.MeshStandardMaterial({ color: 0xb45309, roughness: 0.8 });
const caoCorpo = new THREE.Mesh(new THREE.BoxGeometry(0.95, 0.48, 0.48), matCao);
caoCorpo.position.y = 0.55;
caoCorpo.castShadow = true;
caoGroup.add(caoCorpo);

const caoCabeca = new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.38, 0.38), matCao);
caoCabeca.position.set(0.55, 0.82, 0);
caoGroup.add(caoCabeca);

const caoHome = new THREE.Vector3(13.5, 0.2, 17.5);
caoGroup.position.copy(caoHome);
scene.add(caoGroup);

// 9. Vigilantes Patrimoniais (Protetores do Patrimônio Público)
function criarVigilante(x0, z0) {
  const g = new THREE.Group();
  const tronco = new THREE.Mesh(
    new THREE.CylinderGeometry(0.28, 0.30, 1.1, 12),
    new THREE.MeshStandardMaterial({ color: 0x1e3a8a })
  );
  tronco.position.y = 0.95;
  g.add(tronco);

  const cabeca = new THREE.Mesh(
    new THREE.SphereGeometry(0.22, 12, 12),
    new THREE.MeshStandardMaterial({ color: 0xfdba74 })
  );
  cabeca.position.y = 1.72;
  g.add(cabeca);

  const quepe = new THREE.Mesh(
    new THREE.CylinderGeometry(0.25, 0.24, 0.12, 12),
    new THREE.MeshStandardMaterial({ color: 0x0f172a })
  );
  quepe.position.y = 1.92;
  g.add(quepe);

  g.position.set(x0, 0.2, z0);
  scene.add(g);
  return { mesh: g, home: new THREE.Vector3(x0, 0.2, z0), alertado: false };
}
const vigilantes = [criarVigilante(21.0, 24.5), criarVigilante(34.5, 24.5)];

// 10. Representante do Prof. Creto (Monitor de Computação Gráfica)
const cretoGroup = new THREE.Group();
const cretoBata = new THREE.Mesh(
  new THREE.CylinderGeometry(0.28, 0.32, 1.15, 12),
  new THREE.MeshStandardMaterial({ color: 0x047857 })
);
cretoBata.position.y = 0.95;
cretoGroup.add(cretoBata);
const cretoCabeca = new THREE.Mesh(
  new THREE.SphereGeometry(0.23, 12, 12),
  new THREE.MeshStandardMaterial({ color: 0xfed7aa })
);
cretoCabeca.position.y = 1.75;
cretoGroup.add(cretoCabeca);
const cretoAro = new THREE.Mesh(
  new THREE.TorusGeometry(0.45, 0.04, 8, 24),
  new THREE.MeshStandardMaterial({ color: 0x38bdf8, emissive: 0x0284c7 })
);
cretoAro.rotation.x = Math.PI / 2;
cretoAro.position.y = 0.05;
cretoGroup.add(cretoAro);
cretoGroup.position.set(28.5, 0.2, 24.5);
scene.add(cretoGroup);

// 11. Avatar do Jogador
const playerGroup = new THREE.Group();
const pBody = new THREE.Mesh(
  new THREE.CylinderGeometry(0.26, 0.28, 1.05, 12),
  new THREE.MeshStandardMaterial({ color: 0x2563eb })
);
pBody.position.y = 0.85;
pBody.castShadow = true;
playerGroup.add(pBody);

const pHead = new THREE.Mesh(
  new THREE.SphereGeometry(0.22, 12, 12),
  new THREE.MeshStandardMaterial({ color: 0xffedd5 })
);
pHead.position.y = 1.62;
pHead.castShadow = true;
playerGroup.add(pHead);

const player = {
  pos: new THREE.Vector3(27.0, 0.2, 26.0),
  velY: 0,
  yaw: Math.PI,
  hp: 100,
  onGround: true,
  climbing: null, // { tipo: 'arvore'|'poste'|'escada', nome, x, z, maxH, balance: 0 }
  fountainTimer: 0,
  jumpscareCooldown: 0,
  damageCooldown: 0,
  crossedOut: false,
  quests: {
    fotos: new Set(),
    arvore: false,
    poste: false,
    predioSolar: false,
    travessia: false,
    cretoAcertos: 0
  }
};
playerGroup.position.copy(player.pos);
scene.add(playerGroup);

// ============================================================================
// CÂMERAS (INCLUINDO CÂMERA OBLÍQUA ELEVADA E DESOBSTRUÍDA)
// ============================================================================
const modosCamera = ['3ª Pessoa', '1ª Pessoa', 'Oblíqua Desobstruída', 'Aérea Plongée'];
let camIndex = 0;

function ciclarCamera() {
  camIndex = (camIndex + 1) % modosCamera.length;
  document.getElementById('cam-badge').innerText = 'Câm: ' + modosCamera[camIndex];
  document.getElementById('btn-cam-obliqua').classList.toggle('active', camIndex === 2);
}

function ativarCameraObliqua() {
  camIndex = 2; // 'Oblíqua Desobstruída'
  document.getElementById('cam-badge').innerText = 'Câm: ' + modosCamera[camIndex];
  document.getElementById('btn-cam-obliqua').classList.add('active');
  mostrarAlerta('📐 Câmera Oblíqua ativada em cota elevada e desobstruída!', 'alert-info', 2200);
}

function atualizarCamera() {
  const modo = modosCamera[camIndex];
  if (modo === '1ª Pessoa') {
    const eye = new THREE.Vector3(player.pos.x, player.pos.y + 1.65, player.pos.z);
    const look = new THREE.Vector3(
      eye.x + Math.sin(player.yaw) * 8,
      eye.y - 0.3,
      eye.z + Math.cos(player.yaw) * 8
    );
    camera.position.lerp(eye, 0.45);
    camera.lookAt(look);
  } else if (modo === 'Oblíqua Desobstruída') {
    // Vista oblíqua elevada acima da linha das copas e sem obstrução de paredes
    const target = new THREE.Vector3(player.pos.x, Math.max(player.pos.y + 1.2, 1.5), player.pos.z);
    const desired = new THREE.Vector3(player.pos.x - 14.5, Math.max(player.pos.y + 14.0, 14.5), player.pos.z - 15.5);
    desired.x = Math.max(3.5, Math.min(64.0, desired.x));
    desired.z = Math.max(2.5, Math.min(56.0, desired.z));
    camera.position.lerp(desired, 0.14);
    camera.lookAt(target);
  } else if (modo === 'Aérea Plongée') {
    const desired = new THREE.Vector3(player.pos.x, player.pos.y + 22.0, player.pos.z - 12.0);
    camera.position.lerp(desired, 0.14);
    camera.lookAt(player.pos.x, player.pos.y + 1.0, player.pos.z);
  } else {
    // 3ª Pessoa padrão
    const dist = 6.2;
    const desired = new THREE.Vector3(
      player.pos.x - Math.sin(player.yaw) * dist,
      player.pos.y + 3.1,
      player.pos.z - Math.cos(player.yaw) * dist
    );
    desired.y = Math.max(1.2, desired.y);
    camera.position.lerp(desired, 0.18);
    camera.lookAt(player.pos.x, player.pos.y + 1.4, player.pos.z);
  }
}

// ============================================================================
// SISTEMA DE ALERTAS, DANO, VIGILANTES E JUMPSCARE
// ============================================================================
const alertStack = document.getElementById('alert-stack');
let lastAlertText = '';
let lastAlertTime = 0;

function mostrarAlerta(texto, classe = 'alert-iot', duracao = 2400) {
  const now = performance.now();
  if (texto === lastAlertText && now - lastAlertTime < 1400) return;
  lastAlertText = texto;
  lastAlertTime = now;

  const div = document.createElement('div');
  div.className = 'alert-banner ' + classe;
  div.innerText = texto;
  alertStack.appendChild(div);
  setTimeout(() => { if (div.parentNode) div.remove(); }, duracao);
}

function aplicarDano(qtd, motivo) {
  if (player.damageCooldown > 0) return;
  player.hp = Math.max(0, player.hp - qtd);
  player.damageCooldown = 1.0;
  atualizarHUD();
  mostrarAlerta(`💥 -${qtd} HP: ${motivo}`, 'alert-danger', 2400);
  if (player.hp <= 0) {
    mostrarAlerta('🏥 Energia esgotada! Retornando ao ponto seguro do Bloco 734...', 'alert-danger', 3200);
    resetarJogador();
  }
}

function acionarVigilantes(motivo) {
  let novosAlertados = false;
  vigilantes.forEach((v) => {
    if (!v.alertado) novosAlertados = true;
    v.alertado = true;
  });
  if (novosAlertados) somApitoVigilante();
  document.getElementById('vigilante-badge').innerText = '🚨 Vigilância: PERSEGUINDO!';
  document.getElementById('vigilante-badge').style.color = '#f87171';
  mostrarAlerta(`👮 ALERTA PATRIMONIAL (${motivo})! Suba numa árvore, poste ou escada para escapar!`, 'alert-danger', 3000);
}

function dispararJumpscare() {
  if (player.jumpscareCooldown > 0) return;
  player.jumpscareCooldown = 6.0;
  somJumpscare();
  const ov = document.getElementById('jumpscare-overlay');
  ov.style.display = 'flex';
  setTimeout(() => {
    ov.style.display = 'none';
    mostrarAlerta('😱 SUSTO! A Coruja Guardiã do CT assustou você!', 'alert-danger', 2800);
  }, 1550);
}

// ============================================================================
// MINI-QUEST FOTOGRÁFICA [P]
// ============================================================================
let visorFotoAtivo = false;
function alternarVisorFoto() {
  visorFotoAtivo = !visorFotoAtivo;
  document.getElementById('photo-overlay').style.display = visorFotoAtivo ? 'block' : 'none';
  if (visorFotoAtivo) {
    verificarAlvoFotografico();
  }
}

function verificarAlvoFotografico() {
  const p = player.pos;
  let alvo = null;
  if (p.y >= 6.2 && Math.hypot(p.x - 48.0, p.z - 31.5) < 8.5) {
    alvo = 'Painéis Fotovoltaicos (Cobertura Bloco 734)';
  } else if (Math.hypot(p.x - 15.0, p.z - 22.0) < 6.5) {
    alvo = 'Escultura Cinética em Aço';
  } else if (Math.hypot(p.x - 27.0, p.z - 19.8) < 8.0) {
    alvo = 'Espelho d\'Água & Fachada Arqueada 734';
  } else if (Math.hypot(p.x - caoGroup.position.x, p.z - caoGroup.position.z) < 7.0) {
    alvo = 'Cachorro Guardião do CT';
  } else if (Math.hypot(p.x - 28.5, p.z - 24.5) < 6.5) {
    alvo = 'Representante do Prof. Creto';
  }
  document.getElementById('photo-target-hint').innerText = alvo
    ? `🎯 Alvo Enquadrado: ${alvo} — Pressione [P] para clicar!`
    : 'Alvo: Aproxime-se de um monumento, painel solar ou personagem e clique [P]';
  return alvo;
}

function tirarFoto() {
  somFoto();
  const flash = document.getElementById('flash-screen');
  flash.style.opacity = '0.92';
  setTimeout(() => { flash.style.opacity = '0'; }, 120);

  const alvo = verificarAlvoFotografico() || 'Panorama Geral Bloco 734';
  player.quests.fotos.add(alvo);
  if (alvo.includes('Fotovoltaicos')) {
    player.quests.predioSolar = true;
  }
  mostrarAlerta(`📸 Foto salva: "${alvo}" (${player.quests.fotos.size}/5)!`, 'alert-success', 2600);
  atualizarHUD();
}

// ============================================================================
// ESCALADA SEM CAIR (ÁRVORE, POSTE E ESCADA DO PRÉDIO) [E]
// ============================================================================
function interagirEscalada() {
  if (player.climbing) {
    // Descer com segurança se estiver baixo, ou soltar
    if (player.pos.y <= 1.2) {
      player.pos.y = 0.2;
      player.climbing = null;
      document.getElementById('climb-hud').style.display = 'none';
      mostrarAlerta('✅ Desceu em segurança!', 'alert-info', 1800);
    } else {
      // Soltar no alto faz cair!
      player.climbing = null;
      player.onGround = false;
      player.velY = -1.0;
      document.getElementById('climb-hud').style.display = 'none';
      mostrarAlerta('⚠️ Você soltou no alto! Cuidado com a queda!', 'alert-iot', 2000);
    }
    return;
  }

  // Checar escada do prédio
  if (Math.hypot(player.pos.x - 59.35, player.pos.z - 24.0) < 2.2) {
    player.climbing = { tipo: 'escada', nome: 'Escada do Bloco 734', x: 59.0, z: 24.0, maxH: 6.65, balance: 0 };
    document.getElementById('climb-hud').style.display = 'flex';
    document.getElementById('climb-label').innerText = '🪜 Subindo Escada do Bloco 734! Use [W] para alcançar o telhado solar';
    return;
  }

  // Checar árvores
  for (const a of arvoresData) {
    if (Math.hypot(player.pos.x - a.x, player.pos.z - a.z) < 1.85) {
      player.climbing = { tipo: 'arvore', nome: a.nome, x: a.x, z: a.z, maxH: a.h, balance: 0, coruja: a.temCoruja };
      document.getElementById('climb-hud').style.display = 'flex';
      document.getElementById('climb-label').innerText = `🌳 Escalando ${a.nome}! Use [W] e equilibre com [A/D]`;
      return;
    }
  }

  // Checar postes
  for (const pt of postesData) {
    if (Math.hypot(player.pos.x - pt.x, player.pos.z - pt.z) < 1.65) {
      player.climbing = { tipo: 'poste', nome: pt.nome, x: pt.x, z: pt.z, maxH: pt.h, balance: 0 };
      document.getElementById('climb-hud').style.display = 'flex';
      document.getElementById('climb-label').innerText = `💡 Escalando ${pt.nome}! Use [W] e equilibre com [A/D]`;
      return;
    }
  }

  mostrarAlerta('ℹ️ Aproxime-se de uma Árvore, Poste ou da Escada lateral (X=59.3) e pressione [E].', 'alert-info', 2300);
}

// ============================================================================
// REPRESENTANTE DO PROF. CRETO — QUIZ DE COMPUTAÇÃO GRÁFICA (4 PERGUNTAS)
// ============================================================================
const bancoPerguntasCreto = [
  {
    q: "No ciclo trigonométrico usado em rotações 2D/3D, quais são os sinais de sin(θ) e cos(θ) no 2º Quadrante (90° < θ < 180°)?",
    opts: [
      "sin(θ) > 0 (positivo) e cos(θ) < 0 (negativo)",
      "sin(θ) < 0 (negativo) e cos(θ) > 0 (positivo)",
      "sin(θ) > 0 (positivo) e cos(θ) > 0 (positivo)",
      "sin(θ) < 0 (negativo) e cos(θ) < 0 (negativo)"
    ],
    ans: 0,
    exp: "No 2º Quadrante, a projeção vertical Y (seno) é positiva e a horizontal X (cosseno) é negativa."
  },
  {
    q: "No 3º Quadrante (180° < θ < 270°), quais são os sinais de sin(θ) e cos(θ)?",
    opts: [
      "sin(θ) > 0 e cos(θ) < 0",
      "sin(θ) < 0 (negativo) e cos(θ) < 0 (negativo)",
      "sin(θ) < 0 e cos(θ) > 0",
      "sin(θ) > 0 e cos(θ) > 0"
    ],
    ans: 1,
    exp: "No 3º Quadrante, tanto X (cos θ) quanto Y (sin θ) estão nos semi-eixos negativos."
  },
  {
    q: "No 4º Quadrante (270° < θ < 360°), quais são os sinais de sin(θ) e cos(θ)?",
    opts: [
      "sin(θ) > 0 e cos(θ) > 0",
      "sin(θ) < 0 e cos(θ) < 0",
      "sin(θ) < 0 (negativo) e cos(θ) > 0 (positivo)",
      "sin(θ) > 0 e cos(θ) < 0"
    ],
    ans: 2,
    exp: "No 4º Quadrante, X (cosseno) é positivo à direita e Y (seno) é negativo abaixo do eixo."
  },
  {
    q: "No 1º Quadrante (0° < θ < 90°), quais são os sinais de sin(θ) e cos(θ)?",
    opts: [
      "sin(θ) > 0 (positivo) e cos(θ) > 0 (positivo)",
      "sin(θ) < 0 (negativo) e cos(θ) > 0 (positivo)",
      "sin(θ) > 0 (positivo) e cos(θ) < 0 (negativo)",
      "sin(θ) = 0 e cos(θ) < 0"
    ],
    ans: 0,
    exp: "No 1º Quadrante (assim como no Primeiro Octante 3D), ambas as coordenadas são positivas."
  },
  {
    q: "Por que utilizamos Coordenadas Homogêneas 4D (x, y, z, w) nas matrizes de transformação da Computação Gráfica?",
    opts: [
      "Para permitir representar Translação, Rotação, Escala e Projeção como multiplicação de matrizes 4×4",
      "Para eliminar o eixo Z e renderizar apenas em 2D",
      "Para substituir o cálculo de vetores normais na equação de Phong",
      "Para evitar o uso de seno e cosseno em rotações"
    ],
    ans: 0,
    exp: "Com w=1 (pontos) e w=0 (vetores), a translação torna-se uma operação linear matricial 4×4."
  },
  {
    q: "Na iluminação de Phong, qual componente depende do ângulo entre o vetor refletido R e o vetor do observador V?",
    opts: [
      "Componente Ambiente (Ia · Ka)",
      "Componente Difusa Lambertiana (Id · Kd · (N·L))",
      "Componente Especular (Ie · Ke · (R·V)^n)",
      "Atenuação atmosférica"
    ],
    ans: 2,
    exp: "O brilho especular concentra-se na direção do reflexo espelho R ponderado pelo expoente shininess n."
  },
  {
    q: "Na câmera sintética (M_wc), como são obtidos os eixos ortonormais (xc, yc, zc) a partir de Eye, At e Up?",
    opts: [
      "zc = normalizar(Eye - At), xc = normalizar(Up × zc), yc = zc × xc",
      "zc = Eye + At, xc = Up · At, yc = Eye × Up",
      "xc = Eye - At, yc = Eye + Up, zc = (0,0,0)",
      "zc = Up, xc = At, yc = Eye"
    ],
    ans: 0,
    exp: "O eixo zc aponta da cena para a câmera (Eye - At) no sistema de mão direita."
  }
];

let quizDeck = [];
let quizIdx = 0;

function abrirQuizCreto() {
  // Embaralha e seleciona 4 perguntas novas a cada conversa
  quizDeck = [...bancoPerguntasCreto].sort(() => Math.random() - 0.5).slice(0, 4);
  quizIdx = 0;
  player.quests.cretoAcertos = 0;
  document.getElementById('creto-modal').style.display = 'flex';
  renderizarPerguntaCreto();
}

function fecharModalCreto() {
  document.getElementById('creto-modal').style.display = 'none';
}

function renderizarPerguntaCreto() {
  const item = quizDeck[quizIdx];
  document.getElementById('creto-progress').innerText = `Pergunta ${quizIdx + 1} / 4 • Acertos: ${player.quests.cretoAcertos}`;
  document.getElementById('creto-question').innerText = item.q;
  document.getElementById('creto-feedback').innerText = '';
  const box = document.getElementById('creto-options');
  box.innerHTML = '';
  const letras = ['A', 'B', 'C', 'D'];
  item.opts.forEach((optText, idx) => {
    const btn = document.createElement('button');
    btn.className = 'quiz-opt';
    btn.innerText = `${letras[idx]}) ${optText}`;
    btn.onclick = () => responderCreto(idx);
    box.appendChild(btn);
  });
}

function responderCreto(escolha) {
  const item = quizDeck[quizIdx];
  const fb = document.getElementById('creto-feedback');
  const botoes = document.querySelectorAll('.quiz-opt');
  botoes.forEach((b) => (b.disabled = true));

  if (escolha === item.ans) {
    somAcerto();
    player.quests.cretoAcertos++;
    player.hp = Math.min(100, player.hp + 10);
    fb.style.color = '#4ade80';
    fb.innerText = `🎉 ACERTOU! ${item.exp}`;
    mostrarAlerta(`🎉 Boa! +10 HP (${player.quests.cretoAcertos}/4 acertos com Rep. Prof. Creto)`, 'alert-success', 2200);
  } else {
    somErro();
    fb.style.color = '#f87171';
    fb.innerText = `❌ ERROU! ${item.exp}`;
    aplicarDano(15, 'Resposta incorreta no Quiz de Computação Gráfica!');
  }
  atualizarHUD();

  setTimeout(() => {
    quizIdx++;
    if (quizIdx < 4) {
      renderizarPerguntaCreto();
    } else {
      const aprov = player.quests.cretoAcertos >= 3;
      fb.style.color = aprov ? '#4ade80' : '#fbbf24';
      fb.innerText = aprov
        ? `🏆 Desafio concluído (${player.quests.cretoAcertos}/4)! O Representante do Prof. Creto comemora sua aprovação!`
        : `📚 Desafio encerrado (${player.quests.cretoAcertos}/4). Fale novamente com ele para tentar outra bateria de 4 perguntas!`;
    }
  }, 1700);
}

// ============================================================================
// RESET & ATUALIZAÇÃO DO HUD
// ============================================================================
function resetarJogador() {
  player.pos.set(27.0, 0.2, 26.0);
  player.velY = 0;
  player.hp = 100;
  player.climbing = null;
  player.fountainTimer = 0;
  document.getElementById('climb-hud').style.display = 'none';
  vigilantes.forEach((v) => {
    v.alertado = false;
    v.mesh.position.copy(v.home);
  });
  document.getElementById('vigilante-badge').innerText = '👮 Vigilância: Calma';
  document.getElementById('vigilante-badge').style.color = '#93c5fd';
  atualizarHUD();
}

function atualizarHUD() {
  document.getElementById('hp-text').innerText = `${Math.round(player.hp)} / 100 HP`;
  const bar = document.getElementById('hp-bar');
  bar.style.width = `${player.hp}%`;
  bar.style.background = player.hp > 50
    ? 'linear-gradient(90deg, #22c55e, #4ade80)'
    : player.hp > 25
      ? 'linear-gradient(90deg, #f59e0b, #fbbf24)'
      : 'linear-gradient(90deg, #dc2626, #f87171)';

  document.getElementById('pos-badge').innerText = `Pos: (${player.pos.x.toFixed(1)}, ${player.pos.y.toFixed(1)}, ${player.pos.z.toFixed(1)})`;
  document.getElementById('foto-count').innerText = player.quests.fotos.size;
  document.getElementById('creto-count').innerText = player.quests.cretoAcertos;

  const setQ = (id, ok) => {
    const el = document.getElementById(id);
    if (ok) {
      el.className = 'q-done';
      el.innerText = el.innerText.replace('⬜', '✅');
    }
  };
  setQ('q-foto', player.quests.fotos.size >= 3);
  setQ('q-arvore', player.quests.arvore);
  setQ('q-poste', player.quests.poste);
  setQ('q-predio', player.quests.predioSolar);
  setQ('q-rua', player.quests.travessia);
  setQ('q-creto', player.quests.cretoAcertos >= 4);
}

// ============================================================================
// TECLADO & LOOP DE FÍSICA / REGRAS DO MUNDO
// ============================================================================
const keys = {};
window.addEventListener('keydown', (e) => {
  ensureAudio();
  const k = e.key.toLowerCase();
  keys[k] = true;
  if (k === 'p') {
    if (!visorFotoAtivo) alternarVisorFoto();
    else tirarFoto();
  } else if (k === 'e') {
    interagirEscalada();
  } else if (k === 'f') {
    abrirQuizCreto();
  } else if (k === 'c') {
    ciclarCamera();
  } else if (k === 'escape') {
    fecharModalCreto();
    if (visorFotoAtivo) alternarVisorFoto();
  }
});
window.addEventListener('keyup', (e) => { keys[e.key.toLowerCase()] = false; });

function alturaPiso(x, z) {
  // Telhado do Bloco 734 (Ala Leste + Ala Sul + Arco R=22)
  const naAlaLeste = (x >= 37.0 && x <= 59.0 && z >= 20.0 && z <= 53.0);
  const naAlaSul = (x >= 15.0 && x <= 37.0 && z >= 42.0 && z <= 53.0);
  const noArco = (x <= 37.0 && z <= 42.0 && Math.hypot(x - 37.0, z - 42.0) <= 22.0);
  if ((naAlaLeste || naAlaSul || noArco) && player.pos.y >= 6.1) {
    return 6.60;
  }
  return z < 16.0 ? 0.10 : 0.20;
}

const clock = new THREE.Clock();

function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(clock.getDelta(), 0.05);
  const tNow = clock.getElapsedTime();

  if (player.damageCooldown > 0) player.damageCooldown -= dt;
  if (player.jumpscareCooldown > 0) player.jumpscareCooldown -= dt;

  // Pulsar balizador IoT da beirada
  iotBeacon.scale.setScalar(1.0 + 0.25 * Math.sin(tNow * 8.0));

  // 1. Movimento dos Carros na Via
  carros.forEach((c) => {
    c.mesh.position.x += c.dir * c.speed * dt;
    if (c.dir > 0 && c.mesh.position.x > 65.0) c.mesh.position.x = 5.0;
    if (c.dir < 0 && c.mesh.position.x < 5.0) c.mesh.position.x = 65.0;

    // Colisão com veículo
    if (Math.abs(player.pos.x - c.mesh.position.x) < 2.35 && Math.abs(player.pos.z - c.mesh.position.z) < 1.35 && player.pos.y < 1.5) {
      aplicarDano(25, 'Atropelamento na faixa de veículos!');
    }
  });

  // 2. Lógica de Escalada vs Movimento Livre
  if (player.climbing) {
    const cl = player.climbing;
    player.pos.x = cl.x;
    player.pos.z = cl.z;
    if (keys['w'] || keys['arrowup']) player.pos.y += 2.6 * dt;
    if (keys['s'] || keys['arrowdown']) player.pos.y -= 2.6 * dt;

    // Oscilação de equilíbrio em Árvore e Poste
    if (cl.tipo === 'arvore' || cl.tipo === 'poste') {
      cl.balance += (Math.sin(tNow * 3.7) * 0.55) * dt;
      if (keys['a'] || keys['arrowleft']) cl.balance -= 1.35 * dt;
      if (keys['d'] || keys['arrowright']) cl.balance += 1.35 * dt;
      const pct = 50 + cl.balance * 50;
      document.getElementById('balance-needle').style.left = `${Math.max(4, Math.min(96, pct))}%`;

      if (Math.abs(cl.balance) > 0.96) {
        // Perdeu equilíbrio e caiu!
        player.climbing = null;
        player.onGround = false;
        player.velY = -2.0;
        document.getElementById('climb-hud').style.display = 'none';
        aplicarDano(15, `Perdeu o equilíbrio e caiu de ${cl.nome}!`);
      }
    }

    if (player.climbing && player.pos.y >= cl.maxH) {
      player.pos.y = cl.maxH;
      if (cl.tipo === 'arvore') {
        player.quests.arvore = true;
        mostrarAlerta('🌳 Mini-Quest Concluída: Subiu na Árvore sem cair!', 'alert-success', 2600);
        if (cl.coruja) dispararJumpscare();
      } else if (cl.tipo === 'poste') {
        player.quests.poste = true;
        mostrarAlerta('💡 Mini-Quest Concluída: Subiu no Poste sem cair!', 'alert-success', 2600);
      } else if (cl.tipo === 'escada') {
        // Desembarca no telhado do Bloco 734
        player.pos.set(57.2, 6.60, 24.0);
        player.climbing = null;
        document.getElementById('climb-hud').style.display = 'none';
        mostrarAlerta('☀️ Você chegou ao telhado! Visite os Painéis Fotovoltaicos e tire uma foto [P]!', 'alert-info', 3000);
      }
      atualizarHUD();
    }
    if (player.climbing && player.pos.y <= 0.25) {
      player.pos.y = 0.20;
      player.climbing = null;
      document.getElementById('climb-hud').style.display = 'none';
    }
  } else {
    // Movimento horizontal WASD
    const speed = keys['shift'] ? 9.2 : 5.6;
    let moveF = 0;
    if (keys['w'] || keys['arrowup']) moveF += 1;
    if (keys['s'] || keys['arrowdown']) moveF -= 1;
    if (keys['a'] || keys['arrowleft']) player.yaw += 2.3 * dt;
    if (keys['d'] || keys['arrowright']) player.yaw -= 2.3 * dt;

    const nx = player.pos.x + Math.sin(player.yaw) * moveF * speed * dt;
    const nz = player.pos.z + Math.cos(player.yaw) * moveF * speed * dt;

    // Limites do campus (1º Octante)
    const candX = Math.max(3.0, Math.min(66.5, nx));
    const candZ = Math.max(2.5, Math.min(58.5, nz));

    // Colisão com paredes do térreo do Bloco 734 se player.pos.y < 6.1
    let bloqueado = false;
    if (player.pos.y < 6.1) {
      const inLeste = (candX >= 36.7 && candX <= 59.3 && candZ >= 19.7 && candZ <= 53.3);
      const inSul = (candX >= 14.7 && candX <= 37.3 && candZ >= 41.7 && candZ <= 53.3);
      const inArco = (candX <= 37.3 && candZ <= 42.3 && Math.hypot(candX - 37.0, candZ - 42.0) <= 22.3);
      if (inLeste || inSul || inArco) bloqueado = true;
    }

    if (!bloqueado) {
      player.pos.x = candX;
      player.pos.z = candZ;
    }

    // Pulo & Gravidade
    if (keys[' '] && player.onGround) {
      player.velY = 6.8;
      player.onGround = false;
    }
    const piso = alturaPiso(player.pos.x, player.pos.z);
    if (!player.onGround || player.pos.y > piso + 0.02) {
      player.velY -= 16.5 * dt;
      player.pos.y += player.velY * dt;
      if (player.pos.y <= piso) {
        if (player.velY < -11.5) {
          aplicarDano(20, 'Queda de grande altura!');
        }
        player.pos.y = piso;
        player.velY = 0;
        player.onGround = true;
      }
    } else {
      player.pos.y = piso;
      player.onGround = true;
    }
  }

  playerGroup.position.copy(player.pos);
  playerGroup.rotation.y = player.yaw;

  // 3. Sensores IoT & Regras Ambientais
  // 3A. IoT Beirada do Telhado
  if (player.pos.y >= 6.2) {
    const distArco = Math.abs(Math.hypot(player.pos.x - 37.0, player.pos.z - 42.0) - 22.0);
    const pertoBorda =
      (player.pos.x > 57.5) ||
      (player.pos.z < 21.4) ||
      (player.pos.z > 51.5) ||
      (player.pos.x < 16.5) ||
      (player.pos.x <= 37.2 && player.pos.z <= 42.2 && distArco < 1.6);
    if (pertoBorda) {
      somAlertaIoT();
      mostrarAlerta('📡 SENSOR IoT TELHADO: RISCO DE QUEDA DETECTADO NA BEIRADA DO PRÉDIO!', 'alert-iot', 1800);
    }
  }

  // 3B. IoT Faixa de Veículos & Mini-Quest Travessia
  if (player.pos.y < 1.5 && player.pos.z < 15.8) {
    somAlertaIoT();
    mostrarAlerta('📡 ALERTA IoT VIÁRIO: VOCÊ ESTÁ NA FAIXA DE VEÍCULOS! Use a faixa de pedestres.', 'alert-iot', 1800);
    if (player.pos.x >= 25.0 && player.pos.x <= 29.0 && player.pos.z <= 4.0) {
      player.crossedOut = true;
      mostrarAlerta('🚸 Boa! Metade da travessia concluída — agora retorne pela faixa!', 'alert-info', 2200);
    }
  } else if (player.crossedOut && player.pos.z >= 16.2) {
    player.crossedOut = false;
    player.quests.travessia = true;
    mostrarAlerta('✅ Mini-Quest Concluída: Atravessou a rua na faixa e retornou em segurança!', 'alert-success', 2800);
    atualizarHUD();
  }

  // 3C. Fonte (Espelho d'Água) — Dano se demorar + Vigilante
  const naFonte = (player.pos.x >= 19.65 && player.pos.x <= 34.35 && player.pos.z >= 16.95 && player.pos.z <= 22.65 && player.pos.y < 1.2);
  if (naFonte) {
    player.fountainTimer += dt;
    mostrarAlerta('⚠️ ÁREA RESTRITA: Não é permitido permanecer dentro da fonte!', 'alert-iot', 1500);
    if (player.fountainTimer > 1.6) {
      aplicarDano(10, 'Permanência não autorizada dentro da fonte!');
      acionarVigilantes('Invasão da Fonte');
    }
  } else {
    player.fountainTimer = Math.max(0, player.fountainTimer - dt);
  }

  // 3D. Escultura Cinética — Proteção Patrimonial
  if (Math.hypot(player.pos.x - 15.0, player.pos.z - 22.0) < 2.25 && player.pos.y < 2.5) {
    acionarVigilantes('Aproximação/Escalada na Obra de Arte');
  }

  // 3E. Asfalto Derramado nas Janelas — Dano + Vigilante
  for (const poca of pocasAsfalto) {
    if (Math.hypot(player.pos.x - poca.x, player.pos.z - poca.z) < poca.r && player.pos.y < 1.0) {
      aplicarDano(12, 'Contato com asfalto derramado sob as janelas!');
      acionarVigilantes('Ocorrência nas Janelas (Asfalto)');
      break;
    }
  }

  // 3F. Zona Sombria Ala Sul (segundo gatilho de susto)
  if (Math.hypot(player.pos.x - 16.2, player.pos.z - 40.5) < 1.4 && player.pos.y < 1.5) {
    dispararJumpscare();
  }

  // 4. Cachorro Guardião Territorial (Avança na área oeste)
  const distCaoPlayer = Math.hypot(player.pos.x - caoHome.x, player.pos.z - caoHome.z);
  if (distCaoPlayer < 6.8 && player.pos.y < 1.8) {
    // Avança na direção do jogador
    const dirCao = new THREE.Vector3(player.pos.x - caoGroup.position.x, 0, player.pos.z - caoGroup.position.z);
    if (dirCao.length() > 0.1) {
      dirCao.normalize();
      caoGroup.position.addScaledVector(dirCao, 6.2 * dt);
      caoGroup.rotation.y = Math.atan2(-dirCao.z, dirCao.x);
    }
    if (Math.sin(tNow * 12) > 0.92) somLatidoCao();
    mostrarAlerta('🐕 CUIDADO! Cachorro Guardião avançando em seu território! Afaste-se ou suba em algo!', 'alert-danger', 1600);
    if (Math.hypot(player.pos.x - caoGroup.position.x, player.pos.z - caoGroup.position.z) < 1.25 && player.pos.y < 1.4) {
      aplicarDano(15, 'Mordida do Cachorro Guardião!');
    }
  } else {
    // Retorna calmamente ao posto
    const back = new THREE.Vector3().subVectors(caoHome, caoGroup.position);
    if (back.length() > 0.2) {
      back.normalize();
      caoGroup.position.addScaledVector(back, 2.5 * dt);
    }
  }

  // 5. IA dos Vigilantes Patrimoniais (Perseguição e Fuga Vertical)
  const jogadorEscapouNoAlto = (player.pos.y >= 2.2) || (player.climbing !== null && player.pos.y >= 2.0);
  vigilantes.forEach((v) => {
    if (v.alertado) {
      if (jogadorEscapouNoAlto) {
        // Jogador escapou subindo na árvore, poste ou escada!
        v.alertado = false;
        document.getElementById('vigilante-badge').innerText = '👮 Vigilância: Despistada (Alto)';
        document.getElementById('vigilante-badge').style.color = '#4ade80';
        mostrarAlerta('🧗 Você escapou do vigilante subindo para o alto!', 'alert-success', 2500);
      } else {
        const dirV = new THREE.Vector3(player.pos.x - v.mesh.position.x, 0, player.pos.z - v.mesh.position.z);
        const distV = dirV.length();
        if (distV > 0.1) {
          dirV.normalize();
          v.mesh.position.addScaledVector(dirV, 4.7 * dt);
          v.mesh.rotation.y = Math.atan2(dirV.x, dirV.z);
        }
        if (distV < 1.15) {
          v.alertado = false;
          document.getElementById('vigilante-badge').innerText = '👮 Vigilância: Calma';
          aplicarDano(20, 'Detido pelo Vigilante Patrimonial!');
          player.pos.set(27.0, 0.2, 26.0);
        }
      }
    } else {
      const backV = new THREE.Vector3().subVectors(v.home, v.mesh.position);
      if (backV.length() > 0.25) {
        backV.normalize();
        v.mesh.position.addScaledVector(backV, 2.2 * dt);
      }
    }
  });

  // 6. Proximidade com Representante do Prof. Creto
  if (Math.hypot(player.pos.x - 28.5, player.pos.z - 24.5) < 2.4 && player.pos.y < 1.5) {
    mostrarAlerta('🎓 Representante do Prof. Creto por perto! Pressione [F] para o Quiz de CG (4 perguntas).', 'alert-info', 1600);
  }

  if (visorFotoAtivo) verificarAlvoFotografico();
  atualizarCamera();
  atualizarHUD();
  renderer.render(scene, camera);
}

window.addEventListener('resize', () => {
  camera.aspect = container.clientWidth / container.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(container.clientWidth, container.clientHeight);
});

atualizarHUD();
animate();
</script>
</body>
</html>
"""


# ==============================================================================
# BARRA LATERAL — CONTROLES DO MOTOR ANALÍTICO (RAY CASTING)
# ==============================================================================
with st.sidebar:
    st.title("🏛️ Bloco 734 — CT/UFC")
    st.caption("Motor Ray Casting Vetorizado + Campus Interativo 3D")

    preset_keys = {
        "principal": "Perspectiva Principal (Bloco 734 & Fonte)",
        "obliqua": "Projeção Oblíqua (Gabinete 45° Elevada)",
        "isometrica": "Projeção Ortográfica / Isométrica",
        "1_ponto_fuga": "1 Ponto de Fuga (Eixo Frontal)",
        "2_pontos_fuga": "2 Pontos de Fuga (Horizontal)",
        "3_pontos_fuga": "3 Pontos de Fuga (Plongée Aéreo)",
        "zoom_in": "Zoom In (Escultura & Cáusticas)",
        "zoom_out": "Zoom Out (Vista Geral CT)",
    }
    preset_escolhido = st.selectbox(
        "Preset de Câmera & Projeção:",
        options=list(preset_keys.keys()),
        format_func=lambda k: preset_keys[k],
        index=0,
    )

    resolucao = st.select_slider(
        "Resolução Ray Casting (Quest 13):",
        options=[250, 350, 500],
        value=500,
    )

    col_l1, col_l2 = st.columns(2)
    with col_l1:
        modo_noite = st.toggle("Modo Noturno", value=False)
        ativar_sol = st.toggle("Sol Direcional", value=True)
        ativar_spots = st.toggle("Spots & Holofotes", value=True)
    with col_l2:
        ativar_sombras = st.toggle("Sombras Projetadas", value=True)
        ativar_reflexos = st.toggle("Reflexos (1 Bounce)", value=True)
        ativar_caustics = st.toggle("Cáusticas da Fonte", value=True)

    ssaa = st.toggle("Anti-Aliasing SSAA 2x", value=False)

    st.divider()
    st.markdown(
        """
        **Guia Rápido de Camadas:**
        - **📸 Foto `[P]`**: Inclui ponto técnico nos **Painéis Fotovoltaicos** (telhado).
        - **🧗 Escalar `[E]`**: Árvores, Postes e Escada lateral (equilíbrio `A/D`).
        - **📡 IoT**: Alerta na beirada do telhado e na faixa de veículos.
        - **👮 Vigilância**: Fonte, Escultura e asfalto das janelas acionam vigilantes (fuja para o alto!).
        - **🐕 Cachorro**: Patrulha o gramado oeste e avança ao invadir sua área.
        - **🎓 Rep. Prof. Creto `[F]`**: 4 perguntas de CG (seno/cosseno nos quadrantes).
        """
    )

motor = obter_motor(resolucao, resolucao)
cfg_atual = motor.presets[preset_escolhido]

# ==============================================================================
# ABAS PRINCIPAIS: JOGO 3D INTERATIVO vs MOTOR RAY CASTING ANALÍTICO
# ==============================================================================
tab_jogo, tab_ray = st.tabs([
    "🎮 Campus CT 3D Interativo (Mini-Quests, IoT, Vigilância & Quiz Creto)",
    "🔬 Motor Ray Casting Analítico (motor_ct.py) & Auditoria",
])

with tab_jogo:
    components.html(JOGO_3D_HTML, height=740, scrolling=False)

with tab_ray:
    col_render, col_info = st.columns([1.35, 1.0])

    with col_render:
        t0 = time.perf_counter()
        img_arr, legenda = motor.renderizar_config(
            cfg_atual,
            modo_noite=modo_noite,
            ativar_caustics=ativar_caustics,
            ativar_reflexos=ativar_reflexos,
            ativar_sombras=ativar_sombras,
            ativar_sol=ativar_sol,
            ativar_spots=ativar_spots,
            ssaa=ssaa,
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0
        st.image(
            np.clip(img_arr, 0.0, 1.0),
            caption=f"{legenda} • Tempo: {dt_ms:.1f} ms",
            use_container_width=True,
        )

    with col_info:
        st.subheader("🎯 Quest 12 — Picking Analítico O(1)")
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            px = st.slider("Pixel X", 0, resolucao - 1, resolucao // 2)
        with p_col2:
            py = st.slider("Pixel Y", 0, resolucao - 1, resolucao // 2)

        hit_info = motor.consultar_pick_buffer(px, py)
        if hit_info["hit"]:
            st.success(f"**Objeto Atingido:** {hit_info['nome']}")
            st.markdown(
                f"""
                - **Primitiva:** `{hit_info['tipo_primitiva']}` ({hit_info['grupo']})
                - **Material:** `{hit_info['material']}` (Refletividade: `{hit_info['refletividade']}`)
                - **Distância ($t$):** `{hit_info['distancia']:.2f} m`
                - **Ponto Mundo ($P$):** `{hit_info['ponto_world']}`
                - **Normal ($N$):** `{hit_info['normal_world']}`
                - **Transformação:** `{hit_info['transformacao']}`
                - **Descrição:** {hit_info['descricao']}
                """
            )
        else:
            st.info(f"**{hit_info['nome']}** — {hit_info['descricao']}")

        st.divider()
        st.subheader("📜 Auditoria das Quests (Atos I a VIII)")
        status = motor.obter_status_quests(
            cfg_atual,
            resolucao,
            ativar_sombras=ativar_sombras,
            ativar_sol=ativar_sol,
            ativar_spots=ativar_spots,
        )
        for ato in status.values():
            with st.expander(ato["titulo"], expanded=False):
                for titulo_q, ok, pts in ato["items"]:
                    icone = "✅" if ok else "⚠️"
                    st.write(f"{icone} **{titulo_q}** (`{pts}`)")
