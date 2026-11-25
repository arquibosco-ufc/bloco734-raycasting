import math
import numpy as np


# ==============================================================================
# ÁLGEBRA LINEAR, QUATÉRNIOS E MATRIZES DE TRANSFORMAÇÃO 4x4 (ATO II)
# ==============================================================================
def mat_identidade():
  return np.eye(4, dtype=np.float64)


def mat_translacao(tx, ty, tz):
  M = np.eye(4, dtype=np.float64)
  M[0, 3] = tx
  M[1, 3] = ty
  M[2, 3] = tz
  return M


def mat_escala(sx, sy, sz):
  M = np.eye(4, dtype=np.float64)
  M[0, 0] = sx
  M[1, 1] = sy
  M[2, 2] = sz
  return M


def mat_rotacao_x(rad):
  c, s = math.cos(rad), math.sin(rad)
  M = np.eye(4, dtype=np.float64)
  M[1, 1], M[1, 2] = c, -s
  M[2, 1], M[2, 2] = s, c
  return M


def mat_rotacao_y(rad):
  c, s = math.cos(rad), math.sin(rad)
  M = np.eye(4, dtype=np.float64)
  M[0, 0], M[0, 2] = c, s
  M[2, 0], M[2, 2] = -s, c
  return M


def mat_rotacao_z(rad):
  c, s = math.cos(rad), math.sin(rad)
  M = np.eye(4, dtype=np.float64)
  M[0, 0], M[0, 1] = c, -s
  M[1, 0], M[1, 1] = s, c
  return M


def quat_para_matriz(eixo, angulo_rad):
  """Sub-quest 5.2 (Caminho B): Rotação em eixo arbitrário via Quatérnios unitários."""
  ax = np.asarray(eixo, dtype=np.float64)
  norm = np.linalg.norm(ax)
  if norm < 1e-12:
    return mat_identidade()
  ux, uy, uz = ax / norm
  half = 0.5 * angulo_rad
  w = math.cos(half)
  s = math.sin(half)
  x, y, z = ux * s, uy * s, uz * s

  M = np.eye(4, dtype=np.float64)
  M[0, 0] = 1.0 - 2.0 * (y * y + z * z)
  M[0, 1] = 2.0 * (x * y - w * z)
  M[0, 2] = 2.0 * (x * z + w * y)

  M[1, 0] = 2.0 * (x * y + w * z)
  M[1, 1] = 1.0 - 2.0 * (x * x + z * z)
  M[1, 2] = 2.0 * (y * z - w * x)

  M[2, 0] = 2.0 * (x * z - w * y)
  M[2, 1] = 2.0 * (y * z + w * x)
  M[2, 2] = 1.0 - 2.0 * (x * x + y * y)
  return M


def mat_cisalhamento_xy(sh_xy=0.15, sh_zy=0.0):
  """Quest Bônus 6.1: Matriz de Cisalhamento (Shear)."""
  M = np.eye(4, dtype=np.float64)
  M[1, 0] = sh_xy
  M[1, 2] = sh_zy
  return M


def mat_espelho_plano(ponto_plano, normal_plano):
  """Quest Bônus 6.2: Matriz de Reflexão Especular em Plano Arbitrário."""
  p0 = np.asarray(ponto_plano, dtype=np.float64)
  n = np.asarray(normal_plano, dtype=np.float64)
  n = n / (np.linalg.norm(n) + 1e-12)
  d = -np.dot(n, p0)

  H = np.eye(4, dtype=np.float64)
  H[:3, :3] -= 2.0 * np.outer(n, n)
  H[:3, 3] = -2.0 * d * n
  return H


def aplicar_matriz_pontos(M, pts):
  """Aplica matriz 4x4 a um array (N, 3) de pontos."""
  pts_h = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float64)])
  res = (M @ pts_h.T).T
  return res[:, :3]


def construir_Mwc(eye, at, up):
  """Ato IV (Quests 8 e 9): Matriz Mundo -> Câmera (M_wc)."""
  eye = np.asarray(eye, dtype=np.float64)
  at = np.asarray(at, dtype=np.float64)
  up = np.asarray(up, dtype=np.float64)

  # Eixo Zc aponta de At para Eye (sistema mão direita)
  zc = eye - at
  zc /= np.linalg.norm(zc) + 1e-12

  xc = np.cross(up, zc)
  if np.linalg.norm(xc) < 1e-8:
    up_alt = np.array([0.0, 0.0, 1.0])
    xc = np.cross(up_alt, zc)
  xc /= np.linalg.norm(xc) + 1e-12

  yc = np.cross(zc, xc)

  M = np.eye(4, dtype=np.float64)
  M[0, :3] = xc
  M[1, :3] = yc
  M[2, :3] = zc
  M[0, 3] = -np.dot(xc, eye)
  M[1, 3] = -np.dot(yc, eye)
  M[2, 3] = -np.dot(zc, eye)
  return M


def construir_Mcw(eye, at, up):
  """Matriz inversa Câmera -> Mundo (M_cw)."""
  Mwc = construir_Mwc(eye, at, up)
  Mcw = np.eye(4, dtype=np.float64)
  R = Mwc[:3, :3]
  t = Mwc[:3, 3]
  Mcw[:3, :3] = R.T
  Mcw[:3, 3] = -R.T @ t
  return Mcw


# ==============================================================================
# MATERIAIS E TEXTURAS PROCEDURAIS (ATO I - QUEST 3.1 & 3.2)
# ==============================================================================
class Material:

  def __init__(
      self,
      nome,
      kd,
      ka=None,
      ke=(0.25, 0.25, 0.25),
      shininess=24.0,
      refletividade=0.0,
      textura=None,
      emissivo=(0.0, 0.0, 0.0),
  ):
    self.nome = nome
    self.kd = np.asarray(kd, dtype=np.float64)
    self.ka = (
        np.asarray(ka, dtype=np.float64) if ka is not None else self.kd * 0.45
    )
    self.ke = np.asarray(ke, dtype=np.float64)
    self.shininess = float(shininess)
    self.refletividade = float(refletividade)
    self.textura = textura
    self.emissivo = np.asarray(emissivo, dtype=np.float64)


def criar_textura_placa_734():
  """Gera bitmap procedural da placa '734' sem dependências externas."""
  H, W = 64, 160
  tex = np.ones((H, W, 3), dtype=np.float64) * 0.96
  # Borda escura
  tex[:4, :] = [0.10, 0.14, 0.22]
  tex[-4:, :] = [0.10, 0.14, 0.22]
  tex[:, :4] = [0.10, 0.14, 0.22]
  tex[:, -4:] = [0.10, 0.14, 0.22]

  cor_num = np.array([0.07, 0.10, 0.18])
  # Dígito '7' (cols 22..52)
  tex[14:21, 22:52] = cor_num
  for r in range(21, 52):
    c = int(50 - (r - 21) * 0.72)
    tex[r, c - 4 : c + 3] = cor_num

  # Dígito '3' (cols 66..96)
  tex[14:20, 66:96] = cor_num
  tex[30:36, 72:96] = cor_num
  tex[46:52, 66:96] = cor_num
  tex[14:52, 90:96] = cor_num

  # Dígito '4' (cols 110..142)
  tex[14:36, 110:116] = cor_num
  tex[30:36, 110:142] = cor_num
  tex[14:52, 132:138] = cor_num
  return tex


# ==============================================================================
# PRIMITIVAS ANALÍTICAS PARA RAY CASTING VETORIZADO (ATO I - QUEST 2)
# ==============================================================================
class Primitiva:

  def __init__(
      self,
      pid,
      nome,
      tipo,
      grupo,
      material,
      transformacao_txt,
      descricao,
      recebe_caustica=False,
  ):
    self.pid = pid
    self.nome = nome
    self.tipo = tipo
    self.grupo = grupo
    self.material = material
    self.transformacao_txt = transformacao_txt
    self.descricao = descricao
    self.recebe_caustica = recebe_caustica


class CaixaAABB(Primitiva):

  def __init__(
      self,
      pid,
      nome,
      grupo,
      bmin,
      bmax,
      material,
      transformacao_txt="T(x,y,z) · S(sx,sy,sz)",
      descricao="",
      recebe_caustica=False,
  ):
    super().__init__(
        pid,
        nome,
        "Caixa / Prisma",
        grupo,
        material,
        transformacao_txt,
        descricao,
        recebe_caustica,
    )
    self.bmin = np.asarray(bmin, dtype=np.float64)
    self.bmax = np.asarray(bmax, dtype=np.float64)

  def intersectar(self, O, D):
    invD = np.where(np.abs(D) > 1e-9, 1.0 / D, 1e9)
    t0 = (self.bmin - O) * invD
    t1 = (self.bmax - O) * invD
    tmin = np.minimum(t0, t1)
    tmax = np.maximum(t0, t1)

    t_enter = np.max(tmin, axis=1)
    t_exit = np.min(tmax, axis=1)

    hit = (t_exit >= np.maximum(t_enter, 1e-4)) & (t_enter > 1e-4)
    t_hit = np.where(hit, t_enter, np.inf)

    N = np.zeros_like(D)
    if np.any(hit):
      P = O[hit] + t_hit[hit, None] * D[hit]
      eps = 1e-4
      nh = np.zeros_like(P)
      nh[np.abs(P[:, 0] - self.bmin[0]) < eps, 0] = -1.0
      nh[np.abs(P[:, 0] - self.bmax[0]) < eps, 0] = 1.0
      nh[np.abs(P[:, 1] - self.bmin[1]) < eps, 1] = -1.0
      nh[np.abs(P[:, 1] - self.bmax[1]) < eps, 1] = 1.0
      nh[np.abs(P[:, 2] - self.bmin[2]) < eps, 2] = -1.0
      nh[np.abs(P[:, 2] - self.bmax[2]) < eps, 2] = 1.0
      norm = np.linalg.norm(nh, axis=1, keepdims=True)
      nh = np.where(norm > 1e-8, nh / norm, np.array([0.0, 1.0, 0.0]))
      N[hit] = nh
    return t_hit, N


class Esfera(Primitiva):

  def __init__(
      self,
      pid,
      nome,
      grupo,
      centro,
      raio,
      material,
      transformacao_txt="T(cx,cy,cz) · Q(eixo, θ) · S(r)",
      descricao="",
  ):
    super().__init__(
        pid, nome, "Esfera", grupo, material, transformacao_txt, descricao
    )
    self.c = np.asarray(centro, dtype=np.float64)
    self.r = float(raio)

  def intersectar(self, O, D):
    oc = O - self.c
    b = np.sum(oc * D, axis=1)
    c = np.sum(oc * oc, axis=1) - self.r * self.r
    disc = b * b - c
    valid = disc >= 0.0
    sqrt_d = np.sqrt(np.maximum(disc, 0.0))
    t1 = -b - sqrt_d
    t2 = -b + sqrt_d
    t_hit = np.where(
        valid & (t1 > 1e-4), t1, np.where(valid & (t2 > 1e-4), t2, np.inf)
    )
    hit = np.isfinite(t_hit)
    N = np.zeros_like(D)
    if np.any(hit):
      P = O[hit] + t_hit[hit, None] * D[hit]
      N[hit] = (P - self.c) / self.r
    return t_hit, N


class CilindroY(Primitiva):

  def __init__(
      self,
      pid,
      nome,
      grupo,
      cx,
      cz,
      y0,
      y1,
      raio,
      material,
      transformacao_txt="T(cx,y,cz) · S(r,h,r)",
      descricao="",
  ):
    super().__init__(
        pid, nome, "Cilindro", grupo, material, transformacao_txt, descricao
    )
    self.cx = float(cx)
    self.cz = float(cz)
    self.y0 = float(y0)
    self.y1 = float(y1)
    self.r = float(raio)

  def intersectar(self, O, D):
    ox = O[:, 0] - self.cx
    oz = O[:, 2] - self.cz
    dx = D[:, 0]
    dz = D[:, 2]

    a = dx * dx + dz * dz
    b = 2.0 * (ox * dx + oz * dz)
    c = ox * ox + oz * oz - self.r * self.r
    disc = b * b - 4.0 * a * c

    valid = (disc >= 0.0) & (a > 1e-10)
    sqrt_d = np.sqrt(np.maximum(disc, 0.0))
    inv_2a = np.where(a > 1e-10, 0.5 / a, 0.0)

    t_hit = np.full(O.shape[0], np.inf, dtype=np.float64)
    N = np.zeros_like(D)

    for cand in [(-b - sqrt_d) * inv_2a, (-b + sqrt_d) * inv_2a]:
      y_cand = O[:, 1] + cand * D[:, 1]
      ok = (
          valid
          & (cand > 1e-4)
          & (cand < t_hit)
          & (y_cand >= self.y0)
          & (y_cand <= self.y1)
      )
      if np.any(ok):
        t_hit[ok] = cand[ok]
        px = O[ok, 0] + cand[ok] * D[ok, 0] - self.cx
        pz = O[ok, 2] + cand[ok] * D[ok, 2] - self.cz
        N[ok, 0] = px / self.r
        N[ok, 1] = 0.0
        N[ok, 2] = pz / self.r

    # Tampas superior e inferior
    dy_ok = np.abs(D[:, 1]) > 1e-8
    for y_cap, ny in [(self.y1, 1.0), (self.y0, -1.0)]:
      tc = np.where(dy_ok, (y_cap - O[:, 1]) / D[:, 1], np.inf)
      px = O[:, 0] + tc * D[:, 0] - self.cx
      pz = O[:, 2] + tc * D[:, 2] - self.cz
      ok_c = dy_ok & (tc > 1e-4) & (tc < t_hit) & (px * px + pz * pz <= self.r * self.r)
      if np.any(ok_c):
        t_hit[ok_c] = tc[ok_c]
        N[ok_c] = [0.0, ny, 0.0]

    return t_hit, N


class ArcoBloco734(Primitiva):
  """Parede arqueada tangencial (R=22m, centro=(37,42), quadrante X<=37, Z<=42)."""

  def __init__(
      self,
      pid,
      nome,
      grupo,
      cx,
      cz,
      y0,
      y1,
      raio,
      material,
      transformacao_txt="T(37,0,42) · Cilindro Parcial R=22m",
      descricao="",
  ):
    super().__init__(
        pid,
        nome,
        "Cilindro Parcial (Arco R=22m)",
        grupo,
        material,
        transformacao_txt,
        descricao,
        recebe_caustica=True,
    )
    self.cx = float(cx)
    self.cz = float(cz)
    self.y0 = float(y0)
    self.y1 = float(y1)
    self.r = float(raio)

  def intersectar(self, O, D):
    ox = O[:, 0] - self.cx
    oz = O[:, 2] - self.cz
    dx = D[:, 0]
    dz = D[:, 2]

    a = dx * dx + dz * dz
    b = 2.0 * (ox * dx + oz * dz)
    c = ox * ox + oz * oz - self.r * self.r
    disc = b * b - 4.0 * a * c
    valid = (disc >= 0.0) & (a > 1e-10)
    sqrt_d = np.sqrt(np.maximum(disc, 0.0))
    inv_2a = np.where(a > 1e-10, 0.5 / a, 0.0)

    t_hit = np.full(O.shape[0], np.inf, dtype=np.float64)
    N = np.zeros_like(D)

    for cand in [(-b - sqrt_d) * inv_2a, (-b + sqrt_d) * inv_2a]:
      px = O[:, 0] + cand * D[:, 0]
      py = O[:, 1] + cand * D[:, 1]
      pz = O[:, 2] + cand * D[:, 2]
      ok = (
          valid
          & (cand > 1e-4)
          & (cand < t_hit)
          & (py >= self.y0)
          & (py <= self.y1)
          & (px <= self.cx + 0.05)
          & (pz <= self.cz + 0.05)
      )
      if np.any(ok):
        t_hit[ok] = cand[ok]
        N[ok, 0] = (px[ok] - self.cx) / self.r
        N[ok, 1] = 0.0
        N[ok, 2] = (pz[ok] - self.cz) / self.r

    # Cobertura superior do setor circular (teto)
    dy_ok = np.abs(D[:, 1]) > 1e-8
    tc = np.where(dy_ok, (self.y1 - O[:, 1]) / D[:, 1], np.inf)
    px = O[:, 0] + tc * D[:, 0]
    pz = O[:, 2] + tc * D[:, 2]
    r2 = (px - self.cx) ** 2 + (pz - self.cz) ** 2
    ok_teto = (
        dy_ok
        & (tc > 1e-4)
        & (tc < t_hit)
        & (r2 <= self.r * self.r)
        & (px <= self.cx + 0.05)
        & (pz <= self.cz + 0.05)
    )
    if np.any(ok_teto):
      t_hit[ok_teto] = tc[ok_teto]
      N[ok_teto] = [0.0, 1.0, 0.0]

    return t_hit, N


class ConeY(Primitiva):

  def __init__(
      self,
      pid,
      nome,
      grupo,
      cx,
      cz,
      y_base,
      altura,
      raio_base,
      material,
      transformacao_txt="T(cx,y,cz) · S(r,h,r)",
      descricao="",
  ):
    super().__init__(
        pid, nome, "Cone", grupo, material, transformacao_txt, descricao
    )
    self.cx = float(cx)
    self.cz = float(cz)
    self.y0 = float(y_base)
    self.h = float(altura)
    self.y1 = self.y0 + self.h
    self.r = float(raio_base)
    self.k = (self.r / self.h) ** 2

  def intersectar(self, O, D):
    ox = O[:, 0] - self.cx
    oz = O[:, 2] - self.cz
    oy = O[:, 1] - self.y1

    dx, dy, dz = D[:, 0], D[:, 1], D[:, 2]
    a = dx * dx + dz * dz - self.k * dy * dy
    b = 2.0 * (ox * dx + oz * dz - self.k * oy * dy)
    c = ox * ox + oz * oz - self.k * oy * oy

    disc = b * b - 4.0 * a * c
    valid = (disc >= 0.0) & (np.abs(a) > 1e-10)
    sqrt_d = np.sqrt(np.maximum(disc, 0.0))
    inv_2a = np.where(np.abs(a) > 1e-10, 0.5 / a, 0.0)

    t_hit = np.full(O.shape[0], np.inf, dtype=np.float64)
    N = np.zeros_like(D)

    for cand in [(-b - sqrt_d) * inv_2a, (-b + sqrt_d) * inv_2a]:
      py = O[:, 1] + cand * D[:, 1]
      ok = (
          valid
          & (cand > 1e-4)
          & (cand < t_hit)
          & (py >= self.y0)
          & (py <= self.y1)
      )
      if np.any(ok):
        t_hit[ok] = cand[ok]
        px = O[ok, 0] + cand[ok] * D[ok, 0] - self.cx
        pz = O[ok, 2] + cand[ok] * D[ok, 2] - self.cz
        ry = np.sqrt(px * px + pz * pz) + 1e-9
        nx = px / ry
        nz = pz / ry
        ny = self.r / self.h
        nh = np.stack([nx, np.full_like(nx, ny), nz], axis=1)
        nh /= np.linalg.norm(nh, axis=1, keepdims=True) + 1e-9
        N[ok] = nh

    # Base circular
    dy_ok = np.abs(D[:, 1]) > 1e-8
    tc = np.where(dy_ok, (self.y0 - O[:, 1]) / D[:, 1], np.inf)
    px = O[:, 0] + tc * D[:, 0] - self.cx
    pz = O[:, 2] + tc * D[:, 2] - self.cz
    ok_b = dy_ok & (tc > 1e-4) & (tc < t_hit) & (px * px + pz * pz <= self.r * self.r)
    if np.any(ok_b):
      t_hit[ok_b] = tc[ok_b]
      N[ok_b] = [0.0, -1.0, 0.0]

    return t_hit, N


class MalhaTriangulos(Primitiva):
  """Malha de triângulos com interseção Möller–Trumbore vetorizada (usada nos losangos e abrigo cisalhado)."""

  def __init__(
      self,
      pid,
      nome,
      grupo,
      triangulos,
      material,
      transformacao_txt="Malha Poligonal Transformada (M 4x4)",
      descricao="",
  ):
    super().__init__(
        pid,
        nome,
        "Malha Triangular",
        grupo,
        material,
        transformacao_txt,
        descricao,
    )
    self.tris = np.asarray(triangulos, dtype=np.float64)  # (T, 3, 3)
    e1 = self.tris[:, 1] - self.tris[:, 0]
    e2 = self.tris[:, 2] - self.tris[:, 0]
    fn = np.cross(e1, e2)
    norms = np.linalg.norm(fn, axis=1, keepdims=True)
    self.normals = np.where(norms > 1e-9, fn / norms, np.array([0.0, 1.0, 0.0]))

    # AABB envoltória para aceleração rápida
    all_v = self.tris.reshape(-1, 3)
    self.bmin = np.min(all_v, axis=0) - 0.05
    self.bmax = np.max(all_v, axis=0) + 0.05

  def intersectar(self, O, D):
    N_rays = O.shape[0]
    t_hit = np.full(N_rays, np.inf, dtype=np.float64)
    N = np.zeros_like(D)

    # Teste rápido AABB
    invD = np.where(np.abs(D) > 1e-9, 1.0 / D, 1e9)
    t0 = (self.bmin - O) * invD
    t1 = (self.bmax - O) * invD
    t_enter = np.max(np.minimum(t0, t1), axis=1)
    t_exit = np.min(np.maximum(t0, t1), axis=1)
    mask_box = (t_exit >= np.maximum(t_enter, 1e-4)) & (t_enter > 1e-4)
    if not np.any(mask_box):
      return t_hit, N

    idx_box = np.where(mask_box)[0]
    Ob = O[idx_box]
    Db = D[idx_box]

    for ti in range(self.tris.shape[0]):
      v0, v1, v2 = self.tris[ti]
      e1 = v1 - v0
      e2 = v2 - v0
      h = np.cross(Db, e2)
      a = np.sum(h * e1, axis=1)
      valid = np.abs(a) > 1e-8
      f = np.where(valid, 1.0 / a, 0.0)
      s = Ob - v0
      u = f * np.sum(s * h, axis=1)
      valid &= (u >= 0.0) & (u <= 1.0)
      q = np.cross(s, e1)
      v = f * np.sum(Db * q, axis=1)
      valid &= (v >= 0.0) & (u + v <= 1.0)
      t = f * np.sum(e2 * q, axis=1)
      valid &= (t > 1e-4) & (t < t_hit[idx_box])

      if np.any(valid):
        sub = idx_box[valid]
        t_hit[sub] = t[valid]
        fn = self.normals[ti]
        flip = np.sum(Db[valid] * fn, axis=1) > 0
        nh = np.where(flip[:, None], -fn, fn)
        N[sub] = nh

    return t_hit, N


def gerar_caixa_triangulos(w, h, d, M):
  """Constrói uma caixa centrada na origem transformada pela matriz 4x4 M."""
  hw, hh, hd = w * 0.5, h * 0.5, d * 0.5
  v = np.array(
      [
          [-hw, -hh, -hd],
          [hw, -hh, -hd],
          [hw, hh, -hd],
          [-hw, hh, -hd],
          [-hw, -hh, hd],
          [hw, -hh, hd],
          [hw, hh, hd],
          [-hw, hh, hd],
      ],
      dtype=np.float64,
  )
  vt = aplicar_matriz_pontos(M, v)
  faces = [
      (0, 2, 1),
      (0, 3, 2),
      (4, 5, 6),
      (4, 6, 7),
      (0, 1, 5),
      (0, 5, 4),
      (2, 3, 7),
      (2, 7, 6),
      (0, 4, 7),
      (0, 7, 3),
      (1, 2, 6),
      (1, 6, 5),
  ]
  return [np.array([vt[a], vt[b], vt[c]]) for (a, b, c) in faces]


# ==============================================================================
# RENDERIZADOR PRINCIPAL — BLOCO 734 (CT/UFC)
# ==============================================================================
class RenderizadorCT:

  def __init__(self, largura=500, altura=500):
    self.W = int(largura)
    self.H = int(altura)
    self.placa_tex = criar_textura_placa_734()
    self.materiais = self._criar_materiais()
    self.primitivas = self._construir_cena_bloco734()
    self.presets = self._definir_presets()
    self.pick_buffer = None

  def _criar_materiais(self):
    return {
        "asfalto": Material(
            "Asfalto CBUQ (Textura Procedural)",
            kd=(0.16, 0.17, 0.20),
            ke=(0.12, 0.12, 0.14),
            shininess=12.0,
            textura="asfalto",
        ),
        "calcada": Material(
            "Calçada Intertravada (Textura Procedural)",
            kd=(0.68, 0.67, 0.64),
            ke=(0.15, 0.15, 0.15),
            shininess=10.0,
            textura="calcada",
        ),
        "alvenaria": Material(
            "Alvenaria Branca Bloco 734",
            kd=(0.93, 0.94, 0.96),
            ke=(0.25, 0.25, 0.28),
            shininess=22.0,
        ),
        "vidro": Material(
            "Vidro Fumê Arquitetônico",
            kd=(0.07, 0.12, 0.19),
            ke=(0.85, 0.92, 1.0),
            shininess=96.0,
            refletividade=0.48,
        ),
        "esquadria": Material(
            "Alumínio Anodizado Escuro",
            kd=(0.14, 0.16, 0.20),
            ke=(0.45, 0.48, 0.52),
            shininess=45.0,
        ),
        "aco_escultura": Material(
            "Aço Inox Polido (Escultura Cinética)",
            kd=(0.90, 0.92, 0.95),
            ke=(0.95, 0.95, 1.0),
            shininess=88.0,
            refletividade=0.35,
        ),
        "concreto": Material(
            "Concreto Estrutural",
            kd=(0.42, 0.45, 0.50),
            ke=(0.15, 0.15, 0.15),
            shininess=14.0,
        ),
        "granito": Material(
            "Granito Escuro da Fonte",
            kd=(0.20, 0.23, 0.28),
            ke=(0.35, 0.38, 0.42),
            shininess=36.0,
        ),
        "agua": Material(
            "Água Refrativa n=1.333 (Textura Ondulatória)",
            kd=(0.06, 0.45, 0.68),
            ke=(0.90, 0.96, 1.0),
            shininess=110.0,
            refletividade=0.52,
            textura="agua",
        ),
        "tronco": Material(
            "Madeira / Tronco Natural",
            kd=(0.36, 0.23, 0.13),
            ke=(0.08, 0.06, 0.04),
            shininess=8.0,
        ),
        "folha_escura": Material(
            "Folhagem Verde Floresta",
            kd=(0.10, 0.46, 0.18),
            ke=(0.12, 0.18, 0.12),
            shininess=14.0,
        ),
        "folha_clara": Material(
            "Folhagem Verde Tropical",
            kd=(0.16, 0.62, 0.24),
            ke=(0.14, 0.22, 0.14),
            shininess=16.0,
        ),
        "cone_laranja": Material(
            "PVC Laranja Sinalização",
            kd=(0.95, 0.36, 0.06),
            ke=(0.45, 0.35, 0.30),
            shininess=28.0,
        ),
        "cone_espelhado": Material(
            "PVC Ciano Espelhado (Plano Arbitrário)",
            kd=(0.08, 0.72, 0.85),
            ke=(0.55, 0.75, 0.85),
            shininess=42.0,
            refletividade=0.22,
        ),
        "metal_galv": Material(
            "Aço Galvanizado Cisalhado",
            kd=(0.62, 0.66, 0.72),
            ke=(0.75, 0.78, 0.82),
            shininess=60.0,
            refletividade=0.25,
        ),
        "placa_734": Material(
            "Placa Identificadora 734 (Bitmap Procedural)",
            kd=(0.95, 0.95, 0.95),
            ke=(0.25, 0.25, 0.25),
            shininess=20.0,
            textura="placa734",
        ),
        "spot_luz": Material(
            "Emissor Luminoso Spot / Poste",
            kd=(0.95, 0.92, 0.75),
            ke=(1.0, 1.0, 1.0),
            shininess=80.0,
            emissivo=(0.85, 0.82, 0.55),
        ),
        "carro_prata": Material(
            "Pintura Automotiva Prata Metálico",
            kd=(0.74, 0.78, 0.83),
            ke=(0.85, 0.88, 0.92),
            shininess=75.0,
            refletividade=0.30,
        ),
        "carro_vermelho": Material(
            "Pintura Automotiva Vermelho Esportivo",
            kd=(0.82, 0.11, 0.14),
            ke=(0.88, 0.55, 0.55),
            shininess=78.0,
            refletividade=0.28,
        ),
        "cafeteira": Material(
            "Aço Escovado Cafeteira Expresso CT",
            kd=(0.24, 0.27, 0.33),
            ke=(0.75, 0.80, 0.88),
            shininess=68.0,
            refletividade=0.25,
        ),
        "ave": Material(
            "Plumagem Azul-Índigo (Ave Territorial)",
            kd=(0.18, 0.16, 0.48),
            ke=(0.25, 0.25, 0.45),
            shininess=24.0,
        ),
    }

  def _construir_cena_bloco734(self):
    """Constrói todo o cenário no Primeiro Octante (X, Y, Z >= 0)."""
    prims = []
    pid = 1

    def add(p):
      nonlocal pid
      prims.append(p)
      pid += 1

    # 1. Piso Urbano (Via Asfáltica + Calçada)
    add(
        CaixaAABB(
            pid,
            "Via Asfáltica CT (CBUQ)",
            "Pavimentação & Urbanismo",
            [2.0, 0.0, 2.0],
            [68.0, 0.10, 16.0],
            self.materiais["asfalto"],
            "T(35, 0.05, 9) · S(66, 0.1, 14)",
            "Pista frontal do Centro de Tecnologia com granulometria procedural e faixas duplas amarelas.",
        )
    )
    add(
        CaixaAABB(
            pid,
            "Calçada da Praça Bloco 734",
            "Pavimentação & Urbanismo",
            [2.0, 0.0, 16.0],
            [68.0, 0.20, 60.0],
            self.materiais["calcada"],
            "T(35, 0.10, 38) · S(66, 0.2, 44)",
            "Passeio em blocos intertravados com rejunte procedural no Primeiro Octante.",
            recebe_caustica=True,
        )
    )

    # 2. Bloco 734 (Alas Retangulares + Fachada Arqueada R=22m + Placa 734)
    add(
        CaixaAABB(
            pid,
            "Bloco 734 — Ala Leste",
            "Edificação Principal",
            [37.0, 0.20, 20.0],
            [59.0, 6.60, 53.0],
            self.materiais["alvenaria"],
            "T(48, 3.4, 36.5) · S(22, 6.4, 33)",
            "Volume principal retangular do Bloco 734 com cobertura acessível por escada lateral.",
            recebe_caustica=True,
        )
    )
    add(
        CaixaAABB(
            pid,
            "Bloco 734 — Ala Sul",
            "Edificação Principal",
            [15.0, 0.20, 42.0],
            [37.0, 6.60, 53.0],
            self.materiais["alvenaria"],
            "T(26, 3.4, 47.5) · S(22, 6.4, 11)",
            "Volume posterior do Bloco 734 concordando tangencialmente com o arco frontal.",
        )
    )
    add(
        ArcoBloco734(
            pid,
            "Bloco 734 — Fachada Arqueada (R=22m)",
            "Edificação Principal",
            37.0,
            42.0,
            0.20,
            6.60,
            22.0,
            self.materiais["alvenaria"],
            "T(37, 0.2, 42) · Cilindro Parcial R=22m",
            "Fachada curva característica do Bloco 734 que recebe reflexos e cáusticas da fonte.",
        )
    )

    # Placa 734 na platibanda
    ang_p = 0.14 * (math.pi / 2.0)
    px_p = 37.0 - 22.08 * math.cos(ang_p)
    pz_p = 42.0 - 22.08 * math.sin(ang_p)
    add(
        CaixaAABB(
            pid,
            "Placa Identificadora '734'",
            "Edificação Principal",
            [px_p - 0.35, 5.20, pz_p - 1.55],
            [px_p + 0.35, 6.25, pz_p + 1.55],
            self.materiais["placa_734"],
            f"T({px_p:.2f}, 5.72, {pz_p:.2f}) · Textura Procedural 734",
            "Identificação institucional do Bloco 734 gerada por textura procedural.",
        )
    )

    # Porta de vidro + 9 Janelas na curva
    ang_porta = 0.13 * (math.pi / 2.0)
    add(
        CaixaAABB(
            pid,
            "Porta Principal de Vidro (Bloco 734)",
            "Esquadrias & Vidros",
            [
                37.0 - 22.12 * math.cos(ang_porta) - 0.35,
                0.22,
                42.0 - 22.12 * math.sin(ang_porta) - 1.15,
            ],
            [
                37.0 - 22.12 * math.cos(ang_porta) + 0.35,
                2.75,
                42.0 - 22.12 * math.sin(ang_porta) + 1.15,
            ],
            self.materiais["vidro"],
            "T(arco s=0.13) · Vidro Refletivo",
            "Acesso principal em vidro fumê com reflexão especular.",
        )
    )

    for i in range(9):
      s_mid = 0.24 + ((i + 0.5) / 9.0) * 0.72
      ang_w = s_mid * (math.pi / 2.0)
      wx = 37.0 - 22.08 * math.cos(ang_w)
      wz = 42.0 - 22.08 * math.sin(ang_w)
      add(
          CaixaAABB(
              pid,
              f"Janela Fachada Curva #{i+1}",
              "Esquadrias & Vidros",
              [wx - 0.72, 1.38, wz - 0.72],
              [wx + 0.72, 3.22, wz + 0.72],
              self.materiais["vidro"],
              f"T({wx:.1f}, 2.3, {wz:.1f}) · Arco s={s_mid:.2f}",
              "Janela modulada ao longo do arco R=22m do Bloco 734.",
          )
      )

    # 3. Espelho d'Água / Fonte Frontal + 3 Spots Submersos
    add(
        CaixaAABB(
            pid,
            "Borda Granito — Espelho d'Água",
            "Fonte & Cáusticas",
            [19.35, 0.20, 16.65],
            [34.65, 0.54, 22.95],
            self.materiais["granito"],
            "T(27.0, 0.37, 19.8) · S(15.3, 0.34, 6.3)",
            "Tanque em granito escuro que delimita a lâmina d'água frontal.",
        )
    )
    add(
        CaixaAABB(
            pid,
            "Lâmina d'Água (Fonte Cáustica n=1.333)",
            "Fonte & Cáusticas",
            [19.65, 0.22, 16.95],
            [34.35, 0.56, 22.65],
            self.materiais["agua"],
            "T(27.0, 0.39, 19.8) · Refração n=1.333",
            "Superfície líquida ondulatória que reflete o céu/edifício e projeta cáusticas.",
        )
    )
    for idx_sp, sx in enumerate([22.0, 27.0, 32.0], start=1):
      add(
          Esfera(
              pid,
              f"Spot Submerso #{idx_sp} (Fonte)",
              "Iluminação & Spots",
              [sx, 0.36, 17.20],
              0.18,
              self.materiais["spot_luz"],
              f"T({sx}, 0.36, 17.20) · Spot Submerso",
              "Projetor subaquático responsável pelos filamentos de cáustica na fachada.",
          )
      )

    # 4. Escultura Cinética em Aço (5 Losangos rotacionados a 45° e -24° em Y)
    add(
        CaixaAABB(
            pid,
            "Base da Escultura Cinética",
            "Escultura Cinética",
            [13.3, 0.20, 20.2],
            [16.7, 0.48, 23.8],
            self.materiais["concreto"],
            "T(15.0, 0.34, 22.0) · S(3.4, 0.28, 3.6)",
            "Pedestal em concreto armado da escultura geométrica do CT.",
        )
    )
    tris_esc = []
    M_base_esc = mat_translacao(15.0, 0.48, 22.0) @ mat_rotacao_y(
        math.radians(-24.0)
    )
    for k in range(5):
      fc = k - 2
      Lk = 2.35 * (1.0 - 0.055 * fc)
      rk = Lk / math.sqrt(2.0)
      M_quad = (
          M_base_esc
          @ mat_translacao(fc * 0.22, rk + 0.05, -1.16 + k * 0.58)
          @ mat_rotacao_z(math.pi / 4.0)
      )
      esp, prof = 0.14, 0.16
      tris_esc.extend(
          gerar_caixa_triangulos(
              Lk + esp, esp, prof, M_quad @ mat_translacao(0, Lk * 0.5, 0)
          )
      )
      tris_esc.extend(
          gerar_caixa_triangulos(
              Lk + esp, esp, prof, M_quad @ mat_translacao(0, -Lk * 0.5, 0)
          )
      )
      tris_esc.extend(
          gerar_caixa_triangulos(
              esp, Lk + esp, prof, M_quad @ mat_translacao(Lk * 0.5, 0, 0)
          )
      )
      tris_esc.extend(
          gerar_caixa_triangulos(
              esp, Lk + esp, prof, M_quad @ mat_translacao(-Lk * 0.5, 0, 0)
          )
      )
    add(
        MalhaTriangulos(
            pid,
            "Escultura Cinética em Aço (5 Losangos)",
            "Escultura Cinética",
            tris_esc,
            self.materiais["aco_escultura"],
            "T(15,0.48,22) · Ry(-24°) · Rz(45°) · S(Lk)",
            "Cinco quadros metálicos rotacionados a 45° com progressão de escala.",
        )
    )

    # 5. Escada Lateral de Acesso ao Telhado
    add(
        CaixaAABB(
            pid,
            "Escada de Acesso ao Telhado",
            "Edificação Principal",
            [59.05, 0.20, 23.45],
            [59.65, 6.75, 24.55],
            self.materiais["esquadria"],
            "T(59.35, 3.47, 24.0) · S(0.6, 6.55, 1.1)",
            "Escada marinheiro lateral para acesso à cobertura do Bloco 734.",
        )
    )

    # 6. Árvores (Cilindros + Esferas rotacionadas via Quatérnios) + Ave Predadora
    arvores = [
        ("Árvore Bosque Esquerdo", 10.5, 28.5, 1.15),
        ("Árvore Fundo Esquerdo", 11.5, 38.0, 1.25),
        ("Árvore Praça Direita", 37.5, 17.5, 0.90),
    ]
    for nome_a, ax, az, esc in arvores:
      add(
          CilindroY(
              pid,
              f"Tronco — {nome_a}",
              "Vegetação & Fauna",
              ax,
              az,
              0.20,
              0.20 + 3.2 * esc,
              0.34 * esc,
              self.materiais["tronco"],
              f"T({ax}, 0.2, {az}) · S({esc})",
              "Tronco cilíndrico no Primeiro Octante.",
          )
      )
      q_rot = quat_para_matriz([0.3, 1.0, 0.2], 0.65 * esc)
      offsets = [
          (0.0, 3.6, 0.0, 1.65, "folha_escura"),
          (-1.05, 3.2, 0.55, 1.25, "folha_clara"),
          (1.05, 3.3, -0.50, 1.30, "folha_clara"),
          (0.25, 4.45, 0.35, 1.20, "folha_escura"),
      ]
      for idx_c, (ox, oy, oz, r_c, m_key) in enumerate(offsets, start=1):
        off_rot = (q_rot @ np.array([ox, 0.0, oz, 1.0]))[:3]
        add(
            Esfera(
                pid,
                f"Copa #{idx_c} — {nome_a}",
                "Vegetação & Fauna",
                [ax + off_rot[0] * esc, oy * esc, az + off_rot[2] * esc],
                r_c * esc,
                self.materiais[m_key],
                "T · Quatérnio([0.3,1,0.2]) · S(r)",
                "Volume esférico da copa posicionado via matriz de quatérnio.",
            )
        )

    # Ave Predadora empoleirada na árvore do fundo (X=11.5, Z=38.0)
    add(
        ConeY(
            pid,
            "Ave Predadora Territorial",
            "Vegetação & Fauna",
            11.5,
            38.0,
            5.10,
            1.05,
            0.34,
            self.materiais["ave"],
            "T(11.5, 5.1, 38.0) · Cone",
            "Ave empoleirada na copa da Árvore do Fundo.",
        )
    )

    # Palmeira Tropical Estilizada (X=42.5, Z=17.5)
    add(
        CilindroY(
            pid,
            "Tronco — Palmeira Tropical CT",
            "Vegetação & Fauna",
            42.5,
            17.5,
            0.20,
            4.65,
            0.22,
            self.materiais["tronco"],
            "T(42.5, 0.2, 17.5) · Cilindro h=4.45m",
            "Estipe cilíndrico da palmeira ornamental da praça.",
        )
    )
    add(
        Esfera(
            pid,
            "Copa — Palmeira Tropical CT",
            "Vegetação & Fauna",
            [42.5, 4.85, 17.5],
            1.15,
            self.materiais["folha_clara"],
            "T(42.5, 4.85, 17.5) · Quatérnio",
            "Coroa foliar da palmeira tropical.",
        )
    )

    # 7. Cones de Sinalização + Reflexo em Plano Arbitrário (Quest Bônus 6.2)
    cones_orig = [(35.3, 16.2), (35.9, 15.6), (36.5, 16.9), (37.1, 16.3)]
    M_esp = mat_espelho_plano([38.2, 0.2, 15.2], [1.0, 0.0, -1.0])
    for idx_cn, (cx, cz) in enumerate(cones_orig, start=1):
      add(
          ConeY(
              pid,
              f"Cone Sinalização #{idx_cn}",
              "Sinalização & Espelho",
              cx,
              cz,
              0.20,
              0.95,
              0.32,
              self.materiais["cone_laranja"],
              f"T({cx}, 0.2, {cz}) · Cone",
              "Cone viário original no Primeiro Octante.",
          )
      )
      p_esp = aplicar_matriz_pontos(M_esp, np.array([[cx, 0.20, cz]]))[0]
      add(
          ConeY(
              pid,
              f"Cone Espelhado #{idx_cn} (Plano Oblíquo)",
              "Sinalização & Espelho",
              float(p_esp[0]),
              float(p_esp[2]),
              0.20,
              0.95,
              0.32,
              self.materiais["cone_espelhado"],
              "H = I - 2nnᵀ · Espelho em Plano Arbitrário",
              "Cone duplicado via matriz de reflexão especular em plano oblíquo.",
          )
      )

    # 8. Abrigo Metálico Cisalhado (Quest Bônus 6.1) + 4 Pilares
    M_abrigo = (
        mat_translacao(46.8, 2.65, 18.8)
        @ mat_cisalhamento_xy(sh_xy=-0.12, sh_zy=0.04)
    )
    tris_abrigo = gerar_caixa_triangulos(4.5, 0.16, 3.2, M_abrigo)
    add(
        MalhaTriangulos(
            pid,
            "Abrigo Metálico (Malha Cisalhada XY)",
            "Mobiliário & Abrigo",
            tris_abrigo,
            self.materiais["metal_galv"],
            "T(46.8,2.65,18.8) · ShearXY(-0.12)",
            "Cobertura em malha triangular deformada por cisalhamento XY.",
        )
    )
    for idx_p, (px, pz) in enumerate(
        [(44.7, 17.4), (48.8, 17.4), (44.7, 20.2), (48.8, 20.2)], start=1
    ):
      add(
          CilindroY(
              pid,
              f"Pilar Abrigo #{idx_p}",
              "Mobiliário & Abrigo",
              px,
              pz,
              0.20,
              2.60,
              0.08,
              self.materiais["esquadria"],
              f"T({px}, 0.2, {pz}) · Cilindro",
              "Pilar tubular de sustentação do abrigo.",
          )
      )

    # 9. Máquina de Café Expresso CT (Reposicionada em ilha aberta na praça: X=35.8, Z=18.8)
    add(
        CaixaAABB(
            pid,
            "Balcão — Máquina de Café Expresso CT",
            "Cafeteria & Interação",
            [35.35, 0.20, 18.40],
            [36.25, 1.10, 19.20],
            self.materiais["tronco"],
            "T(35.8, 0.65, 18.8) · Ilha Aberta",
            "Balcão da cafeteira reposicionado em área aberta para câmera livre 360°.",
        )
    )
    add(
        CaixaAABB(
            pid,
            "Máquina de Café Expresso CT",
            "Cafeteria & Interação",
            [35.55, 1.10, 18.58],
            [36.05, 1.68, 19.02],
            self.materiais["cafeteira"],
            "T(35.8, 1.39, 18.8) · S(0.5, 0.58, 0.44)",
            "Cafeteira automática do CT que restaura energia e ativa cutscene.",
        )
    )

    # 10. 4 Postes de Iluminação Pública (Quest 7.1)
    postes = [
        (9.5, 21.0, 0.6),
        (18.5, 16.2, 0.4),
        (41.0, 16.5, 0.0),
        (54.5, 17.5, -0.5),
    ]
    for idx_pt, (px, pz, ox) in enumerate(postes, start=1):
      add(
          CilindroY(
              pid,
              f"Poste Iluminação #{idx_pt}",
              "Iluminação & Spots",
              px,
              pz,
              0.20,
              6.20,
              0.11,
              self.materiais["esquadria"],
              f"T({px}, 0.2, {pz}) · h=6.0m",
              "Poste metálico de iluminação viária/praça.",
          )
      )
      add(
          Esfera(
              pid,
              f"Luminária Pontual #{idx_pt}",
              "Iluminação & Spots",
              [px + ox, 6.32, pz + 0.35],
              0.28,
              self.materiais["spot_luz"],
              f"Luz Pontual #{idx_pt} (Atenuação Quadrática)",
              "Fonte luminosa pontual com atenuação física 1/(c0 + c1·d + c2·d²).",
          )
      )

    # 11. Veículos na Via CT (SUV Prata e Esportivo Vermelho)
    add(
        CaixaAABB(
            pid,
            "SUV Prata na Via CT",
            "Veículos & Tráfego",
            [20.8, 0.25, 6.2],
            [25.2, 1.55, 8.2],
            self.materiais["carro_prata"],
            "T(23.0, 0.9, 7.2) · S(4.4, 1.3, 2.0)",
            "Veículo utilitário transitando na via frontal do CT.",
        )
    )
    add(
        CaixaAABB(
            pid,
            "Carro Esportivo Vermelho",
            "Veículos & Tráfego",
            [36.8, 0.25, 9.6],
            [41.2, 1.42, 11.6],
            self.materiais["carro_vermelho"],
            "T(39.0, 0.84, 10.6) · S(4.4, 1.17, 2.0)",
            "Carro esportivo vermelho na pista do Centro de Tecnologia.",
        )
    )

    return prims

  def _definir_presets(self):
    """Define todos os presets de câmera, pontos de fuga e projeções (Atos IV e V)."""
    return {
        "principal": {
            "titulo": "Perspectiva Principal — Fachada Bloco 734 & Fonte",
            "eye": [14.5, 4.4, 6.5],
            "at": [33.5, 2.8, 28.0],
            "up": [0.0, 1.0, 0.0],
            "d": 1.35,
            "win": (-1.05, 1.05, -1.05, 1.05),
            "proj": "perspectiva",
        },
        "zoom_out": {
            "titulo": "Quest 10 — Perspectiva Zoom Out (Vista Geral CT)",
            "eye": [6.5, 11.5, 2.5],
            "at": [34.0, 2.5, 29.0],
            "up": [0.0, 1.0, 0.0],
            "d": 1.05,
            "win": (-1.20, 1.20, -1.20, 1.20),
            "proj": "perspectiva",
        },
        "zoom_in": {
            "titulo": "Quest 10 — Perspectiva Zoom In (Escultura & Cáusticas)",
            "eye": [11.2, 2.6, 15.5],
            "at": [18.5, 2.1, 22.8],
            "up": [0.0, 1.0, 0.0],
            "d": 1.85,
            "win": (-0.85, 0.85, -0.85, 0.85),
            "proj": "perspectiva",
        },
        "1_ponto_fuga": {
            "titulo": "Bônus — 1 Ponto de Fuga (Eixo Frontal Z)",
            "eye": [27.0, 2.4, 4.0],
            "at": [27.0, 2.4, 35.0],
            "up": [0.0, 1.0, 0.0],
            "d": 1.30,
            "win": (-1.10, 1.10, -1.10, 1.10),
            "proj": "perspectiva",
        },
        "2_pontos_fuga": {
            "titulo": "Bônus — 2 Pontos de Fuga (Horizontal Nivelado)",
            "eye": [12.0, 3.2, 8.0],
            "at": [35.0, 3.2, 28.0],
            "up": [0.0, 1.0, 0.0],
            "d": 1.35,
            "win": (-1.05, 1.05, -1.05, 1.05),
            "proj": "perspectiva",
        },
        "3_pontos_fuga": {
            "titulo": "Bônus — 3 Pontos de Fuga (Plongée Aéreo)",
            "eye": [10.0, 16.5, 5.0],
            "at": [34.0, 1.8, 28.0],
            "up": [0.0, 1.0, 0.0],
            "d": 1.30,
            "win": (-1.10, 1.10, -1.10, 1.10),
            "proj": "perspectiva",
        },
        "isometrica": {
            "titulo": "Bônus — Projeção Ortográfica / Isométrica",
            "eye": [8.0, 22.0, 4.0],
            "at": [34.0, 2.5, 28.0],
            "up": [0.0, 1.0, 0.0],
            "d": 1.0,
            "win": (-24.0, 24.0, -24.0, 24.0),
            "proj": "ortografica",
        },
        "obliqua": {
            "titulo": "Bônus — Projeção Oblíqua (Gabinete 45°)",
            "eye": [28.0, 5.5, 4.0],
            "at": [28.0, 5.5, 30.0],
            "up": [0.0, 1.0, 0.0],
            "d": 1.0,
            "win": (-22.0, 22.0, -18.0, 26.0),
            "proj": "obliqua",
        },
    }

  def _avaliar_textura(self, mat, P):
    """Calcula Kd procedural no ponto P (N, 3)."""
    N = P.shape[0]
    kd = np.broadcast_to(mat.kd, (N, 3)).copy()
    if mat.textura is None:
      return kd

    x, y, z = P[:, 0], P[:, 1], P[:, 2]

    if mat.textura == "asfalto":
      grao = (np.sin(x * 37.0) * np.cos(z * 41.0)) * 0.025
      kd = np.clip(kd + grao[:, None], 0.05, 0.95)
      faixa = (np.abs(z - 11.5) < 0.08) | (np.abs(z - 11.9) < 0.08)
      faixa &= (x >= 4.0) & (x <= 66.0)
      kd[faixa] = [0.92, 0.72, 0.08]

    elif mat.textura == "calcada":
      rejunte = (np.abs((x * 2.0) % 1.0 - 0.5) > 0.46) | (
          np.abs((z * 2.0) % 1.0 - 0.5) > 0.46
      )
      var_bloco = (
          np.sin(np.floor(x * 2.0) * 12.7 + np.floor(z * 2.0) * 31.1) * 0.04
      )
      kd = np.clip(kd + var_bloco[:, None], 0.1, 0.95)
      kd[rejunte] *= 0.72

    elif mat.textura == "agua":
      onda = np.sin(x * 3.8 + z * 2.4) * np.cos(z * 4.5 - x * 1.9)
      kd[:, 1] = np.clip(kd[:, 1] + onda * 0.08, 0.1, 0.95)
      kd[:, 2] = np.clip(kd[:, 2] + onda * 0.12, 0.2, 1.0)

    elif mat.textura == "placa734":
      u = np.clip((z - (42.0 - 22.08 * math.sin(0.14 * math.pi * 0.5) - 1.55)) / 3.10, 0.0, 0.999)
      v = np.clip(1.0 - (y - 5.20) / 1.05, 0.0, 0.999)
      iu = (u * self.placa_tex.shape[1]).astype(int)
      iv = (v * self.placa_tex.shape[0]).astype(int)
      kd = self.placa_tex[iv, iu]

    return kd

  def _cacar_intersecao(self, O, D):
    """Encontra a interseção mais próxima para um lote de raios (O, D)."""
    N = O.shape[0]
    best_t = np.full(N, np.inf, dtype=np.float64)
    best_N = np.zeros_like(D)
    best_pid = np.zeros(N, dtype=np.int32)

    for prim in self.primitivas:
      t, n = prim.intersectar(O, D)
      closer = t < best_t
      if np.any(closer):
        best_t[closer] = t[closer]
        best_N[closer] = n[closer]
        best_pid[closer] = prim.pid

    return best_t, best_N, best_pid

  def _em_sombra(self, P, L_dir, max_dist):
    """Quest 11: Shadow rays para verificar oclusão entre P e a fonte de luz."""
    N = P.shape[0]
    em_sombra = np.zeros(N, dtype=bool)
    O_s = P + L_dir * 2e-3
    D_s = np.broadcast_to(L_dir, (N, 3)) if L_dir.ndim == 1 else L_dir

    for prim in self.primitivas:
      # Ignora objetos luminosos emissores e lâmina d'água transparente
      if prim.material is self.materiais["spot_luz"] or prim.material is self.materiais["agua"]:
        continue
      t, _ = prim.intersectar(O_s, D_s)
      em_sombra |= (t > 1e-4) & (t < max_dist - 1e-2)
      if np.all(em_sombra):
        break
    return em_sombra

  def _sombrear_raios(
      self,
      O,
      D,
      modo_noite=False,
      ativar_caustics=True,
      ativar_reflexos=True,
      ativar_sombras=True,
      ativar_sol=True,
      ativar_spots=True,
      profundidade=0,
  ):
    """Calcula radiância RGB (Phong + Sombras + Spots + Sol + Cáusticas + Reflexão)."""
    N = O.shape[0]
    best_t, best_N, best_pid = self._cacar_intersecao(O, D)

    # Gradiente de céu (Dia vs Noite / Sol desligado)
    t_sky = np.clip(0.5 * (D[:, 1] + 1.0), 0.0, 1.0)[:, None]
    if modo_noite or not ativar_sol:
      sky_bot = np.array([[0.04, 0.07, 0.16]])
      sky_top = np.array([[0.01, 0.02, 0.06]])
    else:
      sky_bot = np.array([[0.78, 0.89, 0.99]])
      sky_top = np.array([[0.28, 0.56, 0.92]])
    cor = sky_bot * (1.0 - t_sky) + sky_top * t_sky

    hit = best_pid > 0
    if not np.any(hit):
      return cor, best_t, best_N, best_pid

    idx_hit = np.where(hit)[0]
    Ph = O[idx_hit] + best_t[idx_hit, None] * D[idx_hit]
    Nh = best_N[idx_hit]
    Vh = -D[idx_hit]
    pids_h = best_pid[idx_hit]

    # Garante normal voltada para o raio incidente
    flip = np.sum(Nh * Vh, axis=1) < 0.0
    Nh = np.where(flip[:, None], -Nh, Nh)

    cor_h = np.zeros((idx_hit.shape[0], 3), dtype=np.float64)

    # Luz Ambiente Global (Sub-quest 7.2)
    Ia = (
        np.array([0.16, 0.20, 0.32])
        if (modo_noite or not ativar_sol)
        else np.array([0.42, 0.45, 0.50])
    )

    # Parâmetros do Sol Direcional (Sub-quest Bônus 7.4)
    sun_dir = np.array([-0.42, 0.82, -0.38])
    sun_dir /= np.linalg.norm(sun_dir)
    I_sun = (
        np.array([0.18, 0.24, 0.42])
        if modo_noite
        else np.array([1.05, 1.00, 0.90])
    )

    # 4 Postes Pontuais (Sub-quest 7.1)
    postes_pos = [
        np.array([10.1, 6.32, 21.35]),
        np.array([18.9, 6.32, 16.55]),
        np.array([41.0, 6.32, 16.85]),
        np.array([54.0, 6.32, 17.85]),
    ]
    I_poste = (
        np.array([1.35, 1.18, 0.78])
        if (modo_noite or not ativar_sol)
        else np.array([0.22, 0.20, 0.15])
    )

    # Holofote Spot na Escultura Cinética + 3 Spots Submersos (Sub-quest Bônus 7.3)
    spot_esc_pos = np.array([12.5, 4.8, 18.5])
    spot_esc_dir = np.array([15.0, 1.8, 22.0]) - spot_esc_pos
    spot_esc_dir /= np.linalg.norm(spot_esc_dir)

    for prim in self.primitivas:
      m_p = pids_h == prim.pid
      if not np.any(m_p):
        continue

      Pp = Ph[m_p]
      Np = Nh[m_p]
      Vp = Vh[m_p]
      mat = prim.material
      kd = self._avaliar_textura(mat, Pp)

      # Ambiente + Emissivo
      acc = Ia * (kd * 0.55) + mat.emissivo

      # 1. Sol Direcional (ligável/desligável!)
      if ativar_sol:
        ndotl = np.maximum(np.sum(Np * sun_dir, axis=1), 0.0)
        if np.any(ndotl > 0.0):
          sombra_sol = (
              self._em_sombra(Pp, sun_dir, 120.0)
              if ativar_sombras
              else np.zeros(Pp.shape[0], dtype=bool)
          )
          vis = (~sombra_sol).astype(np.float64)
          H_sun = sun_dir + Vp
          H_sun /= np.linalg.norm(H_sun, axis=1, keepdims=True) + 1e-9
          spec = np.power(np.maximum(np.sum(Np * H_sun, axis=1), 0.0), mat.shininess)
          acc += vis[:, None] * I_sun * (
              kd * ndotl[:, None] + mat.ke * spec[:, None] * 0.45
          )

      # 2. Luzes Pontuais dos 4 Postes
      for p_luz in postes_pos:
        L_vec = p_luz - Pp
        dist = np.linalg.norm(L_vec, axis=1)
        L_dir = L_vec / (dist[:, None] + 1e-9)
        ndotl_p = np.maximum(np.sum(Np * L_dir, axis=1), 0.0)
        aten = 1.0 / (1.0 + 0.045 * dist + 0.008 * dist * dist)
        H_p = L_dir + Vp
        H_p /= np.linalg.norm(H_p, axis=1, keepdims=True) + 1e-9
        spec_p = np.power(np.maximum(np.sum(Np * H_p, axis=1), 0.0), mat.shininess)
        acc += (aten * ndotl_p)[:, None] * I_poste * kd + (aten * spec_p)[:, None] * I_poste * mat.ke * 0.3

      # 3. Holofotes / Spots (ligáveis/desligáveis!)
      if ativar_spots:
        # Spot direcional focado na Escultura Cinética
        Ls_vec = spot_esc_pos - Pp
        ds = np.linalg.norm(Ls_vec, axis=1)
        Ls_dir = Ls_vec / (ds[:, None] + 1e-9)
        cos_ang = np.sum(-Ls_dir * spot_esc_dir, axis=1)
        in_cone = cos_ang > math.cos(math.radians(26.0))
        if np.any(in_cone):
          fator_spot = np.where(in_cone, np.power(np.maximum(cos_ang, 0.0), 4.0), 0.0)
          ndotl_s = np.maximum(np.sum(Np * Ls_dir, axis=1), 0.0)
          acc += (fator_spot * ndotl_s * 1.45)[:, None] * np.array([0.85, 0.95, 1.0]) * kd

        # Cáusticas refrativas projetadas pelos 3 spots submersos da fonte
        if ativar_caustics and prim.recebe_caustica:
          px, py, pz = Pp[:, 0], Pp[:, 1], Pp[:, 2]
          zona_c = (px >= 16.0) & (px <= 36.0) & (py >= 0.2) & (py <= 4.5) & (pz >= 16.5) & (pz <= 28.0)
          if np.any(zona_c):
            w1 = np.sin(px * 3.4 + py * 4.8 + pz * 2.1)
            w2 = np.cos(px * 4.1 - py * 3.7 + pz * 3.3)
            filamento = np.power(np.maximum(w1 * w2, 0.0), 3.0)
            aten_y = np.exp(-np.maximum(py - 0.5, 0.0) * 0.45)
            c_int = np.where(zona_c, filamento * aten_y * 0.85, 0.0)
            acc += c_int[:, None] * np.array([0.35, 0.88, 1.0])

      # 4. Reflexão Especular Recursiva (1 bounce)
      if ativar_reflexos and profundidade == 0 and mat.refletividade > 0.05:
        R_dir = D[idx_hit][m_p] - 2.0 * np.sum(D[idx_hit][m_p] * Np, axis=1, keepdims=True) * Np
        R_dir /= np.linalg.norm(R_dir, axis=1, keepdims=True) + 1e-9
        cor_refl, _, _, _ = self._sombrear_raios(
            Pp + Np * 3e-3,
            R_dir,
            modo_noite=modo_noite,
            ativar_caustics=False,
            ativar_reflexos=False,
            ativar_sombras=False,
            ativar_sol=ativar_sol,
            ativar_spots=ativar_spots,
            profundidade=1,
        )
        acc = acc * (1.0 - mat.refletividade) + cor_refl * mat.refletividade

      cor_h[m_p] = acc

    cor[idx_hit] = np.clip(cor_h, 0.0, 1.0)
    return cor, best_t, best_N, best_pid

  def renderizar_config(
      self,
      cfg,
      modo_noite=False,
      ativar_caustics=True,
      ativar_reflexos=True,
      ativar_sombras=True,
      ativar_sol=True,
      ativar_spots=True,
      ssaa=False,
  ):
    """Renderiza a imagem completa (W x H) e popula o Pick Buffer O(1)."""
    W, H = self.W, self.H
    Mcw = construir_Mcw(cfg["eye"], cfg["at"], cfg["up"])
    R_cw = Mcw[:3, :3]
    eye = np.asarray(cfg["eye"], dtype=np.float64)

    xmin, xmax, ymin, ymax = cfg["win"]
    d = float(cfg["d"])
    proj = cfg.get("proj", "perspectiva")

    # Coordenadas dos centros dos pixels na janela de projeção
    us = xmin + (np.arange(W) + 0.5) * ((xmax - xmin) / W)
    vs = ymax - (np.arange(H) + 0.5) * ((ymax - ymin) / H)
    uu, vv = np.meshgrid(us, vs)
    uu_flat = uu.ravel()
    vv_flat = vv.ravel()
    N_pix = W * H

    if proj == "ortografica":
      O_cam = np.stack([uu_flat, vv_flat, np.zeros(N_pix)], axis=1)
      D_cam = np.tile(np.array([[0.0, 0.0, -1.0]]), (N_pix, 1))
    elif proj == "obliqua":
      # Projeção Oblíqua Gabinete (L = 0.5, alpha = 45°)
      O_cam = np.stack([uu_flat, vv_flat, np.zeros(N_pix)], axis=1)
      ang_ob = math.radians(45.0)
      L_gab = 0.45
      D_cam = np.tile(
          np.array([[L_gab * math.cos(ang_ob), L_gab * math.sin(ang_ob), -1.0]]),
          (N_pix, 1),
      )
      D_cam /= np.linalg.norm(D_cam, axis=1, keepdims=True)
    else:
      O_cam = np.zeros((N_pix, 3), dtype=np.float64)
      D_cam = np.stack([uu_flat, vv_flat, np.full(N_pix, -d)], axis=1)
      D_cam /= np.linalg.norm(D_cam, axis=1, keepdims=True)

    O_world = (R_cw @ O_cam.T).T + eye
    D_world = (R_cw @ D_cam.T).T
    D_world /= np.linalg.norm(D_world, axis=1, keepdims=True)

    # Processa em fatias de linhas para manter baixo consumo de memória
    img_flat = np.zeros((N_pix, 3), dtype=np.float64)
    t_flat = np.full(N_pix, np.inf, dtype=np.float64)
    N_flat = np.zeros((N_pix, 3), dtype=np.float64)
    pid_flat = np.zeros(N_pix, dtype=np.int32)

    chunk = W * 25
    for start in range(0, N_pix, chunk):
      end = min(start + chunk, N_pix)
      c_blk, t_blk, n_blk, p_blk = self._sombrear_raios(
          O_world[start:end],
          D_world[start:end],
          modo_noite=modo_noite,
          ativar_caustics=ativar_caustics,
          ativar_reflexos=ativar_reflexos,
          ativar_sombras=ativar_sombras,
          ativar_sol=ativar_sol,
          ativar_spots=ativar_spots,
      )
      img_flat[start:end] = c_blk
      t_flat[start:end] = t_blk
      N_flat[start:end] = n_blk
      pid_flat[start:end] = p_blk

    # SSAA leve opcional nas bordas de contraste
    if ssaa:
      dx_w = (xmax - xmin) / (W * 2.0)
      dy_w = (ymax - ymin) / (H * 2.0)
      O_j = O_world + (R_cw @ np.array([dx_w, dy_w, 0.0]))
      D_j = D_world
      for start in range(0, N_pix, chunk):
        end = min(start + chunk, N_pix)
        c_j, _, _, _ = self._sombrear_raios(
            O_j[start:end],
            D_j[start:end],
            modo_noite=modo_noite,
            ativar_caustics=ativar_caustics,
            ativar_reflexos=ativar_reflexos,
            ativar_sombras=ativar_sombras,
            ativar_sol=ativar_sol,
            ativar_spots=ativar_spots,
        )
        img_flat[start:end] = 0.5 * (img_flat[start:end] + c_j)

    # Salva Pick Buffer O(1)
    P_flat = O_world + np.where(np.isfinite(t_flat), t_flat, 0.0)[:, None] * D_world
    self.pick_buffer = {
        "pid": pid_flat.reshape(H, W),
        "t": t_flat.reshape(H, W),
        "P": P_flat.reshape(H, W, 3),
        "N": N_flat.reshape(H, W, 3),
    }

    img_arr = img_flat.reshape(H, W, 3)
    status_luz = f"{'Noite' if modo_noite else 'Dia'} | Sol: {'ON' if ativar_sol else 'OFF'} | Spots: {'ON' if ativar_spots else 'OFF'}"
    legenda = f"{cfg['titulo']} ({W}×{H}px • {status_luz})"
    return img_arr, legenda

  def consultar_pick_buffer(self, px, py):
    """Quest 12: Consulta instantânea O(1) do objeto atingido no pixel (px, py)."""
    if self.pick_buffer is None:
      return {"hit": False, "nome": "Nenhuma imagem renderizada", "descricao": ""}

    px = int(np.clip(px, 0, self.W - 1))
    py = int(np.clip(py, 0, self.H - 1))
    pid = int(self.pick_buffer["pid"][py, px])

    if pid <= 0:
      return {
          "hit": False,
          "nome": "Céu / Horizonte Atmosférico",
          "descricao": "Raio primário não interceptou sólidos (t = ∞).",
      }

    prim = next((p for p in self.primitivas if p.pid == pid), None)
    if prim is None:
      return {"hit": False, "nome": "Objeto Desconhecido", "descricao": ""}

    dist = float(self.pick_buffer["t"][py, px])
    P = self.pick_buffer["P"][py, px]
    N = self.pick_buffer["N"][py, px]
    kd = prim.material.kd

    return {
        "hit": True,
        "pid": pid,
        "nome": prim.nome,
        "tipo_primitiva": prim.tipo,
        "grupo": prim.grupo,
        "material": prim.material.nome,
        "distancia": dist,
        "ponto_world": f"({P[0]:.2f}, {P[1]:.2f}, {P[2]:.2f})",
        "normal_world": f"({N[0]:.2f}, {N[1]:.2f}, {N[2]:.2f})",
        "kd": f"({kd[0]:.2f}, {kd[1]:.2f}, {kd[2]:.2f})",
        "refletividade": prim.material.refletividade,
        "transformacao": prim.transformacao_txt,
        "descricao": prim.descricao,
    }

  def obter_status_quests(
      self,
      cfg_atual,
      resolucao,
      ativar_sombras=True,
      ativar_sol=True,
      ativar_spots=True,
  ):
    """Auditoria matemática automática das Quests (Atos I a VIII)."""
    proj_atual = cfg_atual.get("proj", "perspectiva")
    return {
        "ato1": {
            "titulo": "ATO I — A Construção do Mundo",
            "items": [
                (
                    "Sub-quest 1.1 — Tudo Tem Um Propósito (Cenário Bloco 734 CT)",
                    True,
                    "Obrigatória",
                ),
                (
                    "Sub-quest 1.2 — O Primeiro Octante (X, Y, Z ≥ 0)",
                    True,
                    "Obrigatória",
                ),
                (
                    "Quest 2 — Zoo dos Objetos (Esfera, Cilindro, Cone, Caixa, Arco, Malha)",
                    True,
                    "Obrigatória",
                ),
                (
                    f"Sub-quest 3.1 — Diversidade ({len(self.materiais)} materiais ≥ 4)",
                    True,
                    "Obrigatória",
                ),
                (
                    "Sub-quest 3.2 — A Pele Pintada (Asfalto, Calçada, Água, Placa 734)",
                    True,
                    "Obrigatória",
                ),
            ],
        },
        "ato2": {
            "titulo": "ATO II — As Transformações Mágicas",
            "items": [
                ("Quest 4 — O Ritual da Translação (4×4)", True, "Obrigatória"),
                (
                    "Sub-quest 5.1 — O Eixo Sagrado (Rotação X, Y, Z)",
                    True,
                    "Obrigatória",
                ),
                (
                    "Sub-quest 5.2 — O Eixo Livre (Quatérnios nas copas)",
                    True,
                    "Obrigatória",
                ),
                ("Quest 6 — O Encantamento da Escala", True, "Obrigatória"),
                (
                    "Quest Bônus 6.1 — A Distorção (Cisalhamento Abrigo)",
                    True,
                    "+0.5 pts",
                ),
                (
                    "Quest Bônus 6.2 — O Reflexo Perfeito (Espelho Cones)",
                    True,
                    "+0.5 pts",
                ),
            ],
        },
        "ato3": {
            "titulo": "ATO III — A Luz e a Escuridão",
            "items": [
                (
                    "Sub-quest 7.1 — A Chama Pontual (4 Postes com atenuação)",
                    True,
                    "Obrigatória",
                ),
                (
                    "Sub-quest 7.2 — O Brilho Ambiente (Ia)",
                    True,
                    "Obrigatória",
                ),
                (
                    f"Sub-quest Bônus 7.3 — O Holofote (Spots: {'ON' if ativar_spots else 'OFF'})",
                    ativar_spots,
                    "+1.0 pts",
                ),
                (
                    f"Sub-quest Bônus 7.4 — O Sol Direcional (Sol: {'ON' if ativar_sol else 'OFF'})",
                    ativar_sol,
                    "+0.5 pts",
                ),
            ],
        },
        "ato45": {
            "titulo": "ATOS IV & V — Câmera e Projeções",
            "items": [
                (
                    "Quests 8 e 9 — Eye, At, Up, Distância Focal (d) e Janela",
                    True,
                    "Obrigatória",
                ),
                (
                    "Quest 10 — Perspectiva (Presets Zoom In / Zoom Out)",
                    True,
                    "Obrigatória",
                ),
                (
                    "Bônus Pontos de Fuga — 1, 2 e 3 Pontos de Fuga",
                    True,
                    "+1.5 pts",
                ),
                (
                    f"Bônus Projeções Alternativas — Isométrica / Oblíqua (Atual: {proj_atual})",
                    True,
                    "+1.0 pts",
                ),
            ],
        },
        "ato678": {
            "titulo": "ATOS VI, VII & VIII — Sombras, Picking e Mosaico 500×500",
            "items": [
                (
                    "Quest 11 — A Sombra (Sombras Projetadas)",
                    ativar_sombras,
                    "Obrigatória",
                ),
                (
                    "Quest 12 — O Toque (Picking Interativo 3D & Render)",
                    True,
                    "Obrigatória",
                ),
                (
                    "Bônus — Interface Gráfica Interativa (Streamlit + WebGL)",
                    True,
                    "+1.0 pts",
                ),
                (
                    f"Quest 13 — O Mosaico de 500×500 (Atual: {resolucao}×{resolucao})",
                    resolucao >= 500,
                    "Obrigatória",
                ),
                (
                    "Quest Lendária — Criatividade e Beleza (Cáusticas + Mini-Game)",
                    True,
                    "+1.0 pts",
                ),
            ],
        },
    }
