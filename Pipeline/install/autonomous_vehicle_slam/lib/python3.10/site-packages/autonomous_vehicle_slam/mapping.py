import numpy as np

from .slam import carToWorld, worldToCar

# Filtro das deteccoes usadas no mapa: cones longe demais tem erro de
# profundidade/lateral grande (lidar esparso, intrinsecos da camera) e
# poluem o mapa com fantasmas. O carro passa perto de todos os cones ao
# longo da volta, entao alcance curto nao perde cobertura
MAX_MAP_RANGE = 12.0         # m

# As deteccoes [lateral, frente] saem no frame do SENSOR (Lidar1 esta
# 0.45 m a frente do centro do carro, ver settings.json do FSDS). Sem
# compensar, cada cone mapeado para o mundo desloca 0.45 m na direcao do
# heading -- o mesmo cone visto com headings diferentes (antes/depois de
# uma curva) vira dois cones no mapa
SENSOR_FWD_OFFSET = 0.45     # m

# Ruido de medicao (cresce com a distancia), igual ao do SLAM
MEAS_STD_BASE = 0.15         # m
MEAS_STD_PER_M = 0.03        # m por metro

# Associacao
GATE_MIN = 1.2               # m
GATE_MAX = 2.4               # m
DRIFT_STD_RATE = 0.02        # m/s de deriva esperada desde a ultima vista
DRIFT_AGE_CAP = 60.0         # s

# Gestao de landmarks
CONFIRM_HITS = 3             # deteccoes para confirmar um cone
TENTATIVE_TIMEOUT = 3.0      # s sem rever um tentativo -> descarta
MERGE_RADIUS = 1.5           # m: cones da mesma cor mais perto que isso fundem
MERGE_PERIOD = 2.0           # s entre passadas incrementais de fusao

# Janela de cones locais (formato da percepcao)
LOCAL_AHEAD = 22.0
LOCAL_BEHIND = 2.0
LOCAL_LATERAL = 10.0

# Linha de largada/chegada: cones laranja (2 = grande, 3 = pequeno) que
# o FSDS coloca na largada. Servem de marco fisico para fechar a volta
ORANGE_IDS = (2, 3)
START_ANCHOR_RADIUS = 20.0   # m da origem do mapa onde procurar a linha
START_ANCHOR_MIN_CONES = 2


class ConeMapper:
    """Mapa global de cones robusto, mantido a partir da pose do SLAM.

    O FastSLAM ja carrega um mapa por particula, mas ele troca de "melhor
    particula" a cada reamostragem -- o mapa consultado pelo planejador
    global pula de hipotese em hipotese e o laco nunca estabiliza o
    suficiente para fechar. Este mapeador e um unico mapa persistente,
    alimentado pela pose estimada do SLAM, com tres defesas que o mapa
    por particula nao tem:

      1. Filtro de alcance: so integra deteccoes ate MAX_MAP_RANGE
         (longe, o erro de profundidade do lidar cria cones fantasmas).
      2. Associacao 1-para-1 por frame (greedy do mais perto para o mais
         longe): duas deteccoes nao caem no mesmo landmark e nao nasce
         duplicado colado num cone ja casado.
      3. Fusao incremental de duplicados a cada MERGE_PERIOD, em vez de
         uma unica fusao no fim da volta -- o mapa chega ao fechamento
         do laco ja limpo.

    Cada landmark e um filtro de informacao isotropico (media ponderada
    pela variancia), com contagem de deteccoes para confirmar e poda de
    tentativos, como no SLAM.
    """

    def __init__(self, sensor_fwd_offset=SENSOR_FWD_OFFSET):
        self.sensor_fwd_offset = sensor_fwd_offset
        self.pos = np.empty((0, 2))
        self.var = np.empty(0)
        self.cls = np.empty(0, dtype=int)
        self.hits = np.empty(0, dtype=int)
        self.seen = np.empty(0)
        self.last_merge = 0.0

    def update(self, pose, detections, now):
        """Integra as deteccoes do frame ([lateral, frente, classId]) no mapa."""
        detections = np.asarray(detections)
        if detections.ndim != 2 or len(detections) == 0:
            self._prune(now)
            return

        ranges = np.linalg.norm(detections[:, :2].astype(float), axis=1)
        near = ranges <= MAX_MAP_RANGE
        if not near.any():
            self._prune(now)
            return
        detections = detections[near].astype(float)
        ranges = ranges[near]

        # frame do sensor -> frame do carro (so para o mapa: o planner
        # local continua recebendo as deteccoes cruas, como sempre)
        detections[:, 1] += self.sensor_fwd_offset

        world = carToWorld(pose, detections)

        # processa do mais perto para o mais longe (medicao mais precisa
        # ganha a disputa) e bloqueia cada landmark a uma deteccao por frame
        order = np.argsort(ranges)
        used = set()
        for i in order:
            z = world[i]
            cls = int(detections[i, 2])
            meas_var = (MEAS_STD_BASE + MEAS_STD_PER_M * ranges[i]) ** 2

            idx = np.flatnonzero(self.cls == cls)
            idx = idx[~np.isin(idx, list(used))] if used else idx

            best = -1
            if len(idx):
                d = np.linalg.norm(self.pos[idx] - z, axis=1)
                j = int(np.argmin(d))
                age = min(now - self.seen[idx[j]], DRIFT_AGE_CAP)
                S = self.var[idx[j]] + meas_var + (DRIFT_STD_RATE * age) ** 2
                gate = np.clip(3.0 * np.sqrt(S), GATE_MIN, GATE_MAX)
                if d[j] < gate:
                    best = int(idx[j])

            if best < 0:
                self.pos = np.vstack([self.pos, z])
                self.var = np.append(self.var, meas_var)
                self.cls = np.append(self.cls, cls)
                self.hits = np.append(self.hits, 1)
                self.seen = np.append(self.seen, now)
                used.add(len(self.var) - 1)
                continue

            gain = self.var[best] / (self.var[best] + meas_var)
            self.pos[best] += gain * (z - self.pos[best])
            self.var[best] *= (1.0 - gain)
            self.hits[best] += 1
            self.seen[best] = now
            used.add(best)

        self._prune(now)
        if now - self.last_merge > MERGE_PERIOD:
            self.last_merge = now
            self.mergeDuplicates()

    def _prune(self, now):
        keep = (self.hits >= CONFIRM_HITS) | (now - self.seen < TENTATIVE_TIMEOUT)
        if not keep.all():
            self._apply(keep)

    def _apply(self, keep):
        self.pos = self.pos[keep]
        self.var = self.var[keep]
        self.cls = self.cls[keep]
        self.hits = self.hits[keep]
        self.seen = self.seen[keep]

    def mergeDuplicates(self, radius=MERGE_RADIUS):
        """Funde cones da mesma cor mais proximos que `radius` (media
        ponderada pela informacao 1/var; somam-se hits)."""
        if len(self.var) < 2:
            return
        order = np.argsort(-self.hits)
        keep, pos, w, hits = [], [], [], []
        for i in order:
            placed = False
            for k in range(len(keep)):
                if (self.cls[i] == self.cls[keep[k]]
                        and np.linalg.norm(pos[k] - self.pos[i]) < radius):
                    wi = 1.0 / max(self.var[i], 1e-6)
                    pos[k] = (pos[k] * w[k] + self.pos[i] * wi) / (w[k] + wi)
                    w[k] += wi
                    hits[k] += self.hits[i]
                    placed = True
                    break
            if not placed:
                keep.append(i)
                pos.append(self.pos[i].copy())
                w.append(1.0 / max(self.var[i], 1e-6))
                hits.append(int(self.hits[i]))

        keep = np.array(keep, dtype=int)
        self.pos = np.array(pos)
        self.var = 1.0 / np.array(w)
        self.cls = self.cls[keep]
        self.hits = np.array(hits, dtype=int)
        self.seen = self.seen[keep]

    # ---------- consultas ----------

    def startLineAnchor(self):
        """Centroide dos cones laranja confirmados perto da origem do mapa
        (a linha de largada/chegada), ou None se ainda nao ha cones
        suficientes. Usado para ancorar o fechamento da volta num marco
        fisico em vez da pose inicial do carro."""
        mask = (np.isin(self.cls, ORANGE_IDS)
                & (self.hits >= CONFIRM_HITS)
                & (np.linalg.norm(self.pos, axis=1) < START_ANCHOR_RADIUS))
        if mask.sum() < START_ANCHOR_MIN_CONES:
            return None
        return self.pos[mask].mean(axis=0)

    def confirmed(self):
        """Cones confirmados: array [x, y, classId] no frame do SLAM."""
        mask = self.hits >= CONFIRM_HITS
        if not mask.any():
            return np.empty((0, 3))
        return np.column_stack([self.pos[mask], self.cls[mask]])

    def localCones(self, pose, ahead=LOCAL_AHEAD, behind=LOCAL_BEHIND,
                   lateral=LOCAL_LATERAL):
        """Cones confirmados perto do carro em [lateral, frente, classId]."""
        cones = self.confirmed()
        if len(cones) == 0:
            return np.empty((0, 3))
        local = worldToCar(pose, cones[:, :2])
        mask = ((local[:, 1] > -behind) & (local[:, 1] < ahead)
                & (np.abs(local[:, 0]) < lateral))
        return np.column_stack([local[mask], cones[mask, 2]])
