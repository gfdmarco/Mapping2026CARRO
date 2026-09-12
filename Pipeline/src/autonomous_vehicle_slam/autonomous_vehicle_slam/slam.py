import numpy as np

# ---------------- FastSLAM ----------------
N_PARTICLES = 60

# Ruido do modelo de movimento (amostrado por particula na predicao).
# Deve ser compativel com a qualidade real da odometria: exagerar faz as
# particulas derivarem sozinhas e o mapa duplicar no fechamento da volta
STD_SPEED_FRAC = 0.03        # fracao da velocidade medida
STD_SPEED_MIN = 0.03         # m/s
STD_YAW_RATE = np.deg2rad(1.0)   # rad/s

# Temperagem da verossimilhanca: amortece o colapso dos pesos para
# manter diversidade de hipoteses ate o fechamento da volta
LIK_TEMPER = 0.5

# Refinamento de pose por particula (estilo FastSLAM 2.0): a cada frame a
# pose e puxada pelo alinhamento rigido entre as deteccoes casadas e o
# proprio mapa, impedindo a deriva de acumular em relacao ao mapa
POSE_CORR_GAIN = 0.5         # fracao da correcao aplicada por frame
POSE_CORR_MAX_ROT = np.deg2rad(1.0)  # rad por frame
POSE_CORR_MAX_TRANS = 0.2    # m por frame

# Ruido de medicao dos cones (cresce com a distancia)
MEAS_STD_BASE = 0.15         # m
MEAS_STD_PER_M = 0.03        # m por metro

# Associacao / gestao de landmarks
GATE_MIN = 1.2               # m  -> gate minimo de associacao
GATE_MAX = 2.5               # m  -> gate maximo (cones vizinhos ~3 m+)
DRIFT_STD_RATE = 0.02        # m/s -> deriva de pose esperada por segundo sem
                             #        rever o cone (fechamento de volta)
DRIFT_AGE_CAP = 60.0         # s  -> teto da idade considerada
NEW_LM_LOGLIK = -4.0         #    -> log-verossimilhanca de criar cone novo
CONFIRM_HITS = 3             #    -> deteccoes para confirmar um cone
TENTATIVE_TIMEOUT = 3.0      # s  -> tentativo sem deteccao e descartado
MERGE_RADIUS = 1.5           # m  -> fusao de duplicados no fim da volta
                             #       (cones da mesma cor distam 3 m+)

RESAMPLE_NEFF_FRAC = 0.5     # reamostra quando n_eff < frac * N

# Janela de cones locais (formato da percepcao, com memoria do mapa)
LOCAL_AHEAD = 25.0
LOCAL_BEHIND = 6.0
LOCAL_LATERAL = 12.0

# Deteccao de volta completa
MIN_LAP_DISTANCE = 50.0      # m rodados antes de aceitar fechamento
CLOSE_RADIUS = 4.0           # m da largada para fechar a volta


def quaternionToYaw(q):
    """Yaw (rotacao em torno de z) do quaternion do FSDS (NED, z para baixo)."""
    return np.arctan2(2.0 * (q.w_val * q.z_val + q.x_val * q.y_val),
                      1.0 - 2.0 * (q.y_val ** 2 + q.z_val ** 2))


def carToWorld(pose, cones_car):
    """[lateral, frente] no referencial do carro -> (x, y) no mundo.

    Body frame do FSDS: x = frente, y = direita (NED). A percepcao entrega
    [lateral(+direita), frente], entao body = (frente, lateral).
    """
    x, y, yaw = pose
    c, s = np.cos(yaw), np.sin(yaw)
    cones_car = np.atleast_2d(cones_car)
    fwd, lat = cones_car[:, 1], cones_car[:, 0]
    return np.column_stack([x + fwd * c - lat * s,
                            y + fwd * s + lat * c])


def worldToCar(pose, points_world):
    """(x, y) no mundo -> [lateral, frente] no referencial do carro."""
    x, y, yaw = pose
    c, s = np.cos(yaw), np.sin(yaw)
    points_world = np.atleast_2d(points_world)
    dx, dy = points_world[:, 0] - x, points_world[:, 1] - y
    fwd = dx * c + dy * s
    lat = -dx * s + dy * c
    return np.column_stack([lat, fwd])


class _Particle:
    """Uma hipotese de pose com o seu proprio mapa de landmarks.

    Mapa em arrays: posicao (n,2), variancia isotropica (n,), classe,
    contagem de deteccoes e instante da ultima deteccao.
    """

    __slots__ = ("pose", "lm_pos", "lm_var", "lm_cls", "lm_hits", "lm_seen")

    def __init__(self):
        self.pose = np.zeros(3)
        self.lm_pos = np.empty((0, 2))
        self.lm_var = np.empty(0)
        self.lm_cls = np.empty(0, dtype=int)
        self.lm_hits = np.empty(0, dtype=int)
        self.lm_seen = np.empty(0)

    def copy(self):
        p = _Particle.__new__(_Particle)
        p.pose = self.pose.copy()
        p.lm_pos = self.lm_pos.copy()
        p.lm_var = self.lm_var.copy()
        p.lm_cls = self.lm_cls.copy()
        p.lm_hits = self.lm_hits.copy()
        p.lm_seen = self.lm_seen.copy()
        return p


class FastSLAM:
    """FastSLAM 1.0: filtro de particulas Rao-Blackwellizado.

    Cada particula amostra uma trajetoria possivel do carro (predicao por
    odometria: velocidade + yaw rate com ruido) e mantem um mapa proprio
    onde cada cone e um EKF independente. O peso da particula e a
    verossimilhanca das deteccoes no seu mapa; reamostragem sistematica
    mata as hipoteses ruins. A pose NAO usa a posicao do simulador --
    apenas odometria + os proprios cones, como num carro real.

    O frame do mapa nasce na pose inicial do carro (0, 0, yaw 0).
    """

    def __init__(self, n_particles=N_PARTICLES, seed=None):
        self.rng = np.random.default_rng(seed)
        self.particles = [_Particle() for _ in range(n_particles)]
        self.weights = np.full(n_particles, 1.0 / n_particles)
        self.pose = np.zeros(3)

    def update(self, speed, yaw_rate, detections, dt, now):
        """Um ciclo predicao + correcao; retorna a pose estimada (x, y, yaw)."""
        self._predict(speed, yaw_rate, dt)

        detections = np.asarray(detections)
        if detections.ndim == 2 and len(detections):
            log_lik = np.array([self._updateParticle(p, detections, now)
                                for p in self.particles])
            log_lik *= LIK_TEMPER
            lik = np.exp(log_lik - log_lik.max())
            self.weights *= lik
            self.weights /= self.weights.sum()

            n_eff = 1.0 / np.sum(self.weights ** 2)
            if n_eff < RESAMPLE_NEFF_FRAC * len(self.particles):
                self._resample()

        self._prune(now)
        self.pose = self._estimatePose()
        return self.pose

    # ---------- passos internos ----------

    def _predict(self, speed, yaw_rate, dt):
        if dt <= 0.0:
            return
        n = len(self.particles)
        std_v = max(STD_SPEED_MIN, STD_SPEED_FRAC * abs(speed))
        v = speed + self.rng.normal(0.0, std_v, n)
        w = yaw_rate + self.rng.normal(0.0, STD_YAW_RATE, n)

        for p, vi, wi in zip(self.particles, v, w):
            p.pose[2] += wi * dt
            p.pose[0] += vi * np.cos(p.pose[2]) * dt
            p.pose[1] += vi * np.sin(p.pose[2]) * dt

    def _updateParticle(self, p, detections, now):
        """Associa, atualiza os EKFs do mapa da particula e devolve o log-peso."""
        world = carToWorld(p.pose, detections)
        ranges = np.linalg.norm(detections[:, :2].astype(float), axis=1)

        log_lik = 0.0
        pairs_z, pairs_lm = [], []
        for z, rng_d, det in zip(world, ranges, detections):
            cls = int(det[2])
            meas_var = (MEAS_STD_BASE + MEAS_STD_PER_M * rng_d) ** 2

            idx = np.flatnonzero(p.lm_cls == cls)
            best = -1
            if len(idx):
                d = np.linalg.norm(p.lm_pos[idx] - z, axis=1)
                j = int(np.argmin(d))
                # deriva de pose desde a ultima vez que o cone foi visto
                # entra na covariancia da inovacao: no fechamento da volta
                # o gate cresce e a re-associacao vence a duplicacao
                age = min(now - p.lm_seen[idx[j]], DRIFT_AGE_CAP)
                drift_var = (DRIFT_STD_RATE * age) ** 2
                S = p.lm_var[idx[j]] + meas_var + drift_var
                gate = np.clip(3.0 * np.sqrt(S), GATE_MIN, GATE_MAX)
                if d[j] < gate:
                    best = idx[j]
                    dist = d[j]
                    S_assoc = S

            if best < 0:
                # deteccao sem par: novo landmark tentativo
                p.lm_pos = np.vstack([p.lm_pos, z])
                p.lm_var = np.append(p.lm_var, meas_var)
                p.lm_cls = np.append(p.lm_cls, cls)
                p.lm_hits = np.append(p.lm_hits, 1)
                p.lm_seen = np.append(p.lm_seen, now)
                log_lik += NEW_LM_LOGLIK
                continue

            # verossimilhanca da inovacao + update de Kalman do landmark
            S = S_assoc
            log_lik += -0.5 * dist ** 2 / S - np.log(2.0 * np.pi * S)
            gain = p.lm_var[best] / S
            p.lm_pos[best] += gain * (z - p.lm_pos[best])
            p.lm_var[best] *= (1.0 - gain)
            p.lm_hits[best] += 1
            p.lm_seen[best] = now

            if p.lm_hits[best] >= CONFIRM_HITS:
                pairs_z.append(z)
                pairs_lm.append(p.lm_pos[best].copy())

        self._refinePose(p, pairs_z, pairs_lm)
        return log_lik

    def _refinePose(self, p, pairs_z, pairs_lm):
        """Alinha a pose da particula ao proprio mapa (estilo FastSLAM 2.0).

        Ajuste rigido (rotacao + translacao) que leva as deteccoes vistas
        deste frame para cima dos landmarks confirmados casados. Aplicado
        com ganho e saturacao, impede a deriva da odometria de acumular
        em relacao ao mapa -- o que mantem o fechamento da volta trivial.
        """
        if len(pairs_z) < 2:
            return
        A = np.array(pairs_z)
        B = np.array(pairs_lm)
        ca, cb = A.mean(axis=0), B.mean(axis=0)
        a, b = A - ca, B - cb

        theta = np.arctan2(np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]),
                           np.sum(a * b))
        theta = np.clip(POSE_CORR_GAIN * theta,
                        -POSE_CORR_MAX_ROT, POSE_CORR_MAX_ROT)

        trans = POSE_CORR_GAIN * (cb - ca)
        norm = np.linalg.norm(trans)
        if norm > POSE_CORR_MAX_TRANS:
            trans *= POSE_CORR_MAX_TRANS / norm

        c, s = np.cos(theta), np.sin(theta)
        rel = p.pose[:2] - ca
        p.pose[0] = ca[0] + c * rel[0] - s * rel[1] + trans[0]
        p.pose[1] = ca[1] + s * rel[0] + c * rel[1] + trans[1]
        p.pose[2] += theta

    def _resample(self):
        """Reamostragem sistematica (low-variance)."""
        n = len(self.particles)
        positions = (np.arange(n) + self.rng.random()) / n
        idx = np.searchsorted(np.cumsum(self.weights), positions)
        idx = np.minimum(idx, n - 1)
        self.particles = [self.particles[i].copy() for i in idx]
        self.weights = np.full(n, 1.0 / n)

    def _prune(self, now):
        for p in self.particles:
            keep = (p.lm_hits >= CONFIRM_HITS) | (now - p.lm_seen < TENTATIVE_TIMEOUT)
            if not keep.all():
                p.lm_pos = p.lm_pos[keep]
                p.lm_var = p.lm_var[keep]
                p.lm_cls = p.lm_cls[keep]
                p.lm_hits = p.lm_hits[keep]
                p.lm_seen = p.lm_seen[keep]

    def _estimatePose(self):
        poses = np.array([p.pose for p in self.particles])
        # desfaz a temperagem na estimativa: a temperagem existe so para
        # segurar a reamostragem; a pose deve concentrar nas hipoteses
        # de maior verossimilhanca real
        w = self.weights ** (1.0 / LIK_TEMPER)
        w = w / w.sum()
        x = np.sum(w * poses[:, 0])
        y = np.sum(w * poses[:, 1])
        yaw = np.arctan2(np.sum(w * np.sin(poses[:, 2])),
                         np.sum(w * np.cos(poses[:, 2])))
        return np.array([x, y, yaw])

    # ---------- consultas (sempre no mapa da melhor particula) ----------

    def _best(self):
        return self.particles[int(np.argmax(self.weights))]

    def confirmed(self):
        """Cones confirmados do mapa: array [x, y, classId] no frame do SLAM."""
        p = self._best()
        mask = p.lm_hits >= CONFIRM_HITS
        if not mask.any():
            return np.empty((0, 3))
        return np.column_stack([p.lm_pos[mask], p.lm_cls[mask]])

    def localCones(self, pose=None, ahead=LOCAL_AHEAD, behind=LOCAL_BEHIND,
                   lateral=LOCAL_LATERAL):
        """Cones confirmados perto do carro em [lateral, frente, classId]."""
        if pose is None:
            pose = self.pose
        cones = self.confirmed()
        if len(cones) == 0:
            return np.empty((0, 3))
        local = worldToCar(pose, cones[:, :2])
        mask = ((local[:, 1] > -behind) & (local[:, 1] < ahead)
                & (np.abs(local[:, 0]) < lateral))
        return np.column_stack([local[mask], cones[mask, 2]])

    def mergeDuplicates(self, radius=MERGE_RADIUS):
        """Funde cones duplicados (mesma cor, muito proximos) do melhor mapa.

        Chamado no fechamento da volta, antes de gerar a trajetoria
        global: deteccoes que nao associaram viram um segundo cone ao
        lado do verdadeiro e atrapalham a triangulacao.
        """
        p = self._best()
        order = np.argsort(-p.lm_hits)
        keep, merged_pos, merged_w = [], [], []
        for i in order:
            placed = False
            for k, pos in enumerate(merged_pos):
                if (p.lm_cls[i] == p.lm_cls[keep[k]]
                        and np.linalg.norm(pos - p.lm_pos[i]) < radius):
                    w_new = 1.0 / max(p.lm_var[i], 1e-6)
                    merged_pos[k] = (pos * merged_w[k] + p.lm_pos[i] * w_new) \
                        / (merged_w[k] + w_new)
                    merged_w[k] += w_new
                    p.lm_hits[keep[k]] += p.lm_hits[i]
                    placed = True
                    break
            if not placed:
                keep.append(i)
                merged_pos.append(p.lm_pos[i].copy())
                merged_w.append(1.0 / max(p.lm_var[i], 1e-6))

        keep = np.array(keep, dtype=int)
        p.lm_pos = np.array(merged_pos)
        p.lm_var = 1.0 / np.array(merged_w)
        p.lm_cls = p.lm_cls[keep]
        p.lm_hits = p.lm_hits[keep]
        p.lm_seen = p.lm_seen[keep]

    def particlePoses(self):
        """Poses de todas as particulas (para visualizacao/debug)."""
        return np.array([p.pose for p in self.particles])


class LapTracker:
    """Conta voltas: fecha quando o carro volta perto da largada
    depois de ter rodado pelo menos MIN_LAP_DISTANCE."""

    def __init__(self, min_lap_distance=MIN_LAP_DISTANCE,
                 close_radius=CLOSE_RADIUS):
        self.min_lap_distance = min_lap_distance
        self.close_radius = close_radius
        self.start = None
        self.last_pos = None
        self.distance = 0.0
        self.laps = 0

    def setStart(self, pos):
        """Ancora o fechamento da volta num marco fisico (ex.: o centroide
        dos cones laranja da largada) em vez da primeira pose recebida."""
        self.start = np.asarray(pos[:2], dtype=float)

    def update(self, pos):
        """Recebe a posicao (x, y); retorna True no instante que fecha a volta."""
        pos = np.asarray(pos[:2], dtype=float)
        if self.start is None:
            self.start = pos.copy()
            self.last_pos = pos.copy()
            return False

        self.distance += float(np.linalg.norm(pos - self.last_pos))
        self.last_pos = pos.copy()

        if (self.distance > self.min_lap_distance
                and np.linalg.norm(pos - self.start) < self.close_radius):
            self.laps += 1
            self.distance = 0.0
            return True
        return False
