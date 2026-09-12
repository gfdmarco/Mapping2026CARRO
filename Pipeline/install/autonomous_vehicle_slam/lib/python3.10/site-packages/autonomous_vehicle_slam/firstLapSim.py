"""Simulacao offline da volta 1 (sem o FSDS): valida MPC + mapeamento.

Roda o pipeline EXPLORE completo -- percepcao sintetica ruidosa (dropout
de cones e de frames inteiros, ruido crescendo com a distancia),
FastSLAM com odometria ruidosa, ConeMapper, planner local e pure
pursuit (identico ao carNode) -- numa pista fechada sintetica, e
verifica as duas condicoes que travavam a volta 1 no carro real:

  1. o carro completa o percurso sem sair do corredor de cones;
  2. o mapa fecha o laco e o GlobalPlanner produz a trajetoria global
     (exatamente a condicao que liga o modo TRACK no carNode);
  3. em modo GPS, segue a trajetoria global (MPC) por mais uma volta e
     verifica que o carro cruza a largada e continua na pista.

Por padrao localiza por GPS (pose verdadeira relativa a largada, igual
ao carNode com USE_GPS_POSE=True); com --slam usa o FastSLAM puro, que
para na construcao da trajetoria (a deriva impede a fase TRACK).

Uso: python3 -m autonomous_vehicle_slam.firstLapSim [--plot] [--slam]
"""

import sys
import numpy as np

from . import pathPlanning, control, slam, globalPlanning, mapping

TRACK_HALF_WIDTH = 1.75     # m
CONE_SPACING = 3.5          # m ao longo de cada borda
SIM_DT = 1.0 / 20.0         # s
SIM_TIME_LIMIT = 150.0      # s
EXPLORE_SPEED = 3.0         # m/s (igual ao carNode)


def buildTrack():
    """Pista fechada sintetica: circulo perturbado (curvas para os dois
    lados, raio minimo ~8 m). Retorna (centerline, cones [x, y, classId])."""
    t = np.linspace(0.0, 2.0 * np.pi, 2400, endpoint=False)
    r = 20.0 + 4.0 * np.cos(2.0 * t) + 1.5 * np.sin(3.0 * t)
    center = np.column_stack([r * np.cos(t), r * np.sin(t)])

    tang = np.gradient(center, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    normal = np.column_stack([-tang[:, 1], tang[:, 0]])   # 90 graus CCW

    left = center + TRACK_HALF_WIDTH * normal
    right = center - TRACK_HALF_WIDTH * normal

    cones = []
    for boundary, cls in ((left, pathPlanning.BLUE_ID),
                          (right, pathPlanning.YELLOW_ID)):
        seg = np.linalg.norm(np.diff(boundary, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        si = np.arange(0.0, s[-1], CONE_SPACING)
        x = np.interp(si, s, boundary[:, 0])
        y = np.interp(si, s, boundary[:, 1])
        cones.append(np.column_stack([x, y, np.full_like(x, cls)]))

    # cones laranja grandes (classe 2) marcando a linha de largada, um
    # par de cada lado ~2 m a frente do carro, como no FSDS
    for side in (+1.0, -1.0):
        for fwd_off in (1.5, 2.3):
            p = (center[0] + side * 0.85 * TRACK_HALF_WIDTH * normal[0]
                 + fwd_off * tang[0])
            cones.append(np.array([[p[0], p[1], 2.0]]))
    return center, np.vstack(cones)


def senseCones(cones_world, pose_true, rng):
    """Percepcao sintetica: FOV de camera, alcance limitado, ruido com a
    distancia, dropout por cone e por frame. Retorna [lat, frente, cls]."""
    if rng.random() < 0.03:                       # frame inteiro perdido
        return np.empty((0, 3))

    local = slam.worldToCar(pose_true, cones_world[:, :2])
    lat, fwd = local[:, 0], local[:, 1]
    dist = np.hypot(lat, fwd)
    ang = np.abs(np.arctan2(lat, np.maximum(fwd, 1e-9)))
    visible = (fwd > 0.5) & (dist < 18.0) & (ang < np.deg2rad(55.0))

    out = []
    for i in np.flatnonzero(visible):
        if rng.random() < 0.15:                   # dropout por cone
            continue
        std = 0.05 + 0.02 * dist[i]
        out.append([lat[i] + rng.normal(0.0, std),
                    fwd[i] + rng.normal(0.0, std),
                    cones_world[i, 2]])
    return np.array(out) if out else np.empty((0, 3))


def run(plot=False, seed=3, use_gps=True):
    rng = np.random.default_rng(seed)
    center, cones = buildTrack()
    perimeter = float(np.sum(np.linalg.norm(
        np.diff(np.vstack([center, center[:1]]), axis=0), axis=1)))
    print(f"pista: {perimeter:.0f} m, {len(cones)} cones, "
          f"pose: {'GPS' if use_gps else 'FastSLAM'}")

    # estado verdadeiro do carro (frame do mundo do sim)
    tang0 = center[1] - center[0]
    x, y = center[0]
    yaw = float(np.arctan2(tang0[1], tang0[0]))
    v, delta = 0.0, 0.0
    L = control.WHEELBASE
    max_steer = np.deg2rad(control.MAX_STEER_DEG)

    # "GPS": pose verdadeira relativa a largada (mesmo frame do carNode)
    x0, y0, yaw0 = x, y, yaw

    def gpsPose(px, py, pyaw):
        c, s = np.cos(yaw0), np.sin(yaw0)
        dx, dy = px - x0, py - y0
        d = pyaw - yaw0
        return np.array([dx * c + dy * s, -dx * s + dy * c,
                         np.arctan2(np.sin(d), np.cos(d))])

    # vies de odometria (calibracao imperfeita, como no carro real)
    speed_scale = 1.0 + rng.normal(0.0, 0.02)
    gyro_bias = np.deg2rad(rng.normal(0.0, 0.2))

    fast = slam.FastSLAM(n_particles=40, seed=seed)
    # a percepcao do sim ja entrega no frame do carro (sem offset de sensor)
    mapper = mapping.ConeMapper(sensor_fwd_offset=0.0)
    tracker = slam.LapTracker()
    planner = pathPlanning.PathPlanner()
    controller = control.VehicleController(target_speed=EXPLORE_SPEED)
    gplanner = globalPlanning.GlobalPlanner()

    trajectory, debug = None, {}
    lap_done = False
    started = False
    anchor = None
    last_try = 0.0
    max_offset = 0.0
    stuck_since = None

    for step in range(int(SIM_TIME_LIMIT / SIM_DT)):
        now = step * SIM_DT
        pose_true = np.array([x, y, yaw])

        detections = senseCones(cones, pose_true, rng)

        if use_gps:
            pose = gpsPose(x, y, yaw)
        else:
            w_true = v / L * np.tan(delta)
            v_meas = v * speed_scale + rng.normal(0.0, 0.05)
            w_meas = w_true + gyro_bias + rng.normal(0.0, np.deg2rad(0.5))
            pose = fast.update(v_meas, w_meas, detections, SIM_DT, now)
        mapper.update(pose, detections, now)

        # ancora a largada nos cones laranja assim que confirmados
        if anchor is None and not lap_done:
            anchor = mapper.startLineAnchor()
            if anchor is not None:
                tracker.setStart(anchor)
                print(f"t={now:5.1f} s: largada ancorada nos cones laranja "
                      f"em ({anchor[0]:.1f}, {anchor[1]:.1f})")

        if tracker.update(pose[:2]) and not lap_done:
            lap_done = True
            print(f"t={now:5.1f} s: volta fechada na largada")

        # mesma condicao do carNode: so tenta apos cruzar a largada
        if lap_done and now - last_try > 5.0:
            last_try = now
            mapper.mergeDuplicates()
            debug = {}
            # agressividade 0 na primeira trajetoria, igual ao carNode
            trajectory = gplanner.buildTrajectory(
                mapper.confirmed(), 0.0, debug)
            if trajectory is not None:
                print(f"t={now:5.1f} s: trajetoria global fechada "
                      f"({len(trajectory.points)} pontos)")
                break
            print(f"t={now:5.1f} s: laco ainda aberto {debug}")

        # pipeline EXPLORE identico ao carNode: deteccoes do frame ->
        # planner local -> pure pursuit (o MPC so entra no modo TRACK)
        path = planner.planPath(detections)
        target = EXPLORE_SPEED
        if not started:
            if len(path) >= 3:
                started = True
            elif tracker.distance < 20.0:
                # fase START: reto devagar ate achar os portoes
                path = np.array([[0.0, 0.0], [0.0, 4.0], [0.0, 8.0]])
                target = 1.5
        steering, throttle, brake = controller.compute(
            path, v, SIM_DT, target, use_mpc=False)

        # dinamica verdadeira: atuador de steering com atraso + bicicleta
        delta += (steering * max_steer - delta) * min(1.0, SIM_DT / 0.08)
        yaw += v / L * np.tan(delta) * SIM_DT
        x += v * np.cos(yaw) * SIM_DT
        y += v * np.sin(yaw) * SIM_DT
        v = max(0.0, v + (4.0 * throttle - 7.0 * brake - 0.03 * v) * SIM_DT)

        offset = float(np.min(np.hypot(center[:, 0] - x, center[:, 1] - y)))
        max_offset = max(max_offset, offset)
        if offset > TRACK_HALF_WIDTH:
            raise AssertionError(
                f"t={now:.1f} s: carro saiu do corredor ({offset:.2f} m "
                f"da linha central)")

        if v < 0.2 and now > 5.0:
            stuck_since = stuck_since if stuck_since is not None else now
            if now - stuck_since > 4.0:
                raise AssertionError(f"t={now:.1f} s: carro travado")
        else:
            stuck_since = None

    n_map = len(mapper.confirmed())
    print(f"desvio maximo da linha central: {max_offset:.2f} m")
    print(f"cones no mapa: {n_map} (reais: {len(cones)})")

    if use_gps:
        assert anchor is not None, \
            "cones laranja da largada nao viraram ancora no mapa"
    assert trajectory is not None, \
        f"trajetoria global nao fechou em {SIM_TIME_LIMIT:.0f} s: {debug}"

    traj_len = float(np.sum(np.linalg.norm(np.diff(
        np.vstack([trajectory.points, trajectory.points[:1]]), axis=0), axis=1)))
    err = abs(traj_len - perimeter) / perimeter
    print(f"comprimento da trajetoria: {traj_len:.0f} m "
          f"(real {perimeter:.0f} m, erro {100 * err:.1f}%)")
    assert err < 0.2, "comprimento da trajetoria global incoerente com a pista"
    assert abs(n_map - len(cones)) / len(cones) < 0.25, \
        "mapa com cones demais/de menos (duplicacao ou perda)"

    # fase TRACK (so em modo GPS): segue a trajetoria global por uma
    # volta inteira com o mesmo controlador do carNode (pure pursuit +
    # rampa de velocidade) -- o carro tem que cruzar a largada e
    # continuar na pista
    if use_gps:
        track_dist, max_track_off = 0.0, 0.0
        track_speed = EXPLORE_SPEED
        while track_dist < 1.1 * perimeter:
            pose = gpsPose(x, y, yaw)
            path, tspd, _ = trajectory.localWindow(pose, slam.worldToCar, 30.0)
            track_speed = min(tspd, track_speed + 0.5 * SIM_DT)
            steering, throttle, brake = controller.compute(
                path, v, SIM_DT, track_speed, use_mpc=False)

            delta += (steering * max_steer - delta) * min(1.0, SIM_DT / 0.08)
            yaw += v / L * np.tan(delta) * SIM_DT
            x += v * np.cos(yaw) * SIM_DT
            y += v * np.sin(yaw) * SIM_DT
            v = max(0.0, v + (4.0 * throttle - 7.0 * brake - 0.03 * v) * SIM_DT)
            track_dist += v * SIM_DT

            off = float(np.min(np.hypot(center[:, 0] - x, center[:, 1] - y)))
            max_track_off = max(max_track_off, off)
            if off > TRACK_HALF_WIDTH:
                raise AssertionError(
                    f"TRACK: carro saiu do corredor ({off:.2f} m da linha "
                    f"central) apos {track_dist:.0f} m")
        print(f"TRACK: volta completa na trajetoria global cruzando a "
              f"largada, desvio maximo {max_track_off:.2f} m, "
              f"v final {v:.1f} m/s")

    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(center[:, 0], center[:, 1], '--', c='gray', lw=0.8,
                label='linha central real')
        ax.scatter(cones[:, 0], cones[:, 1], c='lightgray', s=12)
        m = mapper.confirmed()
        blue = m[:, 2].astype(int) == pathPlanning.BLUE_ID
        ax.scatter(m[blue, 0], m[blue, 1], c='blue', s=20, label='mapa azul')
        ax.scatter(m[~blue, 0], m[~blue, 1], c='gold', s=20,
                   edgecolors='k', linewidths=0.3, label='mapa amarelo')
        ax.plot(trajectory.points[:, 0], trajectory.points[:, 1], 'r-',
                lw=1.5, label='trajetoria global (frame do SLAM)')
        ax.set_aspect('equal')
        ax.legend()
        ax.set_title('volta 1 simulada: mapa e trajetoria no frame do SLAM')
        plt.show()

    print("simulacao da volta 1 OK")


if __name__ == "__main__":
    run(plot="--plot" in sys.argv, use_gps="--slam" not in sys.argv)
