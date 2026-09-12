import rclpy
from rclpy.node import Node
import os, sys
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import numpy as np

from . import perception, pathPlanning, control, slam, globalPlanning, mapping
from .visualizer import MapVisualizer

fsds_lib_path = os.path.join(os.path.expanduser("~"),
                             "Formula-Student-Driverless-Simulator", "python")
sys.path.insert(0, fsds_lib_path)

import fsds

DATA_PUBLISH_FREQUENCY = 60

# Liga o plot matplotlib ao vivo do mapa (desligar em producao)
ENABLE_PLOT = True

EXPLORE_SPEED = 2.0      # m/s na volta 1 (mapeando; conservador de proposito)

# Fase START: na largada o carro esta entre os cones laranja e os
# primeiros portoes azul/amarelo podem nao bastar para planejar -- vai
# reto devagar ate o planner produzir uma rota de verdade
START_CREEP_SPEED = 1.5  # m/s indo reto na largada
START_MAX_DISTANCE = 20.0  # m maximos indo reto antes de parar por seguranca
MIN_PATH_POINTS = 3      # rota valida = carro + pelo menos 2 midpoints
TRACK_WINDOW = 30.0      # m de trajetoria global entregue ao pure pursuit
TRAJECTORY_RETRY_PERIOD = 5.0  # s entre tentativas de fechar o mapa
TRACK_SPEED_RAMP = 0.5   # m/s^2 de subida do alvo de velocidade no TRACK

# Controlador lateral do modo TRACK: False = pure pursuit (o mesmo
# controlador ja validado da volta 1; a trajetoria global e suave e ele
# a segue bem). True = MPC -- so ligar para calibrar com o simulador:
# o MPC e linearizado para erros pequenos e, com a malha rodando na
# frequencia do YOLO, fora da linha ele satura e roda em circulos.
USE_MPC_TRACK = False

# Fonte da pose no frame do mapa:
#   True  -> posicao/orientacao do simulador (o "GPS"), sem deriva: ao
#            terminar a volta a pose volta exatamente na largada e a
#            trajetoria global e seguida no lugar certo da pista.
#   False -> FastSLAM puro (odometria + cones, sem posicao do sim). A
#            deriva acumulada na volta desloca o frame do mapa: o carro
#            fecha a volta "em outro lugar".
USE_GPS_POSE = True


class GpsLocalizer:
    """Pose (x, y, yaw) no frame do mapa a partir do estado do simulador.

    Usa a posicao e a orientacao do getCarState() (o "GPS" do FSDS),
    expressas em relacao a pose inicial do carro -- o mesmo frame
    (largada na origem, yaw 0) em que o mapa de cones e a trajetoria
    global sao construidos. Sem deriva: ao terminar a volta a pose
    coincide com a largada e a trajetoria e seguida no lugar certo.
    """

    def __init__(self):
        self.origin = None

    def update(self, state):
        kin = state.kinematics_estimated
        x, y = kin.position.x_val, kin.position.y_val
        yaw = slam.quaternionToYaw(kin.orientation)
        if self.origin is None:
            self.origin = (x, y, yaw)
        x0, y0, yaw0 = self.origin
        c, s = np.cos(yaw0), np.sin(yaw0)
        dx, dy = x - x0, y - y0
        dyaw = yaw - yaw0
        return np.array([dx * c + dy * s,
                         -dx * s + dy * c,
                         np.arctan2(np.sin(dyaw), np.cos(dyaw))])


class CarNodeSlam(Node):
    """Pipeline com FastSLAM em duas fases.

    EXPLORE (volta 1): dirige exatamente como o pipeline original
    (planner local sobre as deteccoes do frame + pure pursuit), devagar,
    enquanto o ConeMapper constroi o mapa global em paralelo. O mapa
    NAO interfere na conducao da volta 1; ele so serve para fechar o
    laco e gerar a trajetoria global.

    TRACK (voltas 2+): com a volta fechada, calcula a trajetoria global
    otimizada no frame do mapa e passa a segui-la com o MPC. A cada
    volta o mapa refina e a agressividade (linha + velocidade) aumenta.

    A pose vem do GPS (posicao do simulador, ver USE_GPS_POSE); com
    USE_GPS_POSE=False volta o FastSLAM puro (odometria + cones), que
    sofre deriva ao longo da volta.
    """

    def __init__(self):
        super().__init__("CarNodeSlam")

        try:
            self.client = fsds.FSDSClient()
            self.client.confirmConnection()
            self.client.enableApiControl(True)

        except Exception as e:
            raise RuntimeError(f"Unable to connect to simulator.\n{e}")

        self.perception = perception.Perception(self.client)
        self.planner = pathPlanning.PathPlanner()
        self.controller = control.VehicleController(target_speed=EXPLORE_SPEED)

        self.slam = slam.FastSLAM()
        self.gpsLocalizer = GpsLocalizer()
        self.mapper = mapping.ConeMapper()
        self.lapTracker = slam.LapTracker()
        self.globalPlanner = globalPlanning.GlobalPlanner()
        self.trajectory = None
        self.laps = 0
        self.lapDone = False
        self.started = False         # ja saiu da fase START (arranque)
        self.startAnchor = None      # linha de largada (cones laranja)
        self.lastTrajAttempt = 0.0
        self.lastTrajIndex = None
        self.trackSpeed = EXPLORE_SPEED

        self.visualizer = MapVisualizer() if ENABLE_PLOT else None

        self.path = np.empty((0, 2))
        self.lastTime = self.get_clock().now()

        self.conesPublisher = self.create_publisher(Float32MultiArray, "cones", 10)
        self.create_timer(1.0 / DATA_PUBLISH_FREQUENCY, self.input_process)

    @property
    def mode(self):
        if self.trajectory is not None:
            return "TRACK"
        return "EXPLORE" if self.started else "START"

    def input_process(self):
        [image] = self.client.simGetImages(
            [fsds.ImageRequest(camera_name='ZED_RGB',
                               image_type=fsds.ImageType.Scene,
                               pixels_as_float=False,
                               compress=True)],
            vehicle_name='FSCar'
        )

        state = self.client.getCarState()
        speed = state.speed
        yaw_rate = state.kinematics_estimated.angular_velocity.z_val

        now_clock = self.get_clock().now()
        dt = (now_clock - self.lastTime).nanoseconds * 1e-9
        self.lastTime = now_clock
        now = now_clock.nanoseconds * 1e-9

        detections = self.perception.detectCones(image)

        # pose no frame do mapa: GPS (posicao do simulador) por padrao;
        # USE_GPS_POSE=False volta ao FastSLAM puro (odometria + cones)
        if USE_GPS_POSE:
            pose = self.gpsLocalizer.update(state)
        else:
            pose = self.slam.update(speed, yaw_rate, detections, dt, now)
        # mapa global robusto, construido sobre a pose escolhida
        self.mapper.update(pose, detections, now)

        # assim que os cones laranja da largada estiverem confirmados no
        # mapa, o fechamento da volta passa a ser ancorado neles (marco
        # fisico da linha de chegada) em vez da pose inicial do carro
        if self.startAnchor is None and not self.lapDone:
            anchor = self.mapper.startLineAnchor()
            if anchor is not None:
                self.startAnchor = anchor
                self.lapTracker.setStart(anchor)
                self.get_logger().info(
                    f"largada ancorada nos cones laranja em "
                    f"({anchor[0]:.1f}, {anchor[1]:.1f})")

        if self.lapTracker.update(pose[:2]) and not self.lapDone:
            self.lapDone = True
            self.get_logger().info(
                "volta 1 fechada na largada; gerando a trajetoria global")

        # o modo TRACK so liga DEPOIS de o carro cruzar fisicamente a
        # largada (LapTracker na pose GPS). Tentar fechar o mapa no meio
        # da volta podia "fechar" por um atalho falso entre trechos
        # paralelos da pista e trocar de modo num lugar aleatorio. Se a
        # construcao falhar na largada, segue em EXPLORE e tenta de novo
        # a cada TRAJECTORY_RETRY_PERIOD
        if (self.trajectory is None and self.lapDone
                and now - self.lastTrajAttempt > TRAJECTORY_RETRY_PERIOD):
            self.lastTrajAttempt = now
            self.try_build_trajectory()

        # planeja e controla conforme o modo
        if self.trajectory is not None:
            self.path, traj_speed, traj_index = self.trajectory.localWindow(
                pose, slam.worldToCar, TRACK_WINDOW)
            # rampa: sobe gradualmente da velocidade de exploracao ate o
            # perfil da trajetoria, sem acelerar de supetao na transicao
            self.trackSpeed = min(traj_speed,
                                  self.trackSpeed + TRACK_SPEED_RAMP * max(dt, 0.0))
            target_speed = self.trackSpeed
            self.count_track_lap(traj_index)
        else:
            # volta 1: planeja direto nas deteccoes do frame, identico ao
            # pipeline original que ja funciona. O mapa e usado apenas
            # para gerar a trajetoria global no fim da volta.
            self.path = self.planner.planPath(detections)
            target_speed = EXPLORE_SPEED

            if not self.started:
                if len(self.path) >= MIN_PATH_POINTS:
                    self.started = True
                    self.get_logger().info(
                        "portoes detectados; saindo da fase START")
                elif self.lapTracker.distance < START_MAX_DISTANCE:
                    # arranque: entre os cones laranja da largada ainda
                    # nao ha portoes azul/amarelo suficientes -- segue
                    # reto devagar ate eles entrarem na percepcao
                    self.path = np.array([[0.0, 0.0], [0.0, 4.0], [0.0, 8.0]])
                    target_speed = START_CREEP_SPEED
                # acima de START_MAX_DISTANCE sem rota: cai no compute
                # sem rota valida, que freia suavemente (seguranca)

        # volta 1 = pure pursuit, igual ao pacote autonomous_vehicle;
        # TRACK tambem, a menos que USE_MPC_TRACK esteja ligado
        steering, throttle, brake = self.controller.compute(
            self.path, speed, dt, target_speed,
            use_mpc=USE_MPC_TRACK and self.trajectory is not None)
        self.client.setCarControls(
            fsds.CarControls(throttle=throttle, steering=steering, brake=brake))

        self.publish_cones(detections)

        if self.visualizer is not None:
            particles = None if USE_GPS_POSE else self.slam.particlePoses()
            self.visualizer.update(pose, self.mapper, self.trajectory,
                                   self.path, self.mode, particles)

    def try_build_trajectory(self):
        # primeira trajetoria com agressividade 0 (linha central pura,
        # velocidade minima): o mapa recem-fechado ainda e o menos
        # confiavel, e a entrada no modo TRACK ja troca o controlador --
        # a agressividade sobe a cada volta completada no TRACK
        lap = max(1, self.laps)
        aggressiveness = min(1.0, globalPlanning.AGGRESSIVENESS_PER_LAP
                             * (lap - 1))

        self.mapper.mergeDuplicates()
        debug = {}
        trajectory = self.globalPlanner.buildTrajectory(
            self.mapper.confirmed(), aggressiveness, debug)

        # no modo FastSLAM o mapa da melhor particula e a segunda chance
        # (no modo GPS o FastSLAM nao roda e nao tem mapa)
        if trajectory is None and not USE_GPS_POSE:
            self.slam.mergeDuplicates()
            trajectory = self.globalPlanner.buildTrajectory(
                self.slam.confirmed(), aggressiveness, debug)

        if trajectory is None:
            self.get_logger().info(
                f"mapa ainda nao fecha o laco ({self.lapTracker.distance:.0f} m "
                f"percorridos): {debug}")
            return

        first_time = self.trajectory is None
        self.trajectory = trajectory
        self.lastTrajIndex = None
        if first_time:
            self.laps = max(1, self.laps)
        self.get_logger().info(
            f"trajetoria global pronta (volta {self.laps}): "
            f"{len(trajectory.points)} pontos, agressividade "
            f"{aggressiveness:.2f}, v max {trajectory.speeds.max():.1f} m/s")

    def count_track_lap(self, traj_index):
        """Conta voltas no modo TRACK pelo progresso ao longo da trajetoria:
        quando o indice mais proximo da o salto de 'fim do laco' para o
        comeco (em qualquer sentido), fechou mais uma volta."""
        n = len(self.trajectory.points)
        if self.lastTrajIndex is not None:
            jump = abs(traj_index - self.lastTrajIndex)
            if jump > 0.6 * n:
                self.laps += 1
                self.get_logger().info(f"volta {self.laps} completa no modo "
                                       "TRACK; reotimizando a trajetoria")
                self.try_build_trajectory()
        self.lastTrajIndex = traj_index

    def publish_cones(self, cones):
        msg = Float32MultiArray()

        cones = np.asarray(cones, dtype=np.float32)
        rows = cones.shape[0] if cones.ndim == 2 else 0
        cols = cones.shape[1] if cones.ndim == 2 else 0

        msg.layout.dim = [
            MultiArrayDimension(label="cones", size=rows, stride=rows * cols),
            MultiArrayDimension(label="cone", size=cols, stride=cols),
        ]
        msg.data = cones.flatten().tolist()

        self.conesPublisher.publish(msg)


def main():
    rclpy.init()
    node = CarNodeSlam()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
