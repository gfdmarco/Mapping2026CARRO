import math
import random

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class MockPerception(Node):

    def __init__(self):
        super().__init__('mock_perception_realistic')

        self.publisher = self.create_publisher(
            Float32MultiArray,
            '/lidar_node/cones',
            10
        )

        # -----------------------------
        # Simulação do veículo
        # -----------------------------
        self.car_s = 0.0          # posição longitudinal "virtual" na pista
        self.speed = 4.0          # m/s
        self.dt = 0.1             # 10 Hz

        # -----------------------------
        # Limites da perception
        # -----------------------------
        self.min_range = 1.5
        self.max_range = 28.0
        self.max_lateral = 9.0

        # -----------------------------
        # Imperfeições da perception
        # -----------------------------
        self.dropout_probability = 0.08

        # ruído típico fictício
        self.lateral_noise_std = 0.08
        self.forward_noise_std = 0.12

        # deslocamento pequeno sistemático da percepção
        self.lateral_bias = 0.03

        # Cones físicos fixos ao longo da pista
        self.track_cones = self._generate_track_cones()

        self.timer = self.create_timer(
            self.dt,
            self.publish_cones
        )

        self.get_logger().info(
            f"Mock iniciado com {len(self.track_cones)} cones físicos."
        )

    # ============================================================
    # GEOMETRIA DA PISTA
    # ============================================================

    def centerline(self, s):
        """
        Deslocamento lateral do centro da pista.

        Produz:
        - reta inicial
        - curva para a direita
        - transição
        - contra-curva para a esquerda
        - trecho final mais suave
        """

        if s < 15.0:
            # reta inicial
            return 0.0

        elif s < 35.0:
            # curva progressiva para a direita
            u = (s - 15.0) / 20.0
            return 3.0 * (0.5 - 0.5 * math.cos(math.pi * u))

        elif s < 55.0:
            # sai da curva e retorna gradualmente
            u = (s - 35.0) / 20.0
            return 3.0 - 4.5 * (0.5 - 0.5 * math.cos(math.pi * u))

        elif s < 80.0:
            # contra-curva suave
            u = (s - 55.0) / 25.0
            return -1.5 + 2.0 * math.sin(math.pi * u)

        else:
            # oscilação suave distante
            return 0.7 * math.sin((s - 80.0) / 12.0)

    def track_width(self, s):
        """
        Largura da pista variando levemente.
        """

        return (
            4.0
            + 0.20 * math.sin(s / 13.0)
            + 0.10 * math.sin(s / 5.5)
        )

    # ============================================================
    # CONES FÍSICOS
    # ============================================================

    def _generate_track_cones(self):
        """
        Gera os cones no mundo uma vez.

        Eles ficam fixos no 'mundo'.
        Quem se move é o carro.

        Cada item:
        {
            's': posição longitudinal,
            'lateral': posição lateral global,
            'class_id': 0 azul / 1 amarelo
        }
        """

        cones = []

        s = 3.0

        while s < 140.0:

            # espaçamento aproximadamente 3-5 metros
            spacing = random.uniform(3.2, 4.8)

            center = self.centerline(s)
            width = self.track_width(s)

            # pequenas imperfeições físicas da colocação dos cones
            left_jitter = random.uniform(-0.10, 0.10)
            right_jitter = random.uniform(-0.10, 0.10)

            blue_lateral = center - width / 2.0 + left_jitter
            yellow_lateral = center + width / 2.0 + right_jitter

            cones.append({
                's': s,
                'lateral': blue_lateral,
                'class_id': 0
            })

            cones.append({
                's': s + random.uniform(-0.15, 0.15),
                'lateral': yellow_lateral,
                'class_id': 1
            })

            s += spacing

        return cones

    # ============================================================
    # SENSOR / PERCEPTION
    # ============================================================

    def publish_cones(self):

        msg = Float32MultiArray()

        detections = []

        for cone in self.track_cones:

            # distância à frente relativa ao carro
            frente_real = cone['s'] - self.car_s

            if frente_real < self.min_range:
                continue

            if frente_real > self.max_range:
                continue

            # posição lateral relativa
            lateral_real = cone['lateral']

            if abs(lateral_real) > self.max_lateral:
                continue

            # ----------------------------------------------------
            # Dropout: cone ocasionalmente não detectado
            # ----------------------------------------------------
            if random.random() < self.dropout_probability:
                continue

            # ----------------------------------------------------
            # Ruído de medição
            # ----------------------------------------------------
            lateral_medida = (
                lateral_real
                + self.lateral_bias
                + random.gauss(0.0, self.lateral_noise_std)
            )

            frente_medida = (
                frente_real
                + random.gauss(0.0, self.forward_noise_std)
            )

            # não deixa ruído criar distância impossível
            if frente_medida <= 0.5:
                continue

            detections.append([
                lateral_medida,
                frente_medida,
                float(cone['class_id'])
            ])

        # --------------------------------------------------------
        # Às vezes ordena aproximadamente por distância,
        # mas não perfeitamente.
        #
        # Perception real não necessariamente publica tudo
        # perfeitamente ordenado.
        # --------------------------------------------------------

        detections.sort(key=lambda c: c[1])

        # pequena bagunça ocasional entre detecções próximas
        if len(detections) >= 4 and random.random() < 0.15:
            i = random.randint(0, len(detections) - 2)
            detections[i], detections[i + 1] = \
                detections[i + 1], detections[i]

        # --------------------------------------------------------
        # Monta Float32MultiArray
        # --------------------------------------------------------

        data = []

        for cone in detections:
            data.extend(cone)

        msg.data = data

        self.publisher.publish(msg)

        self.get_logger().info(
            f"s={self.car_s:6.2f} m | "
            f"cones visíveis={len(detections):2d}"
        )

        # --------------------------------------------------------
        # Move o carro
        # --------------------------------------------------------

        self.car_s += self.speed * self.dt


def main(args=None):

    rclpy.init(args=args)

    node = MockPerception()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
