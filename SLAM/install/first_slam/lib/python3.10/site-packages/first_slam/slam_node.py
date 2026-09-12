import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray
from visualization_msgs.msg import MarkerArray, Marker
import math

class SlamConesNode(Node):
    
    def __init__(self):
        super().__init__('SlamConesNode')
        
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        self.cones_map = []
        
        # Limiar de distância (em metros) para decidir se o cone é novo ou repetido
        self.DIST_THRESHOLD = 0.6 

        self.cones_receiver = self.create_subscription(PoseArray, '/fsdsCones', self.cones_callback, 10)
        self.odom_receiver = self.create_subscription(Odometry, '/fsdsOdometry', self.odom_callback, 10)

        self.marker_publisher = self.create_publisher(MarkerArray, '/conesMapa', 10)
        self.get_logger().info("SLAM started.")

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        sin = 2 * (q.w * q.z + q.x * q.y)
        cos = 1 - 2 * (q.y * q.y + q.z * q.z)

        self.yaw = math.atan2(sin, cos)

    def cones_callback(self, msg):
        for pose_cone in msg.poses:
            x_local = pose_cone.position.x
            y_local = pose_cone.position.y

            # Transformação de coordenadas: Robô -> Global
            x_global = self.x + (x_local * math.cos(self.yaw) - y_local * math.sin(self.yaw))
            y_global = self.y + (x_local * math.sin(self.yaw) + y_local * math.cos(self.yaw))

            # Checagem: Esse cone já existe no nosso mapa?
            is_new_cone = True
            for cone_salvo in self.cones_map:
                # Calcula a distância euclidiana entre o cone detectado e os já salvos
                dist = math.sqrt((x_global - cone_salvo['x'])**2 + (y_global - cone_salvo['y'])**2)
                
                if dist < self.DIST_THRESHOLD:
                    is_new_cone = False
                    # Opcional: Atualiza levemente a posição fazendo uma média móvel filtrada
                    cone_salvo['x'] = 0.9 * cone_salvo['x'] + 0.1 * x_global
                    cone_salvo['y'] = 0.9 * cone_salvo['y'] + 0.1 * y_global
                    break # Encontrou o cone correspondente, pode parar o loop interno
            
            # Se passou pelo filtro e realmente for um cone novo, adiciona ao mapa
            if is_new_cone:
                self.cones_map.append({'x': x_global, 'y': y_global})

        # Publica o mapa atualizado para o RViz
        self.dados_to_rviz()

    def dados_to_rviz(self):
        marker_array = MarkerArray()
        
        for i, cone in enumerate(self.cones_map):
            marker = Marker()
            # IMPORTANTE: Alinhar com o frame global do seu simulador (geralmente 'fsds/map' ou 'map')
            marker.header.frame_id = "map" 
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = i
            
            marker.type = Marker.CYLINDER # Usando cilindro para representar o cone
            marker.action = Marker.ADD
            
            # Posição calculada
            marker.pose.position.x = cone['x']
            marker.pose.position.y = cone['y']
            marker.pose.position.z = 0.25 # Metade da altura para o cilindro ficar apoiado no chão
            
            # Dimensões do marcador (Diâmetro X, Diâmetro Y, Altura Z)
            marker.scale.x = 0.25
            marker.scale.y = 0.25
            marker.scale.z = 0.5
            
            # Cor do Marcador (Laranja)
            marker.color.r = 1.0
            marker.color.g = 0.5
            marker.color.b = 0.0
            marker.color.a = 1.0 # Opacidade total
            
            marker_array.markers.append(marker)
            
        self.marker_publisher.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = SlamConesNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()