# Pipeline de Planejamento — CarNode, Path Planning e Integração com Cartographer

Este documento descreve o estado atual do pipeline de planejamento do carro, com foco em:

- funcionamento do `CarNode`;
- funcionamento do `pathPlanning.py`;
- comunicação com Perception e Controle;
- uso de odometria real do INS;
- integração futura com Cartographer;
- uso da árvore TF iniciada pelo `run.sh`;
- comandos para compilar e executar o sistema.

---

# 1. Visão geral da arquitetura

O pipeline atual foi adaptado para sair do ambiente simulado do FSDS e passar a trabalhar com nós ROS 2 reais.

A estrutura desejada é:

```text
Perception real
    |
    v
/lidar_node/cones
Float32MultiArray
    |
    v
CarNode
    |
    v
PathPlanner
    |
    v
/teste/route
PoseStamped
    |
    v
Controle real
```

Em paralelo, a localização deve vir do INS e do Cartographer:

```text
SBG INS
   |
   v
/imu/odometry
   |
   +---- TF odom -> base_link

Cartographer
   |
   +---- TF map -> odom

Árvore final:

map
 |
 v
odom
 |
 v
base_link
```

O `CarNode` usa essa árvore TF para obter a pose do carro sem precisar conhecer diretamente a origem dos dados.

---

# 2. CarNode

O `CarNode` é o nó central de integração entre:

- Perception;
- Path Planning;
- Controle;
- odometria;
- Cartographer / TF;
- lógica futura de fechamento de volta.

Ele **não controla mais o carro diretamente**.

No ambiente simulado, o `CarNode`:

- recebia imagens do FSDS;
- executava Perception internamente;
- calculava rota;
- calculava steering/throttle/brake;
- enviava `CarControls` diretamente ao simulador.

No carro real, essa responsabilidade foi separada.

Agora o `CarNode`:

```text
recebe cones
    ↓
calcula a rota
    ↓
envia um waypoint para Controle
```

---

# 3. Entrada da Perception

O `CarNode` assina:

```text
/lidar_node/cones
```

Tipo:

```text
std_msgs/msg/Float32MultiArray
```

Os cones são recebidos em grupos de três valores:

```text
[lateral, frente, classe]
```

Exemplo:

```text
[
    -2.0, 4.0, 0.0,
     2.0, 4.1, 1.0,
    -1.8, 8.0, 0.0,
     2.3, 8.2, 1.0
]
```

Convenção usada:

```text
classe 0 = cone azul
classe 1 = cone amarelo
```

No `CarNode`, o callback converte a mensagem para uma matriz NumPy:

```python
def _cb_cones(self, msg):
    flat = np.array(msg.data, dtype=np.float64)

    if flat.size == 0:
        self.cones = np.empty((0, 3))
        return

    if flat.size % 3 != 0:
        self.get_logger().warn(
            f"Mensagem de cones inválida: {flat.size} valores; "
            "esperava múltiplo de 3"
        )
        return

    self.cones = flat.reshape(-1, 3)
```

O resultado interno fica no formato:

```text
N x 3
```

onde cada linha representa:

```text
[lateral, frente, classId]
```

---

# 4. Path Planning

O arquivo `pathPlanning.py` contém a classe:

```python
PathPlanner
```

O objetivo é construir a linha central da pista usando os cones detectados.

Entrada:

```text
[lateral, frente, classId]
```

Saída:

```text
[lateral, frente]
```

para cada waypoint calculado.

---

# 5. Triangulação de Delaunay

O `PathPlanner` utiliza:

```python
scipy.spatial.Delaunay
```

para construir uma triangulação entre os cones detectados.

Depois, o algoritmo analisa as arestas dos triângulos e mantém apenas as arestas que conectam:

```text
cone azul <-> cone amarelo
```

O midpoint dessas arestas é usado como candidato à linha central da pista.

Conceitualmente:

```text
Azul                 Amarelo

  ●---------------------●
            |
            x   <- midpoint
```

Os midpoints formam a sequência de waypoints.

---

# 6. Filtros usados no PathPlanner

Existem dois limites importantes.

## `MAX_EDGE_LENGTH`

Descarta pares azul-amarelo separados por distância excessiva.

Isso evita que a triangulação conecte cones pertencentes a regiões muito distantes da pista.

## `MAX_STEP_LENGTH`

Impede que a rota dê saltos grandes entre dois waypoints consecutivos.

Isso ajuda a evitar uma rota incorreta quando a triangulação produz conexões ruins.

---

# 7. Ordenação dos waypoints

Depois dos midpoints serem encontrados, eles são ordenados começando na posição do carro:

```text
[0, 0]
```

A rota final começa com:

```text
[0, 0]
```

seguida pelos waypoints calculados.

Exemplo:

```text
[
    [0.0, 0.0],
    [0.1, 4.0],
    [0.5, 8.0],
    [1.2, 12.0]
]
```

---

# 8. Saída para Controle

Atualmente o sistema não envia a rota inteira.

O `CarNode` pega:

```python
path[1]
```

ou seja, o primeiro waypoint real depois da origem do carro.

Esse waypoint é publicado em:

```text
/teste/route
```

Tipo:

```text
geometry_msgs/msg/PoseStamped
```

No código:

```python
lateral, frente = path[1]

msg.pose.position.x = float(frente)
msg.pose.position.y = float(-lateral)
```

A convenção usada atualmente é:

```text
x = frente
y = -lateral
```

Isso foi escolhido para ficar compatível com a convenção ROS usual de `base_link`:

```text
+x = frente
+y = esquerda
```

A Perception / PathPlanner usam lateral positivo para a direita, por isso existe o sinal negativo.

Esse ponto deve continuar alinhado com o time de Controle.

---

# 9. Controle de autorização

O `CarNode` assina:

```text
/enable_control
```

Tipo:

```text
std_msgs/msg/Bool
```

Comportamento:

```text
true  -> publica /teste/route
false -> não publica /teste/route
```

Isso permite bloquear a saída para Controle sem encerrar o nó.

---

# 10. Odometria real

O pipeline está sendo preparado para usar o INS SBG.

O driver novo do SBG deve publicar:

```text
/imu/odometry
```

Tipo:

```text
nav_msgs/msg/Odometry
```

Além disso, com a configuração:

```yaml
odometry:
  enable: true
  publishTf: true
  odomFrameId: "odom"
  baseFrameId: "base_link"
  initFrameId: "map"
```

o driver deverá publicar a transformação:

```text
odom -> base_link
```

---

# 11. Por que o CarNode usa TF

O `CarNode` já possui:

```python
get_pose(reference_frame)
```

Essa função faz:

```python
lookup_transform(
    reference_frame,
    'base_link',
    ...
)
```

Isso significa que ela não depende diretamente de um sensor específico.

Exemplos:

```python
get_pose("odom")
```

procura:

```text
odom -> base_link
```

e:

```python
get_pose("map")
```

procura:

```text
map -> base_link
```

Essa abstração é importante porque permite trocar a origem da localização sem alterar a lógica do `CarNode`.

---

# 12. Integração com Cartographer

O Cartographer deverá fornecer a transformação:

```text
map -> odom
```

Enquanto o INS deverá fornecer:

```text
odom -> base_link
```

Assim, a árvore TF completa será:

```text
map
 |
 v
odom
 |
 v
base_link
```

O TF2 consegue compor automaticamente:

```text
map -> odom
+
odom -> base_link
=
map -> base_link
```

Por isso o código atual pode continuar usando:

```python
get_pose("map")
get_pose("odom")
```

sem mudanças estruturais.

---

# 13. `run.sh` do Cartographer

O projeto possui um arquivo de inicialização:

```text
run.sh
```

responsável por iniciar o Cartographer e também as transformações TF necessárias para o sistema.

O papel esperado desse script é deixar o ambiente de localização pronto antes do `CarNode` tentar consultar:

```text
map -> base_link
```

e:

```text
odom -> base_link
```

A ordem lógica do sistema completo será:

```text
1. iniciar INS / sbg_driver
2. iniciar run.sh do Cartographer / TF
3. iniciar Perception
4. iniciar CarNode
5. iniciar Controle
```

---

# 14. Verificar a árvore TF

Depois de iniciar INS e Cartographer, verifique:

```bash
ros2 run tf2_ros tf2_echo odom base_link
```

Esse comando deve mostrar continuamente posição e orientação.

Depois:

```bash
ros2 run tf2_ros tf2_echo map odom
```

E finalmente:

```bash
ros2 run tf2_ros tf2_echo map base_link
```

Se os três funcionarem, o `get_pose()` do `CarNode` terá tudo que precisa.

---

# 15. Fechamento de volta

O código possui a função:

```python
check_lap_closure()
```

Atualmente ela está desabilitada através de:

```python
ENABLE_LAP_CLOSURE = False
```

Ela usa:

```python
map_pose = self.get_pose("map")
odom_pose = self.get_pose("odom")
```

e verifica:

- distância total percorrida;
- distância até a posição inicial;
- diferença de heading.

Quando as condições são satisfeitas:

```text
distância percorrida >= limite
distância até início <= limite
diferença de heading <= limite
```

o nó publica:

```text
/mapping/lap_closed
```

Tipo:

```text
std_msgs/msg/Bool
```

Antes de ativar:

```python
ENABLE_LAP_CLOSURE = True
```

é necessário confirmar que:

```text
map -> odom
odom -> base_link
```

estão sendo publicados corretamente.

---

# 16. Compilar o pipeline

Entre no workspace:

```bash
cd ~/Mapping2026CARRO/Pipeline
```

Compile:

```bash
colcon build --packages-select autonomous_vehicle
```

Depois:

```bash
source install/setup.bash
```

Sempre que alterar arquivos Python ou configuração do pacote, compile novamente se necessário.

---

# 17. Rodar o CarNode

Depois de carregar o ambiente:

```bash
cd ~/Mapping2026CARRO/Pipeline
source install/setup.bash
```

Rode:

```bash
ros2 run autonomous_vehicle CarNode
```

O nome exato do executável depende do `setup.py` do pacote.

---

# 18. Testar entrada de Perception

É possível simular a Perception manualmente:

```bash
ros2 topic pub /lidar_node/cones \
std_msgs/msg/Float32MultiArray \
"{data: [-2.0, 4.0, 0.0, 2.0, 4.0, 1.0, -2.0, 8.0, 0.0, 2.0, 8.0, 1.0]}" \
-r 10
```

---

# 19. Ver a saída para Controle

Em outro terminal:

```bash
ros2 topic echo /teste/route
```

Deve aparecer uma mensagem:

```text
geometry_msgs/msg/PoseStamped
```

com algo como:

```yaml
header:
  frame_id: base_link

pose:
  position:
    x: 4.0
    y: ...
    z: 0.0
```

---

# 20. Testar autorização

Desabilitar:

```bash
ros2 topic pub --once \
/enable_control \
std_msgs/msg/Bool \
"{data: false}"
```

Habilitar:

```bash
ros2 topic pub --once \
/enable_control \
std_msgs/msg/Bool \
"{data: true}"
```

---

# 21. Rodar com dados de rosbag

Para testar usando um bag gravado anteriormente:

```bash
ros2 bag play mock_realistic \
--topics /lidar_node/cones
```

Em paralelo:

```bash
ros2 run autonomous_vehicle CarNode
```

E:

```bash
ros2 topic echo /teste/route
```

Isso permite testar novas versões do planejamento usando exatamente os mesmos dados de Perception.

---

# 22. Rodar o pipeline completo

Quando o INS e Cartographer estiverem prontos, a sequência recomendada é:

## Terminal 1 — INS

Iniciar o serviço ou launch do SBG.

Depois confirmar:

```bash
ros2 topic echo /imu/odometry
```

e:

```bash
ros2 run tf2_ros tf2_echo odom base_link
```

## Terminal 2 — Cartographer

Executar:

```bash
./run.sh
```

Confirmar:

```bash
ros2 run tf2_ros tf2_echo map odom
```

## Terminal 3 — Perception

Executar o nó real de Perception.

Confirmar:

```bash
ros2 topic echo /lidar_node/cones
```

## Terminal 4 — CarNode

```bash
cd ~/Mapping2026CARRO/Pipeline
source install/setup.bash

ros2 run autonomous_vehicle CarNode
```

## Terminal 5 — Controle

Executar o nó do time de Controle.

Confirmar:

```bash
ros2 topic info /teste/route -v
```

O Controle deve aparecer como subscriber.

---

# 23. Estado atual do projeto

Atualmente já foi validado:

- recepção de `Float32MultiArray` em `/lidar_node/cones`;
- conversão das detecções para matriz `N x 3`;
- Path Planning usando Delaunay;
- geração de midpoints;
- ordenação dos waypoints;
- publicação de um waypoint em `/teste/route`;
- comunicação via `PoseStamped`;
- funcionamento de `/enable_control`;
- teste usando dados mockados de Perception;
- gravação e reprodução com `ros2 bag`.

Ainda falta validar no carro:

- atualização do `sbg_ros2_driver`;
- `/imu/odometry` real;
- TF `odom -> base_link`;
- Cartographer com TF `map -> odom`;
- integração completa da árvore TF;
- Perception real;
- Controle real;
- ativação de `ENABLE_LAP_CLOSURE`.

---

# 24. Arquitetura final esperada

```text
                   +------------------+
                   |    Perception    |
                   +------------------+
                            |
                            | /lidar_node/cones
                            v
                   +------------------+
                   |     CarNode      |
                   +------------------+
                            |
                            v
                   +------------------+
                   |   PathPlanner    |
                   +------------------+
                            |
                            | /teste/route
                            v
                   +------------------+
                   |     Controle     |
                   +------------------+


SBG INS
   |
   | /imu/odometry
   |
   +------ TF odom -> base_link


Cartographer / run.sh
   |
   +------ TF map -> odom


Árvore TF final:

map
 |
 v
odom
 |
 v
base_link
```

O objetivo da arquitetura atual é manter o `CarNode` desacoplado dos sensores específicos.

Perception fornece cones.

INS fornece odometria.

Cartographer fornece localização global / mapa.

Controle recebe a rota.

O `CarNode` apenas integra essas informações e executa o planejamento.
