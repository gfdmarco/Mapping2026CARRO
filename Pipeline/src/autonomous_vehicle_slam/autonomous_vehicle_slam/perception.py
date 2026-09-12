import cv2
import numpy as np
from ultralytics import YOLO

FX=448.14
CX=640.0
CY=360.0

CONF_THRESHOLD = 0.85
ANGLE_DEGREE = 2.5
MODEL_PATH = "/home/felipe_capovilla/Documents/E-Racing/Perception/Modelos/16_01.pt"

class Perception:
    def __init__(self,client):
    
        try:
            self.model = YOLO(MODEL_PATH)
        except Exception as e:
            raise RuntimeError(f"YOLO failed to start.\n{e}")

        self.client = client
    
    def lidarMatch(self,points,u,v):
      
        ray2d = np.array([1.0, (u - CX) / FX])
        ray2d /= np.linalg.norm(ray2d)

       
        points2d = points[:, :2] 
        norms = np.linalg.norm(points2d, axis=1)
        
        validMask = norms > 0.1
        validPoints = points[validMask]
        validPoints2D = points2d[validMask]
        validNorms = norms[validMask]

        if validPoints.size == 0: return None

        dirs2D = validPoints2D / validNorms[:, None]
        cossAngles = dirs2D @ ray2d
        mask = cossAngles > np.cos(np.deg2rad(ANGLE_DEGREE))
        candidates = validPoints[mask]

        if candidates.size == 0: return None

        return candidates[np.argmin(np.linalg.norm(candidates[:, :2], axis=1))]
    
    def detectCones(self,img):
        image = cv2.imdecode(np.frombuffer(img.image_data_uint8, dtype=np.uint8), cv2.IMREAD_COLOR)

        lidarData = self.client.getLidarData(lidar_name='Lidar1')
     
        points = np.array(lidarData.point_cloud, dtype=np.float32).reshape((-1, 3))

        if image is None: return np.array([])

        results = self.model(image, verbose=False, conf=CONF_THRESHOLD)
        rawDetection = []

        for box in results[0].boxes:
            u, v = float(box.xywh[0][0]), float(box.xywh[0][1])
            classId = int(box.cls[0])
            hit = self.lidarMatch(points, u, v)

            if hit is not None:
    
                x_frente = float(hit[0])

                y_lateral = (u - CX) * x_frente / FX

                if 0.5 < x_frente < 35.0:
                    rawDetection.append([y_lateral, x_frente, classId])
        
        finalCones = []
        for cone in rawDetection:
            
            if not any(np.linalg.norm(np.array(cone[:2]) - np.array(f[:2])) < 1.2 for f in finalCones):
                finalCones.append(cone)

        return np.array(finalCones) #[lateral,profundidade,id]