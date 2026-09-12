import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/demarco/Mapping2026CARRO/Pipeline/install/autonomous_vehicle_slam'
