from enum import Enum
from scipy.spatial import distance
from geopy.distance import great_circle

class HYDRASMETRICDISTANCE(Enum):
     COSINE_SIMILARITY = 1
     EQUALS = 2
     OVERLAP = 3
     GEO_DISTANCE = 4
     DISTANCE = 5
     
class HYDRASMETRICSCALING(Enum):
     LINEAR = 1
     LOGARITHMIC = 2
     EXPONENTIAL = 3
     

class HYDRASMETRIC:
     def __init__(self, name: str, metric: HYDRASMETRICDISTANCE, scaling: HYDRASMETRICSCALING = HYDRASMETRICSCALING.LINEAR):
         self.metric = metric
         self.scaling = scaling
         if name is None:
             self.name = f"{str(metric)}-{str(scaling)}"     
         else:
          self.name = name
     
     def get_similarity(self, a, b):
         if self.metric == HYDRASMETRICDISTANCE.EQUALS:
             return 1 if a == b and b is not None else 0
         elif self.metric == HYDRASMETRICDISTANCE.OVERLAP:
             return len(a & b) if b is not None else 0
         elif self.metric == HYDRASMETRICDISTANCE.GEO_DISTANCE:
             return min(great_circle(x, y).km for x in a for y in b) if b is not None and len(b) != 0 else 0
         elif self.metric == HYDRASMETRICDISTANCE.COSINE_SIMILARITY:
             return 1 - distance.cosine(a, b)
         elif self.metric == HYDRASMETRICDISTANCE.DISTANCE:
             return abs(a - b) if b is not None else 0
         else:
             print(f"Error: No metric implemented for {str(self.metric)}. ")
             exit(1)
