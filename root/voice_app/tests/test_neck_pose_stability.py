import unittest
import sys
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from neck_v1_cv import estimate_head_pitch


class PoseStabilityTests(unittest.TestCase):
    def test_frontal_pose_does_not_flip_with_pixel_perturbations(self):
        model = np.array([[0,0,0],[0,330,-65],[-225,-170,-135],
                          [225,-170,-135],[-150,150,-125],[150,150,-125]],float)
        for width in (480,960):
            camera=np.array([[width,0,width/2],[0,width,width*9/32],[0,0,1]],float)
            projected,_=cv2.projectPoints(model,np.zeros(3),np.array([0.,0.,1800.]),camera,np.zeros(4))
            for noise in (0,.25,-.25):
                points=np.zeros((468,2))
                points[[1,152,33,263,61,291]]=projected[:,0]+np.array([
                    [noise,0],[0,noise],[-noise,noise],[noise,-noise],[0,-noise],[noise,0]])
                scale=np.linalg.norm(points[33]-points[263])
                pitch=estimate_head_pitch(points,width,int(width*9/16),scale)
                self.assertIsNotNone(pitch)
                self.assertLess(abs(pitch),2)

    def test_incoherent_landmarks_are_unavailable(self):
        points=np.zeros((468,2))
        points[[1,152,33,263,61,291]]=[[200,100],[20,5],[350,250],[40,20],[450,10],[10,260]]
        self.assertIsNone(estimate_head_pitch(points,480,270,100))


if __name__ == '__main__':
    unittest.main()
