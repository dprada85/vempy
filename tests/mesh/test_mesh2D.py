import numpy as np
from vempy.mesh import Mesh2D


class TestMesh2D(object):
    def test_connectivity(self):
        """Test alternative ways of providing connectivity."""
        xy = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        msh1 = Mesh2D(xy=xy, regn_vrtx=[[0, 1, 2, 3]])
        msh2 = Mesh2D(xy=xy,
                      face_vrtx=np.array([[0, 1], [3, 0], [1, 2], [2, 3]]),
                      face_regn=np.array([[0, -1], [0, -1], [0, -1], [0, -1]]))
        msh3 = Mesh2D(xy=xy,
                      face_vrtx=np.array([[0, 1], [3, 0], [1, 2], [2, 3]]),
                      regn_face=[[0, 2, 3, 1]])
        message1 = ("regn-to-vertex did not provide same connectivity "
                    "as face-to-vertex and face-to-region")
        message2 = ("regn-to-vertex did not provide same connectivity "
                    "as face-to-vertex and region-to-face")
        message3 = ("face-to-vertex and face-to-region "
                    "did not provide same connectivity "
                    "as face-to-vertex and region-to-face")
        assert msh1 == msh2, message1
        assert msh1 == msh3, message2
        assert msh2 == msh3, message3
