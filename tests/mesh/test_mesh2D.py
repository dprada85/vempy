import pytest
import numpy as np
from vempy.mesh import Mesh2D


class TestMesh2D(object):
    """A container for all the tests on the Mesh2D class."""

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

    def test_name_setter(self):
        """Test the setter method of the name attribute."""
        xy = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        mesh = Mesh2D(xy=xy, regn_vrtx=[[0, 1, 2, 3]], name="Default name")
        with pytest.raises(ValueError):
            mesh.name = []
        with pytest.raises(ValueError):
            mesh.name = np.array(["Mesh 1"])
        with pytest.raises(ValueError):
            mesh.name = ["Mesh 1"]
        with pytest.raises(ValueError):
            mesh.name = "Mesh 1",
        mesh.name = "Mesh 1"
        assert mesh.name == "Mesh 1"

    def test_vrtx_coords_setter(self):
        """Test the setter method of the vertex coordinates."""
        with pytest.raises(ValueError):
            Mesh2D(xy=[[0, 0], [0, 1], [1, 1], [0, 1]],
                   regn_vrtx=[0, 1, 2, 3])
        with pytest.raises(IndexError):
            xy = np.array([0., 0., 0., 1., 1., 1., 0., 1.])
            Mesh2D(xy=xy, regn_vrtx=[0, 1, 2, 3])
        with pytest.raises(IndexError):
            xy = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
            mesh = Mesh2D(xy=xy, regn_vrtx=[[0, 1, 2, 3]])
            mesh.vrtx_coords = np.random.random((5, 2))
        xy = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        mesh = Mesh2D(xy=xy, regn_vrtx=[[0, 1, 2, 3]])
        xy[0, 0] = 1.0
        assert mesh.vrtx_coords[0, 0] != 1.0

    def test_bbox(self):
        """Test calculation of the bounding box."""
        xy = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        mesh = Mesh2D(xy=xy, regn_vrtx=[[0, 1, 2, 3]])
        assert mesh.bbox() == (0.0, 0.0, 1.0, 1.0)

    def test_face_measure(self):
        """Test calculation of edge length."""
        xy = np.array([[-0.25, -0.25], [0.25, -0.25],
                       [0.25, 0.25], [-0.25, 0.25]])
        mesh = Mesh2D(xy=xy, regn_vrtx=[[0, 1, 2, 3]])
        assert mesh.face_measure().size == 4 and \
            np.allclose(mesh.face_measure(), 0.5)
        assert mesh.face_measure(1) == pytest.approx(0.5)

    def test_face_nor_tng(self):
        """Test calculation of face unit normal and tangent."""
        xy = np.array([[-0.25, -0.25], [0.25, -0.25],
                       [0.25, 0.25], [-0.25, 0.25]])
        mesh = Mesh2D(xy=xy, regn_vrtx=[[0, 1, 2, 3]])
        assert np.allclose(mesh.face_nor(0), [0., -1.]) and \
            np.allclose(mesh.face_nor(1), [-1., 0.]) and \
            np.allclose(mesh.face_nor(2), [1., 0.]) and \
            np.allclose(mesh.face_nor(3), [0., 1.])
        assert np.allclose(mesh.face_tng(0), [1., 0.]) and \
            np.allclose(mesh.face_tng(1), [0., -1.]) and \
            np.allclose(mesh.face_tng(2), [0., 1.]) and \
            np.allclose(mesh.face_tng(3), [-1., 0.])

    def test_regn_area(self):
        """Test calculation of region area."""
        xy = np.array([[-0.25, -0.25], [0.25, -0.25],
                       [0.25, 0.25], [-0.25, 0.25]])
        mesh = Mesh2D(xy=xy, regn_vrtx=[[0, 1, 2, 3]])
        assert mesh.regn_area(0) == pytest.approx(0.25)

    def test_regn_centroid(self):
        """Test calculation of region centroid."""
        xy = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.5], [0.0, 0.5]])
        mesh = Mesh2D(xy=xy, regn_vrtx=[[0, 1, 2, 3]])
        assert np.allclose(mesh.regn_centroid(0), (0.5, 0.25))
        mesh.vrtx_coords = \
            np.array([[-0.5, -0.25], [1.5, -0.25], [1.5, 0.75], [-0.5, 0.75]])
        assert np.allclose(mesh.regn_centroid(0), (0.5, 0.25))

    def test_hmin_hmax_hsqr(self):
        """Test calculation of mesh sizes."""
        xy = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.5], [0.0, 0.5]])
        mesh = Mesh2D(xy=xy, regn_vrtx=[[0, 1, 2, 3]])
        assert mesh.hmin() == pytest.approx(0.5)
        assert mesh.hmax() == pytest.approx(np.sqrt(1.25))
        assert mesh.hsqr() == pytest.approx([np.sqrt(0.5), np.sqrt(0.5)])
