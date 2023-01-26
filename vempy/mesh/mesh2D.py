import numpy as np
from scipy.spatial.distance import pdist


class Mesh2D:
    """A class for polygonal meshes.

    Parameters
    ----------
    nV : int
        Number of vertices.
    nF : int
        Number of faces (edges).
    nR : int
        Number of regions.
    name : str
        Mesh name.
    vrtx_coords : numpy.ndarray
        Vertex coordinates.
    _regn_face : list of lists, private
        Region to face connectivity, accessible by class methods.
    _face_vrtx : numpy.ndarray, private
        Face to vertex connectivity, accessible by class methods.
    _vrtx_face : list of lists, private
        Vertex to face connectivity, accessible by class methods.
    _face_regn : numpy.ndarray, private
        Face to region connectivity, accessible by class methods.
    bnd_vrtx : set
        Set of boundary vertices.
    bnd_face : set
        Set of boundary faces.
    bnd_regn : set
        Set of boundary regions.
    fV : numpy.ndarray
        Vertex flags.
    fF : numpy.ndarray
        Face flags.
    fR : numpy.ndarray
        Region flags.

    Examples
    --------
    First mode, use the class constructor:

    >>> # Create a mesh with a single element given by the unit square
    >>> import numpy as np
    >>> from vempy.mesh import Mesh2D
    >>> coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    >>> mesh = Mesh2D(xy=coords, regn_vrtx=[[0,1,2,3]])
    >>> print(mesh)

    Mesh2D:

    Number of vertices: 4
    Number of faces: 4
    Number of regions: 1

    Vertex coordinates:
    [[0. 0.]
     [1. 0.]
     [1. 1.]
     [0. 1.]]

    Face-to-vertex connectivity:
    [[0 1]
     [3 0]
     [1 2]
     [2 3]]

    Face-to-region connectivity:
    [[ 0 -1]
     [ 0 -1]
     [ 0 -1]
     [ 0 -1]]

    Name:

    """

    __UNSET = -1  # Arbitrary constant to unset class variables

    def __init__(self, xy,
                 regn_vrtx=None,
                 face_vrtx=None, face_regn=None,
                 regn_face=None,
                 fV=None, fF=None, fR=None,
                 name=None):

        # _build_the_mesh takes care of building all the following attributes
        #
        # self._nV, number of vertices
        # self._nF, number of edges
        # self._nR, number of regions
        # self._vrtx_coords, vertex coordinates
        # self._regn_face, region to face connectivity
        # self._face_vrtx, face to vertex connectivity
        # self._vrtx_face, vertex to face connectivity
        # self._face_regn, face to region connectivity
        # self._fV, vertex flags
        # self._fF, face flags
        # self._fR, region flags
        # self._bnd_vrtx, boundary vertices
        # self._bnd_face, boundary faces
        # self._bnd_regn, boundary regions
        self._build_the_mesh(xy,
                             regn_vrtx=regn_vrtx,
                             face_vrtx=face_vrtx, face_regn=face_regn,
                             regn_face=regn_face,
                             fV=fV, fF=fF, fR=fR)

        if name is not None:
            self.name = name
        else:
            self._name = ""

    def __str__(self):
        """String representation."""
        vrtx_coords_str = \
            "\n".join(
                [str(iV) + " " + str(self._vrtx_coords[iV, :]).strip("[]")
                 for iV in range(self._nV)]
            )
        regn_regn_str = \
            "\n".join(
                [str(iR) + " " + " ".join([str(r) for r in self.regn_regn(iR)])
                 for iR in range(self._nR)]
            )
        regn_face_str = \
            "\n".join(
                [str(iR) + " " + " ".join([str(r) for r in self.regn_face(iR)])
                 for iR in range(self._nR)]
            )
        regn_vrtx_str = \
            "\n".join(
                [str(iR) + " " + " ".join([str(r) for r in self.regn_vrtx(iR)])
                 for iR in range(self._nR)]
            )
        face_regn_str = \
            "\n".join(
                [str(iF) + " " + " ".join([str(r) for r in self.face_regn(iF)])
                 for iF in range(self._nF)]
            )
        face_face_str = \
            "\n".join(
                [str(iF) + " " + " ".join([str(r) for r in self.face_face(iF)])
                 for iF in range(self._nF)]
            )
        face_vrtx_str = \
            "\n".join(
                [str(iF) + " " + " ".join([str(r) for r in self.face_vrtx(iF)])
                 for iF in range(self._nF)]
            )
        vrtx_regn_str = \
            "\n".join(
                [str(iV) + " " + " ".join([str(r) for r in self.vrtx_regn(iV)])
                 for iV in range(self._nV)]
            )
        vrtx_face_str = \
            "\n".join(
                [str(iV) + " " + " ".join([str(r) for r in self.vrtx_face(iV)])
                 for iV in range(self._nV)]
            )
        vrtx_vrtx_str = \
            "\n".join(
                [str(iV) + " " + " ".join([str(r) for r in self.vrtx_vrtx(iV)])
                 for iV in range(self._nV)]
            )
        Mesh2D_str = """
Mesh2D:

Number of vertices: {nV}
Number of faces: {nF}
Number of regions: {nR}

Vertex coordinates:
{xy}

Region-to-region:
{regn_regn}

Region-to-face:
{regn_face}

Region-to-vertex:
{regn_vrtx}

Face-to-region:
{face_regn}

Face-to-face:
{face_face}

Face-to-vertex:
{face_vrtx}

Vertex-to-region:
{vrtx_regn}

Vertex-to-face:
{vrtx_face}

Vertex-to-vertex:
{vrtx_vrtx}

Name: {name}
        """.format(nV=self._nV, nF=self._nF, nR=self._nR, xy=vrtx_coords_str,
                   regn_regn=regn_regn_str, regn_face=regn_face_str,
                   regn_vrtx=regn_vrtx_str, face_regn=face_regn_str,
                   face_face=face_face_str, face_vrtx=face_vrtx_str,
                   vrtx_regn=vrtx_regn_str, vrtx_face=vrtx_face_str,
                   vrtx_vrtx=vrtx_vrtx_str, name=self._name)
        return Mesh2D_str

    def __eq__(self, other):
        """Custom comparison."""
        return (self.nV == other.nV) and (self.nF == other.nF) and \
            (self.nR == other.nR) and \
            np.all(self.vrtx_coords == other.vrtx_coords) and \
            (self._regn_face == other._regn_face) and \
            np.all(self._face_vrtx == other._face_vrtx) and \
            (self._vrtx_face == other._vrtx_face) and \
            np.all(self._face_regn == other._face_regn) and \
            np.all(self.fV == other.fV) and \
            np.all(self.fF == other.fF) and \
            np.all(self.fR == other.fR) and \
            (self.bnd_vrtx == other.bnd_vrtx) and \
            (self.bnd_face == other.bnd_face) and \
            (self.bnd_regn == other.bnd_regn) and \
            (self.name == other.name) and \
            str(self) == str(other)

    @property
    def nV(self):
        """Number of vertices."""
        return self._nV

    @property
    def nF(self):
        """Number of faces (edges)."""
        return self._nF

    @property
    def nR(self):
        """Number of regions."""
        return self._nR

    @property
    def name(self):
        """Mesh name."""
        return self._name

    @name.setter
    def name(self, new_name):
        """Set the mesh name.

        Parameters
        ----------
        new_name : str
            New mesh name.

        Returns
        -------

        Raises
        ------
        ValueError
            If new_name is not a string.

        Examples
        --------
        >>> import numpy as np
        >>> from vempy.mesh import Mesh2D
        >>> coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        >>> mesh = Mesh2D(xy=coords, regn_vrtx=[[0,1,2,3]])
        >>> mesh.name = "Mesh 1"
        >>> print(mesh.name)
        Mesh 1
        """
        if type(new_name) is not str:
            raise ValueError("Mesh name must be a string")
        self._name = new_name

    @property
    def vrtx_coords(self):
        """Array of vertex coordinates."""
        return self._vrtx_coords

    @vrtx_coords.setter
    def vrtx_coords(self, new_coords):
        """Set the entire array of vertex coordinates.

        Parameters
        ----------
        new_coords : numpy.ndarray
            Array of new vertex coordinates.

        Returns
        -------

        Examples
        --------
        TBD
        """
        if type(new_coords) is not np.ndarray:
            raise ValueError("Vertex coordinates must be a numpy array")
        elif new_coords.ndim != 2:
            raise IndexError("Vertex coordinates must be a 2D numpy array")
        elif self._nV != 0 and new_coords.shape[0] != self._nV:
            msg = "Vertex coordinates must be a ({}, 2) numpy array"
            raise IndexError(msg.format(self._nV))
        self._vrtx_coords = new_coords.copy()
        self._nV = self._vrtx_coords.shape[0]

    def bbox(self):
        """Evaluate mesh bounding box.

        Parameters
        ----------

        Returns
        -------
        bbox : tuple
            Mesh bounding box: (xmin, ymin, xmax, ymax).

        Examples
        --------
        >>> xy = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        >>> mesh = Mesh2D(xy=xy, regn_vrtx=[[0, 1, 2, 3]])
        >>> print(mesh.bbox())
        (0.0, 0.0, 1.0, 1.0)
        """
        bbox = tuple(np.amin(self._vrtx_coords, axis=0).tolist() +
                     np.amax(self._vrtx_coords, axis=0).tolist())
        return bbox

    def face_measure(self, iF=None):
        """Compute the measure of mesh faces.

        Parameters
        ----------
        iF : int, optional
            Face index. If not specified, the measures of all the faces
            are computed. (Default value = None)

        Returns
        -------
        measure : numpy.float64 or numpy.ndarray, according to the value of iF
            Face measure(s).

        Examples
        --------
        >>> xy = np.array([[-0.2, -0.1], [0.2, -0.1], [0.2, 0.1], [-0.5, 0.1]])
        >>> mesh = Mesh2D(xy=xy, regn_vrtx=[[0, 1, 2, 3]])
        >>> print(mesh.face_measure())
        [0.4        0.36055513 0.2        0.7       ]
        >>> print(mesh.face_measure(2))
        0.2
        """
        if iF is None:
            delta = self._vrtx_coords[self._face_vrtx[:, 0], :] - \
                self._vrtx_coords[self._face_vrtx[:, 1], :]
            measure = np.linalg.norm(delta, axis=1)
        else:
            delta = self._vrtx_coords[self._face_vrtx[iF, 0], :] - \
                self._vrtx_coords[self._face_vrtx[iF, 1], :]
            measure = np.linalg.norm(delta)
        return measure

    def face_nor(self, iF):
        """Get the unit normal to a face.

        Parameters
        ----------
        iF : int
            Face index.

        Returns
        -------
        nor : numpy.ndarray
            Unit normal to face iF.

        Examples
        --------
        >>> xy = numpy.array([[-0.25, -0.25], [0.25, -0.25],
                              [0.25, 0.25], [-0.25, 0.25]])
        >>> mesh = Mesh2D(xy=xy, regn_vrtx=[[0, 1, 2, 3]])
        >>> print(mesh.face_nor(0))
        [0. -1.]
        """
        iV0 = self._face_vrtx[iF, 0]
        iV1 = self._face_vrtx[iF, 1]
        meas_F = self.face_measure(iF)
        nx = (self._vrtx_coords[iV1, 1] - self._vrtx_coords[iV0, 1]) / meas_F
        ny = (self._vrtx_coords[iV0, 0] - self._vrtx_coords[iV1, 0]) / meas_F
        nor = np.array([nx, ny])
        return nor

    def face_tng(self, iF):
        """Get the unit tangent to a face.

        Parameters
        ----------
        iF : int
            Face index.

        Returns
        -------
        tng : numpy.ndarray
            Unit tangent to face iF.

        Examples
        --------
        >>> xy = numpy.array([[-0.25, -0.25], [0.25, -0.25],
                              [0.25, 0.25], [-0.25, 0.25]])
        >>> mesh = Mesh2D(xy=xy, regn_vrtx=[[0, 1, 2, 3]])
        >>> print(mesh.face_tng(0))
        [1. 0.]
        """
        iV0 = self._face_vrtx[iF, 0]
        iV1 = self._face_vrtx[iF, 1]
        meas_F = self.face_measure(iF)
        tx = (self._vrtx_coords[iV1, 0] - self._vrtx_coords[iV0, 0]) / meas_F
        ty = (self._vrtx_coords[iV1, 1] - self._vrtx_coords[iV0, 1]) / meas_F
        tng = np.array([tx, ty])
        return tng

    def regn_area(self, iR):
        """Get the area of a region.

        Parameters
        ----------
        iR : int
            Region index.

        Returns
        -------
        regn_area : numpy.float64
            Region area.

        Examples
        --------
        >>> xy = numpy.array([[-0.25, -0.25], [0.25, -0.25],
                              [0.25, 0.25], [-0.25, 0.25]])
        >>> mesh = Mesh2D(xy=xy, regn_vrtx=[[0, 1, 2, 3]])
        >>> print(mesh.regn_area(0))
        0.25
        """
        regn_area = \
            self._lasserre(lambda iF: self.face_measure(iF), 0, iR)
        return regn_area

    def regn_centroid(self, iR):
        """Get the centroid of a region.

        Parameters
        ----------
        iR : int
            Region index.

        Returns
        -------
        regn_centroid : tuple
            Region centroid.

        Examples
        --------
        >>> xy = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.5], [0.0, 0.5]])
        >>> mesh = Mesh2D(xy=xy, regn_vrtx=[[0, 1, 2, 3]])
        >>> print(mesh.regn_centroid(0))
        (0.5, 0.25)
        """
        regn_area = self.regn_area(iR)
        xr = \
            self._lasserre(lambda iF: self.face_measure(iF) *
                           self._vrtx_coords[self._face_vrtx[iF, :], 0].mean(),
                           1, iR)
        yr = \
            self._lasserre(lambda iF: self.face_measure(iF) *
                           self._vrtx_coords[self._face_vrtx[iF, :], 1].mean(),
                           1, iR)
        regn_centroid = (xr/regn_area, yr/regn_area)
        return regn_centroid

    def hmin(self):
        """Compute the minimum distance over polygons.

        Parameters
        ----------

        Returns
        -------
        hmin : numpy.float64
            Minimum distance between any pair of vertices in the mesh.

        Examples
        --------
        >>> xy = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 1.0], [0.0, 1.0]])
        >>> mesh = Mesh2D(xy=xy, regn_vrtx=[[0, 1, 2, 3]])
        >>> print(mesh.hmin())
        1.0
        """
        hmin = np.amin(self.face_measure())
        return hmin

    def hmax(self):
        """Compute the maximum distance of polygons.

        Parameters
        ----------

        Returns
        -------
        hmax : numpy.float64
            Maximum distance between points across polygons.

        Examples
        --------
        >>> xy = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 1.0], [0.0, 1.0]])
        >>> mesh = Mesh2D(xy=xy, regn_vrtx=[[0, 1, 2, 3]])
        >>> print(mesh.hmax())
        4.123105625617661
        """
        diameters = [np.amax(pdist(self._vrtx_coords[self.regn_vrtx(iR), :]))
                     for iR in range(self._nR)]
        hmax = np.amax(diameters)
        return hmax

    def hsqr(self):
        """Estimate mesh size as maximum or average of
        square root of polygon area.

        Parameters
        ----------

        Returns
        -------
        hsqrt : tuple
            Tuple containing the maximum and the average of polygon area.

        Examples
        --------
        >>> xy = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 1.0], [0.0, 1.0]])
        >>> mesh = Mesh2D(xy=xy, regn_vrtx=[[0, 1, 2, 3]])
        >>> print(mesh.hsqr())
        (2.0, 2.0)
        """
        areas_sq = np.sqrt([self.regn_area(iR) for iR in range(self._nR)])
        hsqrt = (np.amax(areas_sq), np.mean(areas_sq))
        return hsqrt

    def regn_regn(self, iR):
        """Find the neighboring regions of a given region.

        Parameters
        ----------
        iR : int
            Region index.

        Returns
        -------
        regn_regn : tuple
            Tuple of neighboring regions of region iR.

        Examples
        --------
        TBD
        """
        regn_regn = \
            tuple(self._face_regn[iF, 0] if self._face_regn[iF, 1] == iR else
                  self._face_regn[iF, 1] for iF in self._regn_face[iR])
        return regn_regn

    def regn_face(self, iR):
        """Returns the faces of a given region in counter-clockwise order.

        Parameters
        ----------
        iR : int
            Region index.

        Returns
        -------
        regn_face : tuple
            Faces of region iR in counter-clockwise order.

        Examples
        --------
        TBD
        """
        regn_face = tuple(self._regn_face[iR])
        return regn_face

    def orient_regn_face(self, iR, ilF):
        """Returns True if face (edge) is pointing out of region.
        For a face, pointing out of a region means that the region
        is at the left hand side of the face when moving from the
        first to the second vertex of the face.

        Parameters
        ----------
        iR : int
            Region index.
        ilF : int
            Local face index.

        Returns
        -------
        out : bool
            True if local face ilF is pointing out of region iR,
            False otherwise.

        Examples
        --------
        TBD
        """
        iF = self._regn_face[iR][ilF]
        out = iR == self._face_regn[iF, 0]
        return out

    def regn_vrtx(self, iR):
        """Returns the vertices of a given region in counter-clockwise order.

        Parameters
        ----------
        iR : int
            Region index.

        Returns
        -------
        regn_vrtx : tuple
            Tuple of vertices of region iR in counter-clockwise order.

        Examples
        --------
        TBD
        """
        regn_vrtx = \
            tuple(self._face_vrtx[iF, 0] if iR == self._face_regn[iF, 0] else
                  self._face_vrtx[iF, 1] for iF in self._regn_face[iR])
        return regn_vrtx

    def face_regn(self, iF):
        """Returns the regions connected to a given face.

        Parameters
        ----------
        iF : int
            Face index.

        Returns
        -------
        face_regn : tuple
            Regions connected to face iF.

        Examples
        --------
        TBD
        """
        face_regn = (self._face_regn[iF, 0], self._face_regn[iF, 1])
        return face_regn

    def face_face(self, iF):
        """Returns the faces connected to a given face.

        Parameters
        ----------
        iF : int
            Face index.

        Returns
        -------
        face_face : tuple
            Set of faces connected to face iF (face iF excluded).

        Examples
        --------
        TBD
        """
        set0 = set(self._vrtx_face[self._face_vrtx[iF, 0]])
        set1 = set(self._vrtx_face[self._face_vrtx[iF, 1]])
        face_face = tuple(set0.symmetric_difference(set1))
        return face_face

    def face_vrtx(self, iF):
        """Returns the endpoints of a given face.

        Parameters
        ----------
        iF : int
            Face index.

        Returns
        -------
        face_vrtx : tuple
            Endpoints of face iF.

        Examples
        --------
        TBD
        """
        face_vrtx = (self._face_vrtx[iF, 0], self._face_vrtx[iF, 1])
        return face_vrtx

    def vrtx_regn(self, iV):
        """Returns the regions connected to a given vertex.

        Parameters
        ----------
        iV : int
            Vertex index.

        Returns
        -------
        vrtx_regn : tuple
            Regions sharing iV as vertex.

        Examples
        --------
        TBD
        """
        redundant_regions = self._face_regn[self._vrtx_face[iV], :].flatten()
        vrtx_regn = tuple(set(redundant_regions).difference({Mesh2D.__UNSET}))
        return vrtx_regn

    def vrtx_face(self, iV):
        """Returns the faces connected to a given vertex.

        Parameters
        ----------
        iV : int
            Vertex index.

        Returns
        -------
        vrtx_face : tuple
            Faces connected to vertex iV.

        Examples
        --------
        TBD
        """
        vrtx_face = tuple(self._vrtx_face[iV])
        return vrtx_face

    def vrtx_vrtx(self, iV):
        """Returns the vertices connected to a given vertex.

        Parameters
        ----------
        iV : int
            Vertex index.

        Returns
        -------
        vrtx_vrtx : tuple
            Vertices connected to vertex iV.

        Examples
        --------
        TBD
        """
        redundant_vertices = self._face_vrtx[self._vrtx_face[iV], :].flatten()
        vrtx_vrtx = tuple(set(redundant_vertices).difference({iV}))
        return vrtx_vrtx

    @property
    def bnd_vrtx(self):
        """Set of boundary vertices."""
        return self._bnd_vrtx

    @property
    def bnd_face(self):
        """Set of boundary faces."""
        return self._bnd_face

    @property
    def bnd_regn(self):
        """Set of boundary regions."""
        return self._bnd_regn

    @property
    def fV(self):
        """Vertex flags."""
        return self._fV

    @fV.setter
    def fV(self, vflags):
        """Set vertex flags.

        Parameters
        ----------
        vflags : numpy.ndarray of int
            Array of vertex flags.

        Returns
        -------

        Examples
        --------
        TBD
        """
        if type(vflags) is not np.ndarray:
            raise ValueError("Vertex flags must be a numpy array")
        elif vflags.ndim != 1 or vflags.shape[0] != self._nV:
            msg = "New vertex flags must be a {}-long 1D numpy array"
            raise IndexError(msg.format(self._nV))
        self._fV = vflags.copy()

    @property
    def fF(self):
        """Face flags."""
        return self._fF

    @fF.setter
    def fF(self, fflags):
        """Set face flags.

        Parameters
        ----------
        fflags : numpy.ndarray of int
            Array of face flags.

        Returns
        -------

        Examples
        --------
        TBD
        """
        if type(fflags) is not np.ndarray:
            raise ValueError("Face flags must be a numpy array")
        elif fflags.ndim != 1 or fflags.shape[0] != self._nF:
            msg = "New face flags must be a {}-long 1D numpy array"
            raise IndexError(msg.format(self._nF))
        self._fF = fflags.copy()

    @property
    def fR(self):
        """Region flags."""
        return self._fR

    @fR.setter
    def fR(self, rflags):
        """Set region flags.

        Parameters
        ----------
        rflags : numpy.ndarray of int
            Array of region flags.

        Returns
        -------

        Examples
        --------
        TBD
        """
        if type(rflags) is not np.ndarray:
            raise ValueError("Region flags must be a numpy array")
        elif rflags.ndim != 1 or rflags.shape[0] != self._nR:
            msg = "New region flags must be a {}-long 1D numpy array"
            raise IndexError(msg.format(self._nR))
        self._fR = rflags.copy()

    def _lasserre(self, func, deg, iR):
        """Integrate a positively homogeneous function over a region by
        using Lasserre method.

        Parameters
        ----------
        func : function
            A Python function that takes the (global) index of a mesh face
            as input and returns the integral over such a face of the
            homogeneous function.
        deg : int
            Degree of the homogeneous function.
        iR : int
            Region index.

        Returns
        -------
        result : numpy.float64
            Integral of the homogeneous function over region iR.

        Notes
        -----
        We use Lasserre's method [1]_.

        References
        ----------
        .. [1] Chin, E.B., Lasserre, J.B. & Sukumar, N.
           "Numerical integration of homogeneous functions on convex
           and nonconvex polygons and polyhedra".
           Comput Mech 56, 967–981 (2015).
        """
        lasserre = \
            [(2.0 * (iR == self._face_regn[iF, 0]) - 1.0) *
             np.dot(self.face_nor(iF),
                    self._vrtx_coords[self._face_vrtx[iF, 0], :]) *
             func(iF) for iF in self._regn_face[iR]]
        result = 1.0 / (2.0 + np.float64(deg)) * sum(lasserre)
        return result

    def _build_fv_rf_from_rv(self, regn_vrtx):
        """Build self._face_vrtx and self._regn_face from regn to vertex data.

        Parameters
        ----------
        regn_vrtx : list of lists
            Region to vertex connectivity.

        Returns
        -------

        Examples
        --------
        TBD
        """
        elist = np.concatenate([np.stack((vlist, np.roll(vlist, -1)), axis=-1)
                                for vlist in regn_vrtx])
        sorted_elist = np.sort(elist)
        _, rind, rinv = np.unique(sorted_elist, return_index=True,
                                  return_inverse=True, axis=0)
        self._face_vrtx = elist[rind, :]
        ties = np.cumsum([len(ar) for ar in regn_vrtx])[:-1]
        self._regn_face = [ar.tolist() for ar in np.split(rinv, ties)]

    def _invert_int_mapping(self, list_in, list_out_size, unset=None):
        """Invert a mapping between integers.

        Parameters
        ----------
        list_in : list or numpy.ndarray
            Mapping to invert. It can be a list of lists,
            or a 2D numpy array.
        list_out_size : int
            Size of the list providing the inverse mapping.

        Returns
        -------
        list_out : list
            Inverse mapping.

        Examples
        --------
        >>> Mesh2D()._invert_int_mapping([[0,1],[0,2],[1,2]], 3)
        [[0, 1], [0, 2], [1, 2]]
        """
        list_out = [[] for j in range(list_out_size)]
        for i in range(len(list_in)):
            for j in list_in[i]:
                if j != unset:
                    list_out[j].append(i)
        return list_out

    def _build_rf_from_fv_fr(self, face_vrtx, face_regn):
        """Build self._regn_face from face to region and face to vertex data.

        Parameters
        ----------
        face_vrtx : numpy.ndarray
            Face to vertex connectivity.
        face_regn : numpy.ndarray
            Face to region connectivity.

        Returns
        -------

        Examples
        --------
        TBD
        """
        # Number of regions
        nR = np.amax(face_regn) + 1

        self._regn_face = \
            self._invert_int_mapping(face_regn, nR, unset=Mesh2D.__UNSET)

        for iR, flist in enumerate(self._regn_face):
            iF = flist.pop(0)
            new_flist = [iF]
            if iR == face_regn[iF, 0]:
                last_iV = face_vrtx[iF, 1]
            else:
                last_iV = face_vrtx[iF, 0]
            while len(flist) > 0:
                for iF in flist:
                    if face_vrtx[iF, 0] == last_iV:
                        new_flist.append(iF)
                        flist.remove(iF)
                        last_iV = face_vrtx[iF, 1]
                        break
                    if face_vrtx[iF, 1] == last_iV:
                        new_flist.append(iF)
                        flist.remove(iF)
                        last_iV = face_vrtx[iF, 0]
                        break

            self._regn_face[iR] = new_flist

    def _build_boundary_sets(self):
        """Build sets of boundary elements."""
        bnd_faces, = np.where(self._face_regn[:, 1] == Mesh2D.__UNSET)
        self._bnd_vrtx = set(self._face_vrtx[bnd_faces, :].flatten())
        self._bnd_face = set(bnd_faces)
        self._bnd_regn = set(self._face_regn[bnd_faces, 0])

    def _build_the_mesh(self, xy,
                        regn_vrtx=None,
                        face_vrtx=None, face_regn=None,
                        regn_face=None,
                        fV=None, fF=None, fR=None):
        """Build the mesh. Mesh connectivity can be provided in three
        different ways:
        - region-to-vertex
        - face-to-vertex and face-to-region
        - region-to-face and face-to-vertex

        Parameters
        ----------
        xy : numpy.ndarray
            Vertex coordinates.
        regn_vrtx : list of lists
            Region to vertex connectivity.

        Returns
        -------

        Examples
        --------
        TBD
        """
        self._nV = 0
        self._nF = 0
        self._nR = 0

        self.vrtx_coords = xy  # Sets also self._nV

        # Build top -> bottom connectivity (region-to-face and face-to-vertex)
        if regn_vrtx is not None:
            self._build_fv_rf_from_rv(regn_vrtx)
        elif face_vrtx is not None and face_regn is not None:
            self._build_rf_from_fv_fr(face_vrtx, face_regn)
            self._face_vrtx = face_vrtx.copy()
        elif regn_face is not None and face_vrtx is not None:
            self._regn_face = regn_face.copy()
            self._face_vrtx = face_vrtx.copy()
        else:
            msg = """
            _build_the_mesh expected connectivity data to be provided in
            one of the following three ways:
            - region-to-vertex
            - face-to-vertex and face-to-region
            - region-to-face and face-to-vertex
            """
            raise TypeError(msg)

        self._nF = self._face_vrtx.shape[0]
        self._nR = len(self._regn_face)

        # Build bottom -> top connectivity (vertex-to-face and face-to-region)
        self._vrtx_face = self._invert_int_mapping(self._face_vrtx, self._nV)
        self._face_regn = \
            np.array([
                row if len(row) == 2 else [row[0], Mesh2D.__UNSET]
                for row in self._invert_int_mapping(self._regn_face, self._nF)
            ])

        # Build flags
        if fV is not None:
            self.fV = fV
        else:
            self._fV = np.array([])
        if fF is not None:
            self.fF = fF
        else:
            self._fF = np.array([])
        if fR is not None:
            self.fR = fR
        else:
            self._fR = np.array([])

        # Build boundary sets
        self._build_boundary_sets()
