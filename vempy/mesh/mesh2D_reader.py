import numpy as np
from .mesh2D import Mesh2D


def _polygon_area(vrtx_coords):
    """Compute the (oriented) area of a polygon.

    If the vertices of a polygon are oriented in counter-clockwise fashion,
    then its oriented area is positive.

    Parameters
    ----------
    vrtx_coords : numpy.ndarray
        Coordinates of the polygon vertices.

    Returns
    -------
    regn_area : numpy.float64
        Oriented area of the polygon.

    Examples
    --------
    >>> import numpy as np
    >>> vrtx = np.array([[0., 0.], [0.5, 0.], [0.5, 0.5], [0., 0.5]])
    >>> _polygon_area(vrtx)
    0.25
    >>> vrtx = np.flip(vrtx, axis=0)
    >>> _polygon_area(vrtx)
    -0.25
    """
    Nx = np.roll(vrtx_coords[:, 1], -1, axis=0) - vrtx_coords[:, 1]
    Ny = vrtx_coords[:, 0] - np.roll(vrtx_coords[:, 0], -1, axis=0)
    regn_area = (np.dot(Nx, vrtx_coords[:, 0]) +
                 np.dot(Ny, vrtx_coords[:, 1])) / 2.0
    return regn_area


def _read_regn_data(fileobj, vrtx_coords, nR, nodelem, nattr, eleId=False,
                    offset=0, unset=-1):
    """Read region-to-vertex connectivity.

    Parameters
    ----------
    fileobj : file object
        File containing region-to-vertex connectivity.
    vrtx_coords : numpy.ndarray
        Coordinates of all the vertices of a mesh.
    nR : numpy.int64
        Number of expected regions.
    nodelem : numpy.int64
        Number of nodes per region (=0 for general polygons, >0 for polygons
        with a fixed number of nodes).
    nattr : numpy.int64
        Number of attribute per region (either 0 or 1).
    eleId : bool
        True whether each line start with the region id, False otherwise.
    offset : int
        Offset to be subtracted from vertex indices. (Default value = 0)
    unset : int
        Value to unset region flags. (Default value = -1)

    Returns
    -------
    regn_vrtx : list of lists
        Region-to-vertex connectivity.
    fR : numpy.array
        Region flags.
    """
    regn_vrtx = []
    fR = []
    startid = 1 if eleId else 0
    for iR in range(nR):
        line = fileobj.readline().split()[startid:]
        nV = int(line.pop(0)) if nodelem == 0 else nodelem
        face_vrtx = [int(v)-offset for v in line[:nV]]
        if _polygon_area(vrtx_coords[face_vrtx, :]) > 0.0:
            regn_vrtx.append(face_vrtx)
        else:
            msg = 'Region {} is not counter-clockwise oriented, ' + \
                'switching its orientation'
            print(msg.format(iR))
            face_vrtx.reverse()
            regn_vrtx.append(face_vrtx)
        if nattr == 0:
            fR.append(unset)
        elif nattr == 1:
            fR.append(int(line[nV]))
        else:
            msg = 'Expected at most one attribute, got {}, instead'
            raise ValueError(msg.format(nattr))
    fR = np.array(fR, dtype=int)
    return regn_vrtx, fR


def read_off(filename, offset=0, unset=-1):
    r"""Read an Object File Format (.off) file into Mesh2D.

    Parameters
    ----------
    filename : str
        Name of the file where the mesh is stored.
    offset : int
        Offset to be subtracted from vertex indices. (Default value = 0)
    unset : int
        Value to unset vertex and region flags. (Default value = -1)

    Returns
    -------
    Mesh2D

    Examples
    --------
    >>> file_str = "OFF\n4 1 0\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n4 0 1 2 3"
    >>> f = open('example.off', 'w')
    >>> f.write(file_str)
    >>> f.close()
    >>> from vempy.mesh import read_off
    >>> mesh = read_off('example.off')
    >>> print(mesh)

    Mesh2D:

    Number of vertices: 4
    Number of faces: 4
    Number of regions: 1

    Vertex coordinates:
    0 0. 0.
    1 1. 0.
    2 1. 1.
    3 0. 1.

    Region-to-region:
    0 -1 -1 -1 -1

    Region-to-face:
    0 0 2 3 1

    Region-to-vertex:
    0 0 1 2 3

    Face-to-region:
    0 0 -1
    1 0 -1
    2 0 -1
    3 0 -1

    Face-to-face:
    0 1 2
    1 0 3
    2 0 3
    3 1 2

    Face-to-vertex:
    0 0 1
    1 3 0
    2 1 2
    3 2 3

    Vertex-to-region:
    0 0
    1 0
    2 0
    3 0

    Vertex-to-face:
    0 0 1
    1 0 2
    2 2 3
    3 1 3

    Vertex-to-vertex:
    0 1 3
    1 0 2
    2 1 3
    3 0 2

    Name:

    """
    with open(filename, 'r') as file:
        if file.readline().strip() != 'OFF':
            raise TypeError('Unexpected header in input file ' + filename)
        nV, nR, _ = np.int64(file.readline().split())
        # Read vertex coordinates
        vrtx_coords = np.loadtxt(file, usecols=[0, 1], max_rows=nV)
        if vrtx_coords.shape[0] != nV:
            msg = 'Expected {} vertices, got {}, instead'
            raise IndexError(msg.format(nV, vrtx_coords.shape[0]))
        # Read region-to-vertex connectivity
        regn_vrtx, fR = \
            _read_regn_data(file, vrtx_coords, nR, 0, 0, False, offset, unset)
    fV = np.full(nV, fill_value=unset, dtype=int)
    return Mesh2D(vrtx_coords, regn_vrtx=regn_vrtx, fV=fV, fR=fR)


def read_node_ele(filename, offset=0, unset=-1):
    r"""Read TRIANGLE-like format [1]_ files (.node, .ele) into Mesh2D.

    Parameters
    ----------
    filename : str
        Basename of the files where the mesh is stored. Vertex coordinates
        must be stored in filename.node, whereas region-to-connectivity
        data must be stored in filename.ele.
    offset : int
        Offset to be subtracted from vertex indices. (Default value = 0)
    unset : int
        Value to unset vertex and region flags. (Default value = -1)

    Returns
    -------
    Mesh2D

    References
    ----------

    .. [1] https://www.cs.cmu.edu/~quake/triangle.html

    Examples
    --------
    >>> node_s = "4 2 0 1\n0 0.0 0.0 1\n1 0.5 0.0 1\n2 0.5 0.5 1\n3 0.0 0.5 1"
    >>> f = open('example.node', 'w')
    >>> f.write(node_s)
    >>> f.close()
    >>> ele_s = "1 0 1\n0 4 0 1 2 3 10"
    >>> f = open('example.ele', 'w')
    >>> f.write(ele_s)
    >>> f.close()
    >>> from vempy.mesh import read_node_ele
    >>> mesh = read_node_ele('example')
    >>> print(mesh)

    Mesh2D:

    Number of vertices: 4
    Number of faces: 4
    Number of regions: 1

    Vertex coordinates:
    0 0. 0.
    1 0.5 0.
    2 0.5 0.5
    3 0. 0.5

    Region-to-region:
    0 -1 -1 -1 -1

    Region-to-face:
    0 0 2 3 1

    Region-to-vertex:
    0 0 1 2 3

    Face-to-region:
    0 0 -1
    1 0 -1
    2 0 -1
    3 0 -1

    Face-to-face:
    0 1 2
    1 0 3
    2 0 3
    3 1 2

    Face-to-vertex:
    0 0 1
    1 3 0
    2 1 2
    3 2 3

    Vertex-to-region:
    0 0
    1 0
    2 0
    3 0

    Vertex-to-face:
    0 0 1
    1 0 2
    2 2 3
    3 1 3

    Vertex-to-vertex:
    0 1 3
    1 0 2
    2 1 3
    3 0 2

    Name:

    """
    # Read .node first
    with open(filename + '.node', 'r') as file:
        nV, dim, _, nbmrk = np.int64(file.readline().split())
        data = np.loadtxt(file)
        vrtx_coords = data[:, 1:3]
        if dim != 2:
            raise ValueError('Dimension in .node file must be 2')
        if vrtx_coords.shape[0] != nV:
            msg = 'Expected {} vertices, got {}, instead'
            raise IndexError(msg.format(nV, vrtx_coords.shape[0]))
        if nbmrk == 0:
            fV = np.full(nV, fill_value=unset, dtype=int)
        else:
            fV = np.array(data[:, -1], dtype=int)
    # Read .ele
    with open(filename + '.ele', 'r') as file:
        nR, nodelem, nattr = np.int64(file.readline().split())
        regn_vrtx, fR = _read_regn_data(file, vrtx_coords, nR, nodelem, nattr,
                                        True, offset, unset)
    return Mesh2D(vrtx_coords, regn_vrtx=regn_vrtx, fV=fV, fR=fR)


def read_ff(filename, unset=-1):
    """Read a mesh saved in a Face Format file.

    Parameters
    ----------
    filename : str
        Name of the file where the mesh is stored.
    unset : int
        Value to unset vertex and region flags. (Default value = -1)

    Returns
    -------
    Mesh2D

    Notes
    -----
    The structure of a Face Format file is as follows:

    --------------------------------------
    ## the first line is always a comment!

    # offset <offset>

    # nodes (mandatory)
    <# of nodes> 2 0 0
        <node #> <x> <y>
        ...

    # regions (can be omitted)
    <# of regions> 0 0
        <region #> <# of nodes per region> <node> <node> <node> ...
        ...

    # faces (mandatory)
    <# of faces> 0
        <face #> <endpoint> <endpoint> <region> <region>
        ...
    --------------------------------------

    where one <region> is equal to -1 for boundary faces.
    """
    with open(filename, 'r') as file:
        n_sections = 0
        offset = None
        while n_sections < 3:
            line = file.readline()
            if line.startswith('# offset'):
                offset = int(line.split()[-1])
                n_sections += 1
            elif line.startswith('# nodes'):
                nV = int(file.readline().split()[0])
                vrtx_coords = np.loadtxt(file, usecols=[1, 2], max_rows=nV)
                n_sections += 1
            elif line.startswith('# faces'):
                nF = int(file.readline().split()[0])
                data = np.loadtxt(file, max_rows=nF, dtype=int)
                face_vrtx = data[:, 1:3]-offset
                face_regn = data[:, 3:5]-offset
                n_sections += 1
    fV = np.full(nV, fill_value=unset, dtype=int)
    fF = np.full(nF, fill_value=unset, dtype=int)
    return Mesh2D(vrtx_coords, face_vrtx=face_vrtx, face_regn=face_regn,
                  fV=fV, fF=fF)


def read_durham(filename, unset=-1):
    """Read a mesh saved in Durham format.

    Parameters
    ----------
    filename : str
        Name of the file where the mesh is stored.
    unset : int
        Value to unset vertex and region flags. (Default value = -1)

    Returns
    -------
    Mesh2D

    Notes
    -----
    In a Durham file, it is possible to specify connectivity either with
    a region-to-vertex list or with face-to-region and face-to-vertex arrays.
    The structure of a file in Durham format is as follows:

    --------------------------------------
    MESH <version>

    OFFSET <offset>

    POINTS <# of nodes>
    <x> <y>
    ...

    CELL_POINTS <# of regions>
    <# of nodes per region> <node> <node> <node> ...
    ...

    EDGES <# of edges>
    <endpoint> <endpoint> <region> <region>

    <OTHER NON RELEVANT SECTIONS HAVE BEEN OMITTED>
    --------------------------------------
    """
    face_vrtx = None
    face_regn = None
    fF = None
    regn_vrtx = None
    fR = None
    with open(filename, 'r') as file:
        n_sections = 0
        offset = None
        while n_sections < 3:
            line = file.readline()
            if line.startswith('OFFSET'):
                offset = int(line.split()[-1])
                n_sections += 1
            elif line.startswith('POINTS'):
                nV = int(line.split()[-1])
                vrtx_coords = np.loadtxt(file, max_rows=nV)
                fV = np.full(nV, fill_value=unset, dtype=int)
                n_sections += 1
            elif line.startswith('EDGES'):
                nF = int(line.split()[-1])
                data = np.loadtxt(file, max_rows=nF, dtype=int)
                face_vrtx = data[:, :2]-offset
                face_regn = data[:, 2:]-offset
                fF = np.full(nF, fill_value=unset, dtype=int)
                n_sections += 1
            elif line.startswith('CELL_POINTS'):
                nR = int(line.split()[-1])
                regn_vrtx, fR = \
                    _read_regn_data(file, vrtx_coords,
                                    nR, 0, 0, False, offset, unset)
                n_sections += 1
    return Mesh2D(vrtx_coords, regn_vrtx=regn_vrtx,
                  face_vrtx=face_vrtx, face_regn=face_regn,
                  fV=fV, fF=fF, fR=fR)
