import pytest
import os
from vempy.mesh import read_off, read_node_ele, read_ff, read_durham


@pytest.fixture
def write_unit_square_off():
    """Save a mesh made of a single square element to .off format."""
    off_path = "square.off"
    with open(off_path, "w") as off_file:
        off_file.write("OFF\n"
                       "4 1 0\n"
                       "0 0 0\n"
                       "1 0 0\n"
                       "1 1 0\n"
                       "0 1 0\n"
                       "4 0 1 2 3\n")
    yield off_path
    os.remove(off_path)


@pytest.fixture
def write_unit_square_node_ele():
    """Save a mesh made of a single square element to .node, .ele format."""
    nodele_path = "square"
    with open(nodele_path + ".node", "w") as node_file:
        node_file.write("4 2 0 0\n"
                        "0 0.0 0.0\n"
                        "1 1.0 0.0\n"
                        "2 1.0 1.0\n"
                        "3 0.0 1.0\n")
    with open(nodele_path + ".ele", "w") as ele_file:
        ele_file.write("1 0 0\n0 4 0 1 2 3\n")
    yield nodele_path
    os.remove(nodele_path + ".node")
    os.remove(nodele_path + ".ele")


@pytest.fixture
def write_unit_square_ff():
    """Write a mesh made of a single square element to .ff format."""
    ff_path = "square.ff"
    with open(ff_path, "w") as ff_file:
        ff_file.write("# This is a comment\n"
                      "\n"
                      "# offset 1\n"
                      "\n"
                      "# nodes\n"
                      "4 2 0 0\n"
                      "1 0.0 0.0\n"
                      "2 1.0 0.0\n"
                      "3 1.0 1.0\n"
                      "4 0.0 1.0\n"
                      "\n"
                      "# faces\n"
                      "4 0\n"
                      "1 1 2 1 -1\n"
                      "2 4 1 1 -1\n"
                      "3 2 3 1 -1\n"
                      "4 3 4 1 -1\n")
    yield ff_path
    os.remove(ff_path)


@pytest.fixture
def write_unit_square_durham_cells():
    """Write a mesh made of a single square element to Durham format,
    by specifying the CELL_POINTS field."""
    durham_path = "square.durham_cells"
    with open(durham_path, "w") as durham_file:
        durham_file.write("\n"
                          "OFFSET 2\n"
                          "\n"
                          "POINTS 4\n"
                          "0.0 0.0\n"
                          "1.0 0.0\n"
                          "1.0 1.0\n"
                          "0.0 1.0\n"
                          "\n\n"
                          "CELL_POINTS 1\n"
                          "4 2 3 4 5\n\n")
    yield durham_path
    os.remove(durham_path)


@pytest.fixture
def write_unit_square_durham_edges():
    """Write a mesh made of a single square element to Durham format,
    by specifying the EDGES field."""
    durham_path = "square.durham_edges"
    with open(durham_path, "w") as durham_file:
        durham_file.write("\n"
                          "OFFSET 10\n"
                          "\n"
                          "POINTS 4\n"
                          "0.0 0.0\n"
                          "1.0 0.0\n"
                          "1.0 1.0\n"
                          "0.0 1.0\n"
                          "\n\n"
                          "EDGES 4\n"
                          "10 11 10 -1\n"
                          "13 10 10 -1\n"
                          "11 12 10 -1\n"
                          "12 13 10 -1\n\n\n")
    yield durham_path
    os.remove(durham_path)


class TestReaderEquivalence(object):
    """A container for testing readers equivalence."""

    def test_off_node_ele(self,
                          write_unit_square_off, write_unit_square_node_ele):
        """Test equivalence of .off and .node, .ele formats."""
        off_path = write_unit_square_off
        nodele_path = write_unit_square_node_ele
        mesh1 = read_off(off_path)
        mesh2 = read_node_ele(nodele_path)
        assert mesh1 == mesh2

    def test_off_ff(self,
                    write_unit_square_off, write_unit_square_ff):
        """Test equivalence of .off and .ff formats."""
        off_path = write_unit_square_off
        ff_path = write_unit_square_ff
        mesh1 = read_off(off_path)
        mesh2 = read_ff(ff_path)
        assert mesh1 == mesh2

    def test_off_durham(self,
                        write_unit_square_off,
                        write_unit_square_durham_cells,
                        write_unit_square_durham_edges):
        """Test equivalence of .off and Durham formats."""
        off_path = write_unit_square_off
        durham_cells_path = write_unit_square_durham_cells
        durham_edges_path = write_unit_square_durham_edges
        mesh1 = read_off(off_path)
        mesh2 = read_durham(durham_cells_path)
        mesh3 = read_durham(durham_edges_path)
        assert mesh1 == mesh2
        assert mesh1 == mesh3
