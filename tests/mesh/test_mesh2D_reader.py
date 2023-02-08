import pytest
from vempy.mesh import read_off, read_node_ele


@pytest.fixture
def write_single_element_off_node_ele(tmpdir):
    """Save a mesh made of a single square element to
    .off and .node, .ele formats."""
    off_path = tmpdir.join("square.off")
    with open(off_path, "w") as off_file:
        off_file.write("OFF\n"
                       "4 1 0\n"
                       "0 0 0\n"
                       "1 0 0\n"
                       "1 1 0\n"
                       "0 1 0\n"
                       "4 0 1 2 3\n")
    nodele_path = tmpdir.join("square")
    with open(nodele_path + ".node", "w") as node_file:
        node_file.write("4 2 0 0\n"
                        "0 0.0 0.0\n"
                        "1 1.0 0.0\n"
                        "2 1.0 1.0\n"
                        "3 0.0 1.0\n")
    with open(nodele_path + ".ele", "w") as ele_file:
        ele_file.write("1 0 0\n0 4 0 1 2 3\n")
    yield off_path, nodele_path


class TestReaderEquivalence(object):
    """A container for testing readers equivalence."""

    def test_off_node_ele(self, write_single_element_off_node_ele):
        """Test equivalence of .off and .node, .ele formats."""
        off_path, nodele_path = write_single_element_off_node_ele
        mesh1 = read_off(off_path)
        mesh2 = read_node_ele(nodele_path)
        assert mesh1 == mesh2
