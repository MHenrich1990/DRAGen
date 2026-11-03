"""Tetrahedral mesh of an RVEResult through meshio."""
from pathlib import Path

import numpy as np

from dragen._extras import require


def _mesh(result):
    if result.tet_mesh is None:
        raise ValueError("the RVE has no tetrahedral mesh; generate it with RVESpec(mesh='tet')")
    return result.tet_mesh


def _cell_data(result, tet):
    grains = tet.grain_ids
    return {'grain_id': grains.astype(np.int32),
            'phase_id': result.phase_ids[grains - 1].astype(np.int32),
            'orientation': result.orientation_of(grains)}


def export_vtu(result, path: Path) -> Path:
    """<path>.vtu with cell data grain_id, phase_id and orientation (quaternion w, x, y, z)."""
    import meshio
    tet = _mesh(result)
    path = Path(path).with_suffix('.vtu')
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {k: [v] for k, v in _cell_data(result, tet).items()}
    meshio.Mesh(tet.points, [(tet.cell_type, tet.cells)], cell_data=data).write(path)
    return path


def export_gmsh(result, path: Path) -> Path:
    """<path>.msh, Gmsh format 2.2 (ASCII), read by gmsh, meshio and MOOSE/libMesh. Physical and
    geometrical tag of every element = its grain ID."""
    import meshio
    tet = _mesh(result)
    path = Path(path).with_suffix('.msh')
    path.parent.mkdir(parents=True, exist_ok=True)
    grains = tet.grain_ids.astype(np.int32)
    mesh = meshio.Mesh(tet.points, [(tet.cell_type, tet.cells)],
                       cell_data={'gmsh:physical': [grains], 'gmsh:geometrical': [grains]})
    meshio.write(path, mesh, file_format='gmsh22', binary=False)
    return path


def export_exodus(result, path: Path) -> Path:
    """<path>.e, Exodus II for MOOSE: one element block per grain, block ID = grain ID (MOOSE
    subdomain ID), block name grain_<ID>. Needs netCDF4 (extra 'extra').

    meshio writes the blocks (numbered from 0, without names or cell data); the IDs and names are
    set in the file afterwards."""
    import meshio
    netCDF4 = require('netCDF4', 'extra')
    tet = _mesh(result)
    path = Path(path).with_suffix('.e')
    path.parent.mkdir(parents=True, exist_ok=True)
    grains = tet.grain_ids
    order = np.argsort(grains, kind='stable')
    ids, starts = np.unique(grains[order], return_index=True)
    blocks = np.split(order, starts[1:])
    meshio.write(path, meshio.Mesh(tet.points, [(tet.cell_type, tet.cells[b]) for b in blocks]),
                 file_format='exodus')
    with netCDF4.Dataset(path, 'r+') as ds:
        ds['eb_prop1'][:] = ids.astype(np.int32)
        ds['eb_prop1'].setncattr('name', 'ID')
        if 'len_name' not in ds.dimensions:
            ds.createDimension('len_name', 33)
        names = ds.createVariable('eb_names', 'S1', ('num_el_blk', 'len_name'))
        width = len(ds.dimensions['len_name'])
        names[:] = netCDF4.stringtochar(np.array([f'grain_{g}' for g in ids], dtype=f'S{width}'))
    return path
