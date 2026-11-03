"""Exports of an RVEResult (dragen.api), all with the signature export_xxx(result, path).

export_damask   DAMASK grid: <dir>/grid.vti + <dir>/material.yaml          (extra 'damask')
export_vtu      tetrahedral mesh as VTK unstructured grid, <path>.vtu
export_gmsh     tetrahedral mesh as Gmsh MSH 2.2, <path>.msh; physical group = grain ID
export_exodus   tetrahedral mesh as Exodus II, <path>.e; one element block per grain (MOOSE)
                (needs netCDF4, extra 'extra')

Lengths in metres, as in the RVEResult.
"""
from dragen.export.damask_grid import export_damask
from dragen.export.tet import export_exodus, export_gmsh, export_vtu

__all__ = ['export_damask', 'export_vtu', 'export_gmsh', 'export_exodus']
