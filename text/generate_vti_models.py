

import os
import numpy as np

try:
    import vtk
    from vtk.util import numpy_support
except ImportError:
    import subprocess
    subprocess.run(['pip', 'install', 'vtk'])
    import vtk
    from vtk.util import numpy_support


def _save_as_vti(model, output_path, spacing=(100.0, 100.0, 100.0)):
    nz, ny, nx = model.shape
    dz, dy, dx = spacing

    image_data = vtk.vtkImageData()
    image_data.SetDimensions(nx, ny, nz)
    image_data.SetSpacing(dx, dy, dz)
    image_data.SetOrigin(0, 0, 0)

    density_flat = model.flatten(order='F')
    vtk_array = numpy_support.numpy_to_vtk(density_flat, deep=True,
                                           array_type=vtk.VTK_FLOAT)
    vtk_array.SetName("Density")
    image_data.GetPointData().SetScalars(vtk_array)

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(image_data)
    writer.Write()

    print(f"  ✓ Saved: {output_path}  ({os.path.getsize(output_path) / 1024:.1f} KB)")


def generate_figure8_staircase(grid_shape=(16, 32, 32), density=0.5):
    nz, ny, nx = grid_shape
    model = np.zeros((nz, ny, nx), dtype=np.float32)

    step_count = 6
    step_size = 2

    for i in range(step_count):
        x_center = 6 + i * 3
        z_center = 3 + i * 2
        y_center = 12
        x_s, x_e = max(0, x_center - step_size // 2), min(nx, x_center + step_size // 2 + 1)
        y_s, y_e = max(0, y_center - step_size // 2), min(ny, y_center + step_size // 2 + 1)
        z_s, z_e = max(0, z_center - step_size // 2), min(nz, z_center + step_size // 2 + 1)
        model[z_s:z_e, y_s:y_e, x_s:x_e] = density

    for i in range(step_count):
        x_center = 25 - i * 3
        z_center = 3 + i * 2
        y_center = 20
        x_s, x_e = max(0, x_center - step_size // 2), min(nx, x_center + step_size // 2 + 1)
        y_s, y_e = max(0, y_center - step_size // 2), min(ny, y_center + step_size // 2 + 1)
        z_s, z_e = max(0, z_center - step_size // 2), min(nz, z_center + step_size // 2 + 1)
        model[z_s:z_e, y_s:y_e, x_s:x_e] = density

    return model


def generate_center_prism(grid_shape=(16, 32, 32), density=1.0):
    nz, ny, nx = grid_shape
    model = np.zeros((nz, ny, nx), dtype=np.float32)
    model[4:12, 10:22, 10:22] = density
    return model


def generate_dual_prism(grid_shape=(16, 32, 32), density=1.0):
    nz, ny, nx = grid_shape
    model = np.zeros((nz, ny, nx), dtype=np.float32)
    model[4:12, 10:22, 4:12] = density
    model[4:12, 10:22, 20:28] = -density
    return model


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "test_models")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 50)
    print("VTI Model Generator")
    print("=" * 50)

    print("\n[1/3] Figure-8 Staircase")
    m1 = generate_figure8_staircase()
    p1 = os.path.join(out_dir, "figure8_staircase.vti")
    _save_as_vti(m1, p1)
    print(f"  Non-zero voxels: {np.count_nonzero(m1)}")
    print(f"  Value range: [{m1.min():.2f}, {m1.max():.2f}]")

    print("\n[2/3] Center Prism")
    m2 = generate_center_prism()
    p2 = os.path.join(out_dir, "center_prism.vti")
    _save_as_vti(m2, p2)
    print(f"  Non-zero voxels: {np.count_nonzero(m2)}")
    print(f"  Value range: [{m2.min():.2f}, {m2.max():.2f}]")

    print("\n[3/3] Dual Prism")
    m3 = generate_dual_prism()
    p3 = os.path.join(out_dir, "dual_prism.vti")
    _save_as_vti(m3, p3)
    print(f"  Non-zero voxels: {np.count_nonzero(m3)}")
    print(f"  Value range: [{m3.min():.2f}, {m3.max():.2f}]")

    print("\n" + "=" * 50)
    print("All models generated!")
    print(f"Output directory: {out_dir}")
    print("=" * 50)
