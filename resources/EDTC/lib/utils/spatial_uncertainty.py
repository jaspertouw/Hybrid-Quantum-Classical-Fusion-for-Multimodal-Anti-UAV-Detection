"""Spatial-variance normalization shared by detector inference and tests."""


def corrected_variance_components(sigma2_grid, width_px, height_px, stride):
    """Return dimensionless variances from grid-coordinate variances."""
    width2 = width_px.clip(min=1.0) ** 2
    height2 = height_px.clip(min=1.0) ** 2
    sigma2_px = sigma2_grid * (stride ** 2)
    return (
        sigma2_px[..., 0:1] / width2,
        sigma2_px[..., 1:2] / height2,
        sigma2_px[..., 2:3] / width2,
        sigma2_px[..., 3:4] / height2,
    )


def inherited_variance_components(sigma2_grid, width_px, height_px):
    """Return the inherited Suttorp-code values for audit comparisons."""
    return (
        sigma2_grid[..., 0:1] / (width_px + 1e-6),
        sigma2_grid[..., 1:2] / (height_px + 1e-6),
        sigma2_grid[..., 2:3] / (width_px ** 2 + 1e-6),
        sigma2_grid[..., 3:4] / (height_px ** 2 + 1e-6),
    )
