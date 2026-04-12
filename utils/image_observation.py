"""Image observation utilities and speckle pattern helpers."""

import numpy as np
import dolfin as dl
import hippylib as hp
from hippylib.modeling.variables import STATE
from geometric_mcmc.model.observable import Observable
from geometric_mcmc.utilities.collective import get_global, set_global
import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple, List, Optional, Union, Callable

jax.config.update("jax_enable_x64", True) # Use 64-bit precision

def setup_image_observation(
    image_corners_coords: np.ndarray, 
    inside: Callable[[np.ndarray], np.ndarray], 
    pixel_density: float, 
    oversampling_factor: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sets up the image observation problem geometry.

    Calculates the pixel grid based on physical corners and density, determines which
    pixels fall inside the material domain, and returns the mask and physical coordinates
    of those valid pixels.

    Args:
        image_corners_coords (np.ndarray): A 2x2 array [[x0, y0], [x1, y1]] defining the 
            physical bounding box of the image.
        inside (Callable): A function that takes an (N, 2) array of physical coordinates 
            and returns a boolean array indicating if they are inside the material domain.
        pixel_density (float): Pixels per unit length.
        oversampling_factor (int, optional): Factor to increase resolution for internal 
            calculations before downsampling. Defaults to 1.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - mask (np.ndarray): A boolean 2D array (H, W) indicating valid material pixels.
            - targets (np.ndarray): An (N_valid, 2) array of physical coordinates for the 
              centers of pixels inside the mask.
    """
    physical_width = image_corners_coords[1, 0] - image_corners_coords[0, 0]
    physical_height = image_corners_coords[1, 1] - image_corners_coords[0, 1]
    W = int(round(physical_width * pixel_density)*oversampling_factor)
    H = int(round(physical_height * pixel_density)*oversampling_factor)
    
    # Generate pixel center coordinates
    y_indices, x_indices = np.mgrid[0:H, 0:W]
    pixel_coords = np.vstack([x_indices.ravel(), y_indices.ravel()]).T
    physical_x = image_corners_coords[0, 0] + (pixel_coords[:, 0] + 0.5) * (physical_width / W)
    physical_y = image_corners_coords[0, 1] + (pixel_coords[:, 1] + 0.5) * (physical_height / H)
    
    physical_coords = np.stack([physical_x, physical_y], axis=1)
    
    # Determine mask and extract target coordinates
    mask = inside(physical_coords).reshape(H, W)
    targets = physical_coords[mask.ravel()]
    return mask, targets

def generate_speckle_pattern(
    image_corners: np.ndarray, 
    density: float, 
    base_speckle_radius: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates speckle centers and radii using Poisson disk sampling.

    This function uses an efficient Poisson disk sampling algorithm (Bridson's)
    to first generate a maximally dense pattern, and then prunes it to achieve
    the desired density.

    Args:
        image_corners (np.ndarray): Physical corners [[x0, y0], [x1, y1]].
        density (float): The target fraction of the area to be covered by speckles (0 to 1).
        base_speckle_radius (float, optional): The radius for the initial dense packing.
                                               If None, it's estimated from the domain size.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - centers (np.ndarray): The (x, y) physical coordinates of speckle centers.
            - radii (np.ndarray): The physical radius of each speckle.
    """
    physical_width = image_corners[1, 0] - image_corners[0, 0]
    physical_height = image_corners[1, 1] - image_corners[0, 1]
    
    if physical_width == 0 or physical_height == 0:
        return np.array([]), np.array([])

    if base_speckle_radius is None:
        base_speckle_radius = min(physical_width, physical_height) / 100.0

    min_dist = 2 * base_speckle_radius
    cell_size = min_dist / np.sqrt(2)
    
    grid_width = int(np.ceil(physical_width / cell_size))
    grid_height = int(np.ceil(physical_height / cell_size))
    
    grid = -np.ones((grid_height, grid_width), dtype=int)
    
    points = []
    active_list = []
    
    # Initial point
    p0 = np.random.rand(2)
    p0[0] = image_corners[0, 0] + p0[0] * physical_width
    p0[1] = image_corners[0, 1] + p0[1] * physical_height
    
    points.append(p0)
    active_list.append(0)
    
    grid_x = int((p0[0] - image_corners[0, 0]) / cell_size)
    grid_y = int((p0[1] - image_corners[0, 1]) / cell_size)
    grid[grid_y, grid_x] = 0
    
    k = 30 # Number of candidates to check around each active point

    while active_list:
        active_idx_ptr = np.random.randint(len(active_list))
        active_idx = active_list[active_idx_ptr]
        active_point = points[active_idx]
        found_candidate = False
        
        for _ in range(k):
            angle = 2 * np.pi * np.random.rand()
            radius = min_dist * (1 + np.random.rand())
            candidate = active_point + radius * np.array([np.cos(angle), np.sin(angle)])
            
            if not (image_corners[0, 0] <= candidate[0] < image_corners[1, 0] and \
                    image_corners[0, 1] <= candidate[1] < image_corners[1, 1]):
                continue

            cand_grid_x = int((candidate[0] - image_corners[0, 0]) / cell_size)
            cand_grid_y = int((candidate[1] - image_corners[0, 1]) / cell_size)

            is_valid = True
            for ny in range(max(0, cand_grid_y - 2), min(grid_height, cand_grid_y + 3)):
                for nx in range(max(0, cand_grid_x - 2), min(grid_width, cand_grid_x + 3)):
                    point_idx = grid[ny, nx]
                    if point_idx != -1:
                        dist_sq = np.sum((points[point_idx] - candidate)**2)
                        if dist_sq < min_dist**2:
                            is_valid = False
                            break
                if not is_valid:
                    break
            
            if is_valid:
                new_point_idx = len(points)
                points.append(candidate)
                active_list.append(new_point_idx)
                grid[cand_grid_y, cand_grid_x] = new_point_idx
                found_candidate = True
                break
        
        if not found_candidate:
            active_list.pop(active_idx_ptr)

    all_centers = np.array(points)
    if len(all_centers) == 0:
        return np.array([]), np.array([])

    # --- Pruning Stage to achieve target density ---
    # Calculate the number of speckles needed to achieve the target density.
    domain_area = physical_width * physical_height
    speckle_area = np.pi * base_speckle_radius**2
    
    # Ensure speckle_area is not zero to avoid division by zero
    if speckle_area == 0:
        return np.array([]), np.array([])
        
    num_to_keep = int((density * domain_area) / speckle_area)
    
    # We cannot keep more points than we generated.
    num_to_keep = min(num_to_keep, len(all_centers))
    
    # Randomly select the subset of centers to keep.
    indices_to_keep = np.random.choice(len(all_centers), size=num_to_keep, replace=False)
    final_centers = all_centers[indices_to_keep]

    num_placed_speckles = len(final_centers)
    radii = np.random.normal(base_speckle_radius, base_speckle_radius / 10, num_placed_speckles)
    
    return final_centers, radii


def speckled_reference(
    mask: np.ndarray, 
    image_corners: np.ndarray, 
    centers: np.ndarray, 
    radii: np.ndarray, 
    contrast: float = 0.8
) -> np.ndarray:
    """
    Renders a speckle image from physical speckle centers and radii.

    Args:
        mask (np.ndarray): The original low-resolution binary mask.
        image_corners (np.ndarray): Physical corners [[x0, y0], [x1, y1]].
        centers (np.ndarray): Physical (x, y) coordinates of speckle centers.
        radii (np.ndarray): Physical radii of speckles.
        contrast (float): The target contrast (0.0 to 1.0).

    Returns:
        np.ndarray: The final rendered speckle pattern as a float array.
    """
    height, width = mask.shape

    if not (0.0 <= contrast <= 1.0):
        raise ValueError("Contrast must be between 0.0 and 1.0")
    mid_point = 112.5
    dark_value = mid_point * (1.0 - contrast)
    light_value = mid_point * (1.0 + contrast)

    image = np.full((height, width), fill_value=light_value, dtype=np.float64)

    # Convert physical units to high-resolution pixel units
    physical_width = image_corners[1, 0] - image_corners[0, 0]
    high_res_pixel_width = physical_width / width

    centers_pixel_x = (centers[:, 0] - image_corners[0, 0]) / high_res_pixel_width
    centers_pixel_y = (centers[:, 1] - image_corners[0, 1]) / high_res_pixel_width # Assume square pixels
    radii_pixel = radii / high_res_pixel_width

    # Create coordinate grids for efficient masking
    y_grid, x_grid = np.ogrid[:height, :width]

    for i in range(len(centers)):
        cx = int(centers_pixel_x[i])
        cy = int(centers_pixel_y[i])
        r = int(radii_pixel[i])

        # Define bounding box to minimize calculations
        y_min = max(0, cy - r)
        y_max = min(height, cy + r + 1)
        x_min = max(0, cx - r)
        x_max = min(width, cx + r + 1)

        if y_min >= y_max or x_min >= x_max:
            continue

        # Calculate distance squared within the bounding box
        # x_grid is (1, W), y_grid is (H, 1)
        dist_sq = (x_grid[:, x_min:x_max] - cx)**2 + (y_grid[y_min:y_max, :] - cy)**2
        mask_circle = dist_sq <= r**2

        # Apply the circle to the image
        image[y_min:y_max, x_min:x_max][mask_circle] = dark_value

    image[~mask] = 0.0
    return image.astype(np.float64)

@partial(jax.jit, static_argnames=['window_size'])
def forward_splat(
    reference_image: jnp.ndarray,
    reference_mask: jnp.ndarray,
    displacement_field: jnp.ndarray,
    image_corners: jnp.ndarray,
    sigma: float = 1.0,
    window_size: int = 5,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Deforms an image using a JIT-compatible and corrected differentiable
    forward-warping (splatting) algorithm. 
    
    This version handles physical displacement fields and converts them to pixel coordinates.
    It uses a Gaussian splatting kernel to avoid holes during large deformations.

    Args:
        reference_image (jnp.ndarray): The 2D source image.
        reference_mask (jnp.ndarray): A 2D binary mask where values > 0 indicate the
            material domain.
        displacement_field (jnp.ndarray): A 3D array of shape (H, W, 2) containing the
            physical (x, y) displacement for each pixel.
        image_corners (jnp.ndarray): Physical corners of the image [[x0, y0], [x1, y1]]
        sigma (float): Standard deviation of the Gaussian kernel in pixels.
        window_size (int): Size of the splatting window (window_size x window_size).
    
    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: 
            - deformed_image: The warped image.
            - weight_canvas: Accumulated splat weights (useful for debugging or normalization).
    """
    H, W = reference_image.shape
    
    # Convert physical displacement to pixel displacement
    physical_width = image_corners[1, 0] - image_corners[0, 0]
    physical_height = image_corners[1, 1] - image_corners[0, 1]
    pixel_width_phys = physical_width / W
    pixel_height_phys = physical_height / H
    
    # Convert physical displacement to pixel displacement
    u_pix_x = displacement_field[..., 0] / pixel_width_phys
    u_pix_y = displacement_field[..., 1] / pixel_height_phys
    
    # Stack into pixel displacement field [H, W, 2] with (row, col) ordering
    pixel_displacement_field = jnp.stack([u_pix_y, u_pix_x], axis=-1)

    # 1. Create a full grid of source coordinates to maintain static shapes for JIT.
    rows, cols = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing='ij')
    source_coords = jnp.stack([rows, cols], axis=-1)

    # Explicitly nullify any displacement vectors outside the material domain.
    effective_displacement = pixel_displacement_field * reference_mask[..., None]

    # 2. Calculate destinations.
    dest_coords_float = source_coords + effective_displacement
    dest_coords_round = jnp.round(dest_coords_float).astype(jnp.int32)

    # 3. Generate neighbor offsets for the window
    start = -(window_size // 2)
    offsets = jnp.arange(start, start + window_size)
    
    neighbor_coords_list = []
    intensities_list = []
    weights_list = []

    # Loop over the window
    for dy in offsets:
        for dx in offsets:
            # Neighbor coordinate (integer)
            neighbor_coord = dest_coords_round + jnp.array([dy, dx])
            
            # Distance squared from the floating point destination
            # Note: we compute distance in pixel units
            dist_vec = neighbor_coord.astype(jnp.float64) - dest_coords_float
            dist_sq = jnp.sum(dist_vec**2, axis=-1)
            
            # Gaussian weight
            weight = jnp.exp(-dist_sq / (2.0 * sigma**2))
            
            # Mask by reference mask (only material pixels contribute)
            # This ensures that background pixels in the source image do not contribute
            # to the deformed image, preventing background values from being splatted.
            weight = weight * reference_mask
            intensity = reference_image * weight
            
            neighbor_coords_list.append(neighbor_coord.reshape(-1, 2))
            intensities_list.append(intensity.ravel())
            weights_list.append(weight.ravel())

    # 4. Prepare for scatter operation by flattening all arrays.
    all_neighbor_coords = jnp.vstack(neighbor_coords_list)
    all_intensities_contr = jnp.hstack(intensities_list)
    all_weights_contr = jnp.hstack(weights_list)

    # 5. Perform JIT-compatible boundary check and scatter operation.
    scatter_rows, scatter_cols = all_neighbor_coords[:, 0], all_neighbor_coords[:, 1]
    valid_mask = (scatter_rows >= 0) & (scatter_rows < H) & (scatter_cols >= 0) & (scatter_cols < W)

    safe_rows = jnp.where(valid_mask, scatter_rows, 0)
    safe_cols = jnp.where(valid_mask, scatter_cols, 0)
    
    safe_intensities = jnp.where(valid_mask, all_intensities_contr, 0)
    safe_weights = jnp.where(valid_mask, all_weights_contr, 0)

    intensity_canvas = jnp.zeros((H, W)).at[safe_rows, safe_cols].add(safe_intensities)
    weight_canvas = jnp.zeros((H, W)).at[safe_rows, safe_cols].add(safe_weights)

    # Use a safe division that doesn't bias the color for low weights (avoiding dark halos)
    # We only normalize where we have significant weight.
    deformed_image = jnp.where(weight_canvas > 1e-12, intensity_canvas / weight_canvas, 0.0)

    return deformed_image, weight_canvas

def _make_warp_then_downsample(
    downsample_factor: int, 
    min_weight: float = 1e-12, 
    sigma: float = 1.0, 
    window_size: int = 5
) -> Callable:
    """
    Creates a JIT-compiled function that warps an image and then averages it over fxf neighborhoods.
    
    For each output pixel (one fxf block):
      - If the representative pixel (top-left of the block) is valid (weight > min_weight), 
        output the mean of all valid pixels in the block.
      - Otherwise output zero.

    Args:
        downsample_factor (int): The factor f by which to downsample the image.
        min_weight (float): Minimum weight threshold for a pixel to be considered valid.
        sigma (float): Sigma for the Gaussian splatting kernel.
        window_size (int): Window size for the splatting kernel.

    Returns:
        Callable: A JIT-compiled JAX function for warping and downsampling.
    """
    f = int(downsample_factor)

    @jax.jit
    def warp_then_downsample(reference_image: jnp.ndarray,
                             reference_mask: jnp.ndarray,
                             displacement_field: jnp.ndarray,
                             image_corners: jnp.ndarray) -> jnp.ndarray:
        deformed, weights = forward_splat(reference_image, reference_mask, displacement_field, image_corners, sigma, window_size)
        valid_bool = weights > min_weight
        valid = valid_bool.astype(deformed.dtype)

        if f == 1:
            # Keep only valid pixels; invalid become zero.
            return jnp.where(valid_bool, deformed, 0.0)

        H, W = deformed.shape
        h = H // f
        w = W // f

        # Reshape-based block sums (has well-supported VJP)
        def block_sum(arr):
            return arr.reshape(h, f, w, f).sum(axis=(1, 3))

        sum_vals = block_sum(deformed * valid)
        count = block_sum(valid)

        # Block average over valid pixels; zero where no valid pixel exists
        # 'valid' mask ensures that we only average over pixels that received 
        # sufficient contribution from the material domain. Background pixels 
        # (where weights <= min_weight) are excluded from the sum and the count.
        avg = jnp.where(count > 0.0, sum_vals / count, 0.0)

        # Gate by the representative pixel (top-left of each block)
        rep_valid = valid_bool[0::f, 0::f]
        return jnp.where(rep_valid, avg, 0.0)

    return warp_then_downsample

# --- Observable Class using the JAX Warping Function ---

class ImageObservable(Observable):
    """
    An Observable that models image-based measurements.
    
    It maps the state (displacement field) to a deformed image. The forward operator involves:
    1. Interpolating the displacement field from the finite element mesh to image pixels.
    2. Warping the reference image using the interpolated displacement field (via splatting).
    3. Optionally downsampling the result.
    """
    def __init__(
        self, 
        Vu: dl.FunctionSpace, 
        image_corners_coords: np.ndarray, 
        reference_image: np.ndarray, 
        reference_mask: np.ndarray, 
        targets: np.ndarray, 
        components: Optional[List[int]] = None, 
        prune_and_sort: bool = False, 
        bc0: Union[List[dl.DirichletBC], dl.DirichletBC] = [], 
        downsampling_factor: int = 1,
        sigma: float = 1.0, 
        window_size: int = 5
    ):
        """
        Initialize the ImageObservable.

        Args:
            Vu (dl.FunctionSpace): The function space for the displacement field.
            image_corners_coords (np.ndarray): Physical corners of the image [[x0, y0], [x1, y1]].
            reference_image (np.ndarray): The reference (undeformed) image.
            reference_mask (np.ndarray): Binary mask indicating the material domain in the image.
            targets (np.ndarray): Physical coordinates of the pixel centers (where displacement is evaluated).
            components (List[int], optional): Components of the field to observe. Defaults to None.
            prune_and_sort (bool, optional): Whether to prune and sort observation points. Defaults to False.
            bc0 (Union[List[dl.DirichletBC], dl.DirichletBC], optional): Dirichlet boundary conditions 
                for the state. Defaults to [].
            downsampling_factor (int, optional): Factor by which to downsample the output image. Defaults to 1.
            sigma (float, optional): Sigma for the Gaussian splatting kernel. Defaults to 1.0.
            window_size (int, optional): Window size for the splatting kernel. Defaults to 5.
        """
        super().__init__()
        self.reference_image = reference_image
        self.targets = targets
        self.B = hp.assemblePointwiseObservation(Vu, targets, components, prune_and_sort)
        self.mpi_comm = self.B.mpi_comm()
        self._help_obs = dl.Vector(self.mpi_comm)
        self.B.init_vector(self._help_obs, 0)
        self._dimension = self.reference_image.size
        if isinstance(bc0, dl.DirichletBC):
            self.bc0 = [bc0]
        else:
            self.bc0 = bc0
        
        # Store reference mask (if provided)
        self.reference_mask = reference_mask

        # Convert data to JAX arrays
        self.ref_image_jax = jnp.array(reference_image.astype(np.float64))
        self.image_corners_coords_jax = jnp.array(image_corners_coords)
        
        # Convert mask to JAX array if it exists
        self.reference_mask_jax = jnp.array(reference_mask.astype(np.float64))
        
        # Pre-compute coordinate mapping for efficiency
        self._precompute_coordinate_mapping()

        # Use downsampling after warp
        self.downsample_factor = int(downsampling_factor)
        H, W = self.reference_image.shape
        if self.downsample_factor > 1:
            if (H % self.downsample_factor) or (W % self.downsample_factor):
                raise ValueError("reference_image shape must be divisible by downsample_factor")
            self.out_shape = (H // self.downsample_factor, W // self.downsample_factor)
        else:
            self.out_shape = (H, W)
        self._dimension = self.out_shape[0] * self.out_shape[1]

        # JAX function that warps then downsamples
        self._warp_then_downsample = _make_warp_then_downsample(self.downsample_factor, sigma=sigma, window_size=window_size)

    def __str__(self):
        return f"ImageObservable(dim={self._dimension}, downsample={self.downsample_factor}, targets={len(self.targets)})"

    def _precompute_coordinate_mapping(self):
        """Pre-compute pixel coordinate mapping to avoid repeated calculations."""
        height, width = self.reference_image.shape
        physical_width = self.image_corners_coords_jax[1, 0] - self.image_corners_coords_jax[0, 0]
        physical_height = self.image_corners_coords_jax[1, 1] - self.image_corners_coords_jax[0, 1]
        pixel_width_phys = physical_width / width
        pixel_height_phys = physical_height / height
        
        target_x_pix = np.round((self.targets[:, 0] - self.image_corners_coords_jax[0, 0]) / pixel_width_phys - 0.5).astype(int)
        target_y_pix = np.round((self.targets[:, 1] - self.image_corners_coords_jax[0, 1]) / pixel_height_phys - 0.5).astype(int)
        
        self.target_y_pix = target_y_pix
        self.target_x_pix = target_x_pix

        # Sanity checks to preserve linearity and adjointness of scatter/gather
        H, W = height, width
        in_bounds = (
            (self.target_y_pix >= 0) & (self.target_y_pix < H) &
            (self.target_x_pix >= 0) & (self.target_x_pix < W)
        )
        if not np.all(in_bounds):
            bad = np.where(~in_bounds)[0][:10]
            raise ValueError(f"Out-of-bounds target-to-pixel indices detected (showing up to 10): {bad}")

        keys = self.target_y_pix.astype(np.int64) * W + self.target_x_pix.astype(np.int64)
        if np.unique(keys).size != keys.size:
            raise ValueError("Duplicate target pixel indices detected; scatter assignment will break adjointness.")

    def _create_displacement_field(self, u_at_targets: np.ndarray) -> np.ndarray:
        """
        Create dense displacement field from target displacements.
        
        Args:
            u_at_targets: Displacements at target points [n_targets, 2]
            
        Returns:
            displacement_field: Dense displacement field [height, width, 2]
        """
        height, width = self.reference_image.shape
        displacement_field = np.zeros((height, width, 2))

        # Place displacements at corresponding pixel locations.
        # No filtering is needed as the coordinate conversion is now robust.
        displacement_field[self.target_y_pix, self.target_x_pix] = u_at_targets
        
        return displacement_field

    def _extract_adjoint_at_targets(self, adj_displacement_field: np.ndarray) -> np.ndarray:
        """
        Extract adjoint values at target locations from dense adjoint field.
        
        Args:
            adj_displacement_field: Dense adjoint field [height, width, 2]
            
        Returns:
            adj_at_targets: Adjoint at all target points [n_targets, 2]
        """
        # Extract adjoint at target locations directly.
        adj_at_targets = adj_displacement_field[self.target_y_pix, self.target_x_pix]
        
        return adj_at_targets

    def dim(self) -> int:
        return self._dimension

    def eval(self, x: List[dl.Vector]) -> np.ndarray:
        """
        Evaluate the observation operator at state x.
        
        Args:
            x (List[dl.Vector]): State list [u, m, p] containing the displacement field at index STATE.
            
        Returns:
            np.ndarray: Flattened array of the deformed (and potentially downsampled) image.
        """
        # Get displacements at target points
        self.B.mult(x[STATE], self._help_obs)
        u_at_targets = get_global(self.mpi_comm, self._help_obs).reshape((-1, 2))
        
        # Create dense displacement field
        displacement_field = self._create_displacement_field(u_at_targets)
        
        # Warp at full resolution, then downsample
        image_ds = self._warp_then_downsample(
            self.ref_image_jax, 
            self.reference_mask_jax,
            jnp.array(displacement_field), 
            self.image_corners_coords_jax
        )
        return np.array(image_ds.ravel())

    def jacobian_mult(self, x: List[dl.Vector], i: int, dir: dl.Vector) -> np.ndarray:
        """
        Apply the Jacobian of the observation operator to a direction.
        
        Args:
            x (List[dl.Vector]): Linearization point (state list).
            i (int): Variable index (must be STATE).
            dir (dl.Vector): Direction vector.
            
        Returns:
            np.ndarray: The result of J * dir.
        """
        if i != STATE:
            return np.zeros(self._dimension)
        
        # Enforce BCs on the directional increment to preserve adjointness
        dir_bc = dir.copy()
        [bc.apply(dir_bc) for bc in self.bc0]

        # Get displacements and directional derivatives at target points
        self.B.mult(x[STATE], self._help_obs)
        u_at_targets = get_global(self.mpi_comm, self._help_obs).reshape((-1, 2))
        
        self.B.mult(dir_bc, self._help_obs)
        delta_u_at_targets = get_global(self.mpi_comm, self._help_obs).reshape((-1, 2))
        
        # Create dense displacement fields
        displacement_field = self._create_displacement_field(u_at_targets)
        delta_displacement_field = self._create_displacement_field(delta_u_at_targets)

        # JVP through warp-then-downsample
        _, J_dir = jax.jvp(
            lambda p: self._warp_then_downsample(self.ref_image_jax, self.reference_mask_jax, p, 
                                                 self.image_corners_coords_jax),
            (jnp.array(displacement_field),),
            (jnp.array(delta_displacement_field),)
        )
        if np.isnan(J_dir).any() or np.isinf(J_dir).any():
            raise FloatingPointError("NaN/Inf detected in ImageObservable.jacobian_mult output")
        return np.array(J_dir.ravel())

    def jacobian_transpmult(self, x: List[dl.Vector], i: int, dir: np.ndarray, out: dl.Vector):
        """
        Apply the transpose of the Jacobian to a direction.
        
        Args:
            x (List[dl.Vector]): Linearization point (state list).
            i (int): Variable index (must be STATE).
            dir (np.ndarray): Direction vector (in observation space).
            out (dl.Vector): Output vector (in state space).
        """
        if i != STATE:
            out.zero()
            return

        out.zero()  # avoid accumulation across calls

        # Get displacements at target points
        self.B.mult(x[STATE], self._help_obs)
        u_at_targets = get_global(self.mpi_comm, self._help_obs).reshape((-1, 2))
        
        # Create dense displacement field
        displacement_field = self._create_displacement_field(u_at_targets)

        # VJP through warp-then-downsample
        _, vjp_fun = jax.vjp(
            lambda p: self._warp_then_downsample(self.ref_image_jax, self.reference_mask_jax, p, 
                                                 self.image_corners_coords_jax),
            jnp.array(displacement_field)
        )
        
        dir_reshaped = jnp.array(dir).reshape(self.out_shape)
        adj_displacement_field_jax = vjp_fun(dir_reshaped)[0]
        adj_displacement_field = np.array(adj_displacement_field_jax)

        # Extract adjoint at target locations (gather = adjoint of scatter)
        full_adj_at_targets = self._extract_adjoint_at_targets(adj_displacement_field)
        
        # Apply transpose of observation operator B
        set_global(self.mpi_comm, full_adj_at_targets.ravel(), self._help_obs)
        self.B.transpmult(self._help_obs, out)

        # Enforce BCs on the adjoint
        [bc.apply(out) for bc in self.bc0]

    def setLinearizationPoint(self, x: List[dl.Vector]):
        pass

    def apply_ijk(self, i, j, k, dir_j, dir_k, out):
        """
        Apply the second derivative of the observation operator.
        
        Note: This implementation returns zero, effectively using a Gauss-Newton approximation
        for the Hessian of the misfit functional. The true second derivative of the 
        warping operator is ignored.
        """
        if isinstance(out, np.ndarray):
            out.fill(0)
        else:
            out.zero()
        return