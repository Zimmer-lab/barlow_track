"""Interactive napari viewer for Step 1 (global-before-crop) augmentation.

Shows ONE augmented view at a time next to the original, so augmentation
parameters can be tuned by eye:

Main viewer (full volume, 3D)
    - image layers: ``raw`` and ``augmented`` (toggle with the eye icons)
    - points layers: ``points_raw`` and ``points_aug`` (same transform
      applied to voxels and coordinates, so they stay consistent)
    - shapes layers: ``crop_box_raw`` / ``crop_box_aug`` (12-edge 3D box
      outline of the currently selected neuron crop, so the crop source is
      obvious in the full volume)

Crop viewer (single selected neuron, 3D)
    - image layers: ``crop_raw`` and ``crop_aug``

A magicgui dock widget exposes every augmentation parameter (checkbox to
enable/disable each transform, spinbox/slider for its values) plus
``frame`` / ``crop_idx`` / ``seed`` and a ``Re-augment`` button.
``frame`` and ``crop_idx`` sliders update live (frame reloads the volume,
crop reuses the current augmentation); other parameters apply on
``Re-augment``.

napari and magicgui are OPTIONAL dependencies: all computation lives in
:class:`ViewerState` / :func:`augment_frame_for_viewer` (pure numpy/torch)
and the GUI is only imported inside :func:`show_volume_augmentation`.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from barlow_track.utils.data_loading import get_3d_crop_using_bbox_or_centroid
from barlow_track.utils.volume_data import (
    DEFAULT_CROP_PHOTOMETRIC_ARGS,
    DEFAULT_GLOBAL_ARGS,
    DEFAULT_POSITION_ARGS,
    apply_global_affine,
    apply_position_jitter,
    build_photometric_transform,
    extract_crops,
    get_centroids_for_volume,
    load_volume,
    resolve_dropout_p,
    sample_dropout_keep,
    sample_global_affine,
)

DEFAULT_TARGET_SZ = (8, 64, 64)

# napari Viewer is a pydantic model (no arbitrary attributes), so keep dock
# widgets alive here; Qt parenting via add_dock_widget owns them as well.
_LIVE_WIDGETS = []


# --------------------------------------------------------------------------
# Project loading (with compat for old project configs)
# --------------------------------------------------------------------------

def load_project_for_viewer(project_path):
    """Load a wbfm ProjectData for visualization.

    Old projects (e.g. ZIM2165_Gcamp7b_worm1-2022_11_28) store the misspelled
    key ``exposture_time`` in ``physical_units``, which current wbfm rejects
    with ``TypeError: unexpected keyword argument 'exposture_time'``. Retry
    with that key dropped (its value equals the current default anyway)
    instead of forcing the user to edit their project config.
    """
    from wbfm.utils.projects.finished_project_data import ProjectData

    try:
        return ProjectData.load_final_project_data(project_path, allow_hybrid_loading=True)
    except TypeError as e:
        if 'exposture_time' not in str(e):
            raise
        logging.warning("Old project config with 'exposture_time' key; retrying without it")
        import wbfm.utils.projects.physical_units as pu

        _orig_init = pu.PhysicalUnitConversion.__init__

        def _patched_init(self, *args, **kwargs):
            kwargs.pop('exposture_time', None)
            _orig_init(self, *args, **kwargs)

        pu.PhysicalUnitConversion.__init__ = _patched_init
        try:
            return ProjectData.load_final_project_data(project_path, allow_hybrid_loading=True)
        finally:
            pu.PhysicalUnitConversion.__init__ = _orig_init


# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------

@dataclass
class ViewerParams:
    """All tunable augmentation parameters plus frame/crop selection.

    Boolean ``use_*`` flags map to the widget checkboxes; when False the
    corresponding transform is skipped, when True it is always applied
    (probability is fixed to 1, unlike training where it is stochastic).
    """
    # Global affine (applied to full volume + points)
    use_affine: bool = True
    max_degrees_z: float = DEFAULT_GLOBAL_ARGS['max_degrees_z']
    scale_jitter: float = DEFAULT_GLOBAL_ARGS['scale_jitter']
    max_translation_z: float = DEFAULT_GLOBAL_ARGS['max_translation'][0]
    max_translation_xy: float = DEFAULT_GLOBAL_ARGS['max_translation'][1]
    p_flip: float = DEFAULT_GLOBAL_ARGS['p_flip']
    # Per-crop photometric
    use_blur: bool = DEFAULT_CROP_PHOTOMETRIC_ARGS['p_blur'] > 0
    use_noise: bool = DEFAULT_CROP_PHOTOMETRIC_ARGS['p_noise'] > 0
    std_noise: float = DEFAULT_CROP_PHOTOMETRIC_ARGS['std_noise']
    # Position-only (applied after the global affine, per view, like training)
    jitter_z: float = 0.0
    jitter_xy: float = 0.0
    use_dropout: bool = False
    dropout_p: float = DEFAULT_POSITION_ARGS['dropout_p']
    min_keep: int = DEFAULT_POSITION_ARGS['min_keep']
    # Selection
    seed: int = 0
    frame: int = 0
    crop_idx: int = 0

    def global_args_dict(self) -> Dict:
        return dict(
            p_global_affine=1.0 if self.use_affine else 0.0,
            max_degrees_z=self.max_degrees_z,
            scale_jitter=self.scale_jitter if self.use_affine else 0.0,
            max_translation=(self.max_translation_z, self.max_translation_xy,
                             self.max_translation_xy),
            p_flip=self.p_flip if self.use_affine else 0.0,
        )

    def photometric_args_dict(self) -> Dict:
        return dict(
            p_blur=1.0 if self.use_blur else 0.0,
            p_noise=1.0 if self.use_noise else 0.0,
            std_noise=self.std_noise,
        )

    def position_args_dict(self) -> Dict:
        """Same keys as the train-config ``position_augment`` section."""
        return dict(
            use_dropout=bool(self.use_dropout),
            jitter_std=(float(self.jitter_z), float(self.jitter_xy),
                        float(self.jitter_xy)),
            dropout_p=float(self.dropout_p),
            min_keep=int(self.min_keep),
        )


# --------------------------------------------------------------------------
# Pure computation (no GUI imports)
# --------------------------------------------------------------------------

@dataclass
class AugmentedFrame:
    """Result of augmenting one frame for display (all numpy float32)."""
    vol_aug: np.ndarray          # (Z, X, Y)
    pts_aug: np.ndarray          # (K, 3) zxy, kept subset after position dropout
    crop_raw: np.ndarray         # (tz, tx, ty) selected crop, normalized
    crop_aug: np.ndarray         # (tz, tx, ty) selected crop, augmented
    bbox_raw: list               # [z0, x0, y0, z1, x1, y1] in raw volume
    bbox_aug: list               # same, in augmented volume
    R: np.ndarray = field(repr=False)
    t_vec: np.ndarray = field(repr=False)
    keep_idx: np.ndarray = field(repr=False, default=None)
    # keep_idx maps kept (augmented-view) rows back to raw point rows, i.e.
    # pts_aug[k] corresponds to the raw point keep_idx[k]. Identity
    # (arange) when dropout is disabled.


def _normalize_crops(crops: np.ndarray) -> np.ndarray:
    """Same intensity normalization the training path ends with."""
    import torchio as tio

    norm = tio.RescaleIntensity(percentiles=(5, 100))
    out = norm(crops)
    return np.asarray(out, dtype=np.float32)


def augment_frame_for_viewer(volume: np.ndarray, points_zxy: np.ndarray,
                             params: ViewerParams,
                             target_sz: Tuple[int, int, int] = DEFAULT_TARGET_SZ
                             ) -> AugmentedFrame:
    """Apply Step 1 augmentation to one frame (single augmented view).

    Mirrors ``VolumeCoordsDataset._augmented_view``: sample one global affine,
    apply it to the volume AND the points, add per-view position jitter,
    drop a per-view subset, extract the selected crop from both, and run the
    shared photometric transform on the augmented crop.
    """
    import torch

    rng = np.random.RandomState(params.seed)
    R, t_vec = sample_global_affine(rng, **params.global_args_dict())
    vol_aug, pts_aug = apply_global_affine(np.asarray(volume, dtype=np.float32),
                                           np.asarray(points_zxy, dtype=float), R, t_vec)
    pos_args = params.position_args_dict()
    pts_jit = apply_position_jitter(pts_aug, rng, pos_args['jitter_std'])
    _, keep_idx = sample_dropout_keep(len(pts_jit), rng,
                                      dropout_p=resolve_dropout_p(pos_args),
                                      min_keep=pos_args['min_keep'])
    keep_idx = np.asarray(keep_idx, dtype=int)
    pts_kept = pts_jit[keep_idx] if len(keep_idx) else np.zeros((0, 3))
    target_sz = tuple(target_sz)
    n_raw = len(points_zxy)
    n_kept = len(keep_idx)
    idx = int(np.clip(params.crop_idx, 0, max(n_kept - 1, 0))) if n_kept else 0
    raw_idx = int(keep_idx[idx]) if n_kept else 0

    full_sz = np.array([1, *np.shape(volume)])
    raw_crops = extract_crops(np.asarray(volume, dtype=np.float32),
                              np.asarray(points_zxy, dtype=float), np.array(target_sz))
    # Normalize over the full stack, exactly like the training path, so the
    # raw and augmented crops are directly comparable
    crop_raw = _normalize_crops(raw_crops)[raw_idx] if n_raw else np.zeros(target_sz, np.float32)

    aug_crops = extract_crops(vol_aug, pts_kept, np.array(target_sz))
    transform = build_photometric_transform(params.photometric_args_dict())
    if n_kept:
        with torch.no_grad():
            out = transform(torch.from_numpy(aug_crops))
        crop_aug = np.asarray(out if not torch.is_tensor(out) else out.numpy(),
                              dtype=np.float32)[idx]
    else:
        crop_aug = np.zeros(target_sz, np.float32)

    _, bbox_raw = get_3d_crop_using_bbox_or_centroid(
        np.asarray(points_zxy[raw_idx]) if n_raw else np.zeros(3), full_sz, np.array(target_sz), volume)
    _, bbox_aug = get_3d_crop_using_bbox_or_centroid(
        pts_kept[idx] if n_kept else np.zeros(3), full_sz, np.array(target_sz), vol_aug)
    return AugmentedFrame(vol_aug=vol_aug, pts_aug=np.asarray(pts_kept, dtype=np.float32),
                          crop_raw=crop_raw, crop_aug=crop_aug,
                          bbox_raw=list(bbox_raw), bbox_aug=list(bbox_aug),
                          R=R, t_vec=t_vec, keep_idx=keep_idx)


def box_edges_from_bbox(bbox) -> np.ndarray:
    """12 edges of an XYZ box as a (12, 2, 3) array for a napari shapes layer.

    napari rectangles are 2D-only, but (possibly non-planar) ``line`` shapes
    are nD-capable, so the box is expressed as its 12 edges in (z, x, y).
    """
    z0, x0, y0, z1, x1, y1 = [float(v) for v in bbox]
    corners = np.array([[z, x, y] for z in (z0, z1) for x in (x0, x1) for y in (y0, y1)])
    # corners index bits: (z, x, y); edges connect pairs differing in one bit
    edges = []
    for i in range(8):
        for bit in range(3):
            j = i ^ (1 << (2 - bit))
            if j > i:
                edges.append([corners[i], corners[j]])
    return np.asarray(edges, dtype=np.float32)


def raw_point_labels(num_points: int, seg_ids=None) -> list:
    """Text labels for raw points: plain ``"{idx}"`` matching the crop slider."""
    return [str(i) for i in range(int(num_points))]


def aug_point_labels(keep_idx, seg_ids=None) -> list:
    """Text labels for kept (augmented-view) points: plain ``"{kept_idx}"``.

    The kept position is what the ``crop_idx`` slider selects, so the number
    on each point is exactly the slider value that shows its crop.
    """
    keep_idx = np.asarray(list(keep_idx), dtype=int) if keep_idx is not None else np.zeros((0,), dtype=int)
    return [str(k) for k in range(len(keep_idx))]


def crop_window_title(state: 'ViewerState') -> str:
    """Crop viewer title showing kept idx, raw idx and seg id for selection."""
    k = int(state.params.crop_idx)
    raw = state.current_raw_idx
    seg = state.current_seg_id
    if raw is None:
        return f"augment: crop {k}"
    if seg is None:
        return f"augment: crop {k} (raw {raw})"
    return f"augment: crop {k} (raw {raw}, seg {seg})"


def highlight_sizes(n: int, selected: int, base: float = 3.0,
                    selected_size: float = 8.0) -> np.ndarray:
    """Per-point sizes with the selected crop enlarged (napari points layer)."""
    sizes = np.full(int(n), float(base), dtype=float)
    if int(n) and 0 <= int(selected) < int(n):
        sizes[int(selected)] = float(selected_size)
    return sizes


class ViewerState:
    """Headless state for the viewer: frame data + current augmentation.

    All methods are pure numpy/torch so they can be tested without a display.
    The GUI in :func:`show_volume_augmentation` just calls these and pushes
    the arrays into napari layers.
    """

    def __init__(self, project_data, target_sz: Tuple[int, int, int] = DEFAULT_TARGET_SZ,
                 params: Optional[ViewerParams] = None):
        self.project_data = project_data
        self.target_sz = tuple(target_sz)
        self.params = params or ViewerParams()
        self.volume: Optional[np.ndarray] = None
        self.points: Optional[np.ndarray] = None
        self.seg_ids: Optional[np.ndarray] = None
        self.result: Optional[AugmentedFrame] = None
        self.load_frame(self.params.frame)

    @property
    def num_points(self) -> int:
        return 0 if self.points is None else len(self.points)

    @property
    def num_kept(self) -> int:
        """Number of objects kept in the current augmented view (after dropout)."""
        if self.result is None or self.result.keep_idx is None:
            return self.num_points
        return len(self.result.keep_idx)

    @property
    def current_raw_idx(self) -> Optional[int]:
        """Raw point row for the current crop (maps through dropout keep_idx)."""
        if self.num_points == 0:
            return None
        if self.result is None or self.result.keep_idx is None or len(self.result.keep_idx) == 0:
            return int(np.clip(self.params.crop_idx, 0, self.num_points - 1))
        idx = int(np.clip(self.params.crop_idx, 0, len(self.result.keep_idx) - 1))
        return int(self.result.keep_idx[idx])

    @property
    def current_seg_id(self) -> Optional[int]:
        """Raw segmentation id of the currently selected crop, if known."""
        if self.seg_ids is None or len(self.seg_ids) == 0:
            return None
        raw_idx = self.current_raw_idx
        if raw_idx is None:
            return None
        return int(self.seg_ids[int(np.clip(raw_idx, 0, len(self.seg_ids) - 1))])

    def load_frame(self, t: int) -> None:
        t = int(np.clip(t, 0, self.project_data.num_frames - 1))
        self.params.frame = t
        self.volume = load_volume(self.project_data, t)
        self.points, self.seg_ids = get_centroids_for_volume(self.project_data, t)
        self.points = np.asarray(self.points, dtype=float)
        if self.num_points == 0:
            raise ValueError(f"Frame {t} has no detected centroids; pick another frame")
        self.reaugment()

    def reaugment(self) -> AugmentedFrame:
        self.params.crop_idx = int(np.clip(self.params.crop_idx, 0, self.num_points - 1))
        self.result = augment_frame_for_viewer(self.volume, self.points,
                                               self.params, self.target_sz)
        # Dropout may shrink the kept set; keep the slider in range
        self.params.crop_idx = int(np.clip(self.params.crop_idx, 0, self.num_kept - 1)) \
            if self.num_kept else 0
        return self.result

    def set_crop_idx(self, idx: int) -> AugmentedFrame:
        """Switch the secondary (single-neuron) crop window to another neuron.

        ``idx`` indexes the KEPT (post-dropout) augmented view, matching the
        ``crop_idx`` slider. Updates ``params.crop_idx`` and recomputes only
        the selected crops + bboxes, reusing the already-augmented
        volume/points so the global affine (and the main viewer) stays put.
        Falls back to a full :meth:`reaugment` when no augmentation exists yet.
        """
        if self.num_points == 0:
            raise ValueError("No centroids loaded; cannot select a neuron")
        n_kept = self.num_kept if self.result is not None else self.num_points
        idx = int(np.clip(int(idx), 0, max(n_kept - 1, 0)))
        if self.result is None or self.volume is None:
            self.params.crop_idx = idx
            return self.reaugment()
        # No early return on idx == params.crop_idx: live slider callbacks set
        # params BEFORE calling here, so such a check would always hit and the
        # crop window would never update (only the highlight would). Recompute
        # is cheap (two small crops), so always refresh.
        self.params.crop_idx = idx

        import torch

        keep_idx = self.result.keep_idx
        if keep_idx is None or len(keep_idx) == 0:
            keep_idx = np.arange(self.num_points, dtype=int)
        raw_idx = int(keep_idx[int(np.clip(idx, 0, len(keep_idx) - 1))])

        target_sz = tuple(self.target_sz)
        full_sz = np.array([1, *np.shape(self.volume)])
        raw_crops = extract_crops(np.asarray(self.volume, dtype=np.float32),
                                  np.asarray(self.points, dtype=float), np.array(target_sz))
        self.result.crop_raw = _normalize_crops(raw_crops)[raw_idx]

        aug_crops = extract_crops(self.result.vol_aug, self.result.pts_aug,
                                  np.array(target_sz))
        transform = build_photometric_transform(self.params.photometric_args_dict())
        with torch.no_grad():
            out = transform(torch.from_numpy(aug_crops))
        self.result.crop_aug = np.asarray(
            out if not torch.is_tensor(out) else out.numpy(), dtype=np.float32)[idx]

        _, bbox_raw = get_3d_crop_using_bbox_or_centroid(
            np.asarray(self.points[raw_idx]), full_sz, np.array(target_sz), self.volume)
        _, bbox_aug = get_3d_crop_using_bbox_or_centroid(
            self.result.pts_aug[idx], full_sz, np.array(target_sz), self.result.vol_aug)
        self.result.bbox_raw = list(bbox_raw)
        self.result.bbox_aug = list(bbox_aug)
        return self.result

    def select_neuron_by_seg_id(self, seg_id: int) -> AugmentedFrame:
        """Switch the crop window by raw segmentation id (rather than index).

        Maps the raw index to the kept (post-dropout) position; raises if the
        neuron was dropped from the current augmented view.
        """
        if self.seg_ids is None:
            raise ValueError("No segmentation ids loaded")
        matches = np.where(np.asarray(self.seg_ids) == int(seg_id))[0]
        if len(matches) == 0:
            raise ValueError(f"seg_id {seg_id} not present in frame {self.params.frame}")
        raw_idx = int(matches[0])
        keep_idx = self.result.keep_idx if self.result is not None else None
        if keep_idx is not None and len(keep_idx):
            kept_pos = np.where(np.asarray(keep_idx) == raw_idx)[0]
            if len(kept_pos) == 0:
                raise ValueError(f"seg_id {seg_id} was dropped from the current view; "
                                 f"re-augment or lower dropout_p")
            return self.set_crop_idx(int(kept_pos[0]))
        return self.set_crop_idx(raw_idx)


# --------------------------------------------------------------------------
# Secondary crop-window helper
# --------------------------------------------------------------------------

def set_crop_neuron(state: ViewerState, idx: int, viewer_crop=None,
                    viewer_main=None) -> AugmentedFrame:
    """Change which neuron is shown in the secondary (single-neuron) crop window.

    Parameters
    ----------
    state:
        Active :class:`ViewerState` (holds volume + current augmentation).
    idx:
        Index into the KEPT (post-dropout) augmented view for the new neuron.
        Out-of-range values are clipped (same rule as ``ViewerState``).
    viewer_crop:
        Optional napari crop viewer. When given, its ``crop_raw`` /
        ``crop_aug`` layers are updated in place and the title shows the
        new index (plus raw idx / seg id), so no full refresh (or
        re-augmentation) is needed.
    viewer_main:
        Optional napari main viewer. When given, its ``crop_box_raw`` /
        ``crop_box_aug`` shapes are moved to the new neuron and the selected
        points are enlarged.

    Returns the updated :class:`AugmentedFrame`. Pure-state usage
    (both viewers ``None``) just updates ``state`` and is headless-testable.
    """
    res = state.set_crop_idx(idx)
    if viewer_crop is not None:
        viewer_crop.layers['crop_raw'].data = res.crop_raw
        viewer_crop.layers['crop_aug'].data = res.crop_aug
        try:
            viewer_crop.title = crop_window_title(state)
        except (AttributeError, TypeError):
            pass
    if viewer_main is not None:
        viewer_main.layers['crop_box_raw'].data = box_edges_from_bbox(res.bbox_raw)
        viewer_main.layers['crop_box_aug'].data = box_edges_from_bbox(res.bbox_aug)
        _highlight_selection(viewer_main, state)
    return res


def _highlight_selection(viewer_main, state: ViewerState) -> None:
    """Enlarge the selected points in the main viewer (best-effort, GUI only)."""
    try:
        viewer_main.layers['points_raw'].size = highlight_sizes(
            state.num_points, state.current_raw_idx or 0)
        viewer_main.layers['points_aug'].size = highlight_sizes(
            state.num_kept, int(state.params.crop_idx))
    except (AttributeError, KeyError, TypeError, ValueError):
        pass


def _set_points_text(layer, labels: list) -> None:
    """Set napari points text labels, tolerating API differences."""
    try:
        layer.text.string = labels
        return
    except (AttributeError, TypeError, ValueError):
        pass
    try:
        layer.text = {'string': labels, 'size': 10, 'color': 'white'}
    except (AttributeError, TypeError, ValueError):
        pass


# --------------------------------------------------------------------------
# GUI (napari + magicgui, imported lazily so they stay optional)
# --------------------------------------------------------------------------

def build_control_widget(state: ViewerState, on_reaugment,
                         on_crop_live=None, on_frame_live=None) -> object:
    """magicgui dock widget: enable/disable each transform + vary its params.

    Pressing ``Re-augment`` copies all widget values into ``state.params``
    and calls ``on_reaugment(old_frame)``. The GUI passes a callback that
    refreshes napari layers; headless tests pass one that only updates state.

    ``crop_idx`` / ``frame`` sliders update live (no button press needed):
    moving ``crop_idx`` calls ``on_crop_live(idx)`` (fast crop-only refresh
    that reuses the current augmentation), moving ``frame`` calls
    ``on_frame_live(old_frame)`` (defaults to ``on_reaugment``). Pass
    ``None`` to keep button-only behavior (e.g. in headless tests).

    Layout is two stacked boxes: augmentation parameters with the
    ``Re-augment`` button on top, and the live ``frame`` / ``crop_idx``
    selection sliders below it.
    """
    from magicgui.widgets import (
        CheckBox,
        Container,
        FloatSpinBox,
        Label,
        PushButton,
        Slider,
        SpinBox,
    )

    p = state.params
    w = {}
    _guard = {'active': False}

    def _slider(name, value, min_val, max_val, step=1):
        w[name] = Slider(value=int(value), min=int(min_val), max=int(max_val), step=int(step),
                         label=name)

    def _float(name, value, min_val, max_val, step=0.05):
        w[name] = FloatSpinBox(value=float(value), min=float(min_val), max=float(max_val),
                               step=float(step), label=name)

    # --- Top box: augmentation parameters (applied on Re-augment) ---
    aug_names = []
    # Global affine
    w['use_affine'] = CheckBox(value=p.use_affine, label='use_affine')
    _float('max_degrees_z', p.max_degrees_z, 0.0, 180.0, step=5.0)
    _float('scale_jitter', p.scale_jitter, 0.0, 0.5)
    _float('max_translation_z', p.max_translation_z, 0.0, 10.0, step=1.0)
    _float('max_translation_xy', p.max_translation_xy, 0.0, 32.0, step=1.0)
    _float('p_flip', p.p_flip, 0.0, 1.0, step=0.1)
    # Photometric
    w['use_blur'] = CheckBox(value=p.use_blur, label='use_blur')
    w['use_noise'] = CheckBox(value=p.use_noise, label='use_noise')
    _float('std_noise', p.std_noise, 0.0, 1.0)
    # Position-only (per-view, applied after the global affine like training).
    # use_dropout is the master switch (mirrors the train-config
    # position_augment section); dropout_p is ignored while it is off.
    _float('jitter_z', p.jitter_z, 0.0, 10.0, step=0.5)
    _float('jitter_xy', p.jitter_xy, 0.0, 10.0, step=0.5)
    w['use_dropout'] = CheckBox(value=p.use_dropout, label='use_dropout')
    _float('dropout_p', p.dropout_p, 0.0, 0.9, step=0.05)
    w['min_keep'] = SpinBox(value=int(p.min_keep), min=0, max=max(state.num_points, 2),
                            step=1, label='min_keep')
    w['seed'] = SpinBox(value=p.seed, min=0, max=10000, step=1, label='seed')
    w['reaugment'] = PushButton(label='Re-augment')
    aug_names = [name for name in w]
    # --- Bottom box: live selection (updates the viewers immediately) ---
    max_frame = int(state.project_data.num_frames - 1)
    _slider('frame', p.frame, 0, max_frame)
    _slider('crop_idx', p.crop_idx, 0, max(state.num_kept - 1, 0))
    live_names = ['frame', 'crop_idx']

    def _sync_all():
        for name, widget in w.items():
            if name != 'reaugment':
                setattr(p, name, widget.value)

    def _sync_crop_range():
        # Kept set (post-dropout) drives the slider, not the raw count
        w['crop_idx'].max = max(state.num_kept - 1, 0)
        if w['crop_idx'].value != p.crop_idx:
            _guard['active'] = True
            try:
                w['crop_idx'].value = int(np.clip(p.crop_idx, 0, max(state.num_kept - 1, 0)))
            finally:
                _guard['active'] = False

    def _on_click(*_args):
        old_frame = state.params.frame
        _sync_all()
        _sync_crop_range()
        on_reaugment(old_frame)

    def _on_crop_slider(*_args):
        if _guard['active'] or on_crop_live is None:
            return
        p.crop_idx = int(w['crop_idx'].value)
        on_crop_live(int(p.crop_idx))

    def _on_frame_slider(*_args):
        if _guard['active']:
            return
        old_frame = state.params.frame
        _sync_all()
        _sync_crop_range()
        if on_frame_live is not None:
            on_frame_live(old_frame)
        else:
            on_reaugment(old_frame)

    w['reaugment'].changed.connect(_on_click)
    w['crop_idx'].changed.connect(_on_crop_slider)
    w['frame'].changed.connect(_on_frame_slider)
    aug_box = Container(widgets=[w[name] for name in aug_names],
                        labels=True, label='Augmentation (press Re-augment)')
    live_box = Container(
        widgets=[Label(value='Live selection (updates viewers immediately)'),
                 *[w[name] for name in live_names]],
        labels=True, label='Live selection')
    box = Container(widgets=[aug_box, live_box], labels=False)
    box._widgets = w  # expose for tests / external tweaks
    box._sync_crop_range = _sync_crop_range
    box._guard = _guard
    return box


def _refresh_after_param_change(state: ViewerState, old_frame: int) -> None:
    if state.params.frame != old_frame or state.volume is None:
        state.load_frame(state.params.frame)
    else:
        state.reaugment()


def show_volume_augmentation(project_data, target_sz: Tuple[int, int, int] = DEFAULT_TARGET_SZ,
                             params: Optional[ViewerParams] = None):
    """Open the main + crop napari viewers with a magicgui control dock.

    Returns ``(viewer_main, viewer_crop, state)``. Start the Qt event loop
    afterwards with ``napari.run()`` (or omit it in notebooks/tests).
    """
    import napari

    state = ViewerState(project_data, target_sz=target_sz, params=params)
    res = state.result

    main = napari.Viewer(title='augment: volume', axis_labels=('z', 'x', 'y'))
    main.add_image(state.volume, name='raw', colormap='gray')
    main.add_image(res.vol_aug, name='augmented', colormap='green')
    main.add_points(state.points, name='points_raw', face_color='white',
                    size=highlight_sizes(state.num_points, state.current_raw_idx or 0),
                    text={'string': raw_point_labels(state.num_points, state.seg_ids),
                          'size': 10, 'color': 'white'})
    main.add_points(np.asarray(res.pts_aug), name='points_aug', face_color='magenta',
                    size=highlight_sizes(state.num_kept, int(state.params.crop_idx)),
                    text={'string': aug_point_labels(res.keep_idx, state.seg_ids),
                          'size': 10, 'color': 'magenta'})
    main.add_shapes(box_edges_from_bbox(res.bbox_raw), shape_type='line',
                    name='crop_box_raw', edge_color='white', edge_width=1)
    main.add_shapes(box_edges_from_bbox(res.bbox_aug), shape_type='line',
                    name='crop_box_aug', edge_color='magenta', edge_width=1)

    crop = napari.Viewer(title=crop_window_title(state), axis_labels=('z', 'x', 'y'))
    crop.add_image(res.crop_raw, name='crop_raw', colormap='gray')
    crop.add_image(res.crop_aug, name='crop_aug', colormap='green')

    widget_ref = {}

    def refresh():
        r = state.result
        main.layers['augmented'].data = r.vol_aug
        main.layers['points_raw'].data = state.points
        main.layers['points_aug'].data = np.asarray(r.pts_aug)
        _set_points_text(main.layers['points_raw'],
                         raw_point_labels(state.num_points, state.seg_ids))
        _set_points_text(main.layers['points_aug'],
                         aug_point_labels(r.keep_idx, state.seg_ids))
        _highlight_selection(main, state)
        main.layers['crop_box_raw'].data = box_edges_from_bbox(r.bbox_raw)
        main.layers['crop_box_aug'].data = box_edges_from_bbox(r.bbox_aug)
        crop.layers['crop_raw'].data = r.crop_raw
        crop.layers['crop_aug'].data = r.crop_aug
        try:
            crop.title = crop_window_title(state)
        except (AttributeError, TypeError):
            pass
        w = widget_ref.get('widget')
        if w is not None:
            try:
                sync = getattr(w, '_sync_crop_range', None)
                if sync is not None:
                    sync()
                else:
                    w._widgets['crop_idx'].max = max(state.num_kept - 1, 0)
            except (AttributeError, KeyError, TypeError):
                pass

    def refresh_crop_only(idx: int):
        # Fast path for the crop_idx slider: no re-augmentation, just move
        # the crop window + boxes + selection highlight.
        set_crop_neuron(state, idx, viewer_crop=crop, viewer_main=main)

    def on_reaugment(old_frame: int):
        _refresh_after_param_change(state, old_frame)
        refresh()

    widget = build_control_widget(state, on_reaugment,
                                  on_crop_live=refresh_crop_only,
                                  on_frame_live=on_reaugment)
    widget_ref['widget'] = widget
    main.window.add_dock_widget(widget, name='augmentation controls')

    # Keep a reference so the widget is not garbage-collected
    _LIVE_WIDGETS.append(widget)
    return main, crop, state
