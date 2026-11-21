import functools
import re

import mne
import numpy as np
import sklearn.metrics

from ..data import get_leadfield_cov_path, get_leadfield_npz_path
from ..utils._checks import _check_type
from ..utils.logs import logger

_AUX_LABEL_TOKENS = frozenset(
    {
        "eog",
        "veog",
        "heog",
        "ecg",
        "ekg",
        "emg",
        "misc",
        "stim",
        "status",
        "trigger",
        "aux",
        "resp",
        "gsr",
        "ppg",
        "acc",
        "gyro",
        "bip",
    }
)
_REF_LABEL_TOKENS = frozenset({"ref", "avg", "ar", "le", "m1", "m2", "a1", "a2", "cz"})
_LINKED_REF_PAIRS = {
    "m1m2": ("M1", "M2", "M1+M2(avg)"),
    "tp9tp10": ("TP9", "TP10", "TP9+TP10(avg)"),
    "a1a2": ("A1", "A2", "A1+A2(avg)"),
}


def _match_template_labels(
    ch_names: list[str] | tuple[str, ...],
    template_names: list[str] | tuple[str, ...],
) -> list[int]:
    """Match channel names to template labels without substring collisions.

    Performs a two-pass match matching MATLAB ``GEDAI_match_template_labels``:
    Pass 1 handles exact case-insensitive trimmed matches. Pass 2 strips
    common BIDS/EDF channel prefixes (``EEG``) and reference suffixes
    (``REF``, ``LE``, ``AR``, ``M1``, ``M2``, ``A1``, ``A2``, ``Cz``) while
    skipping auxiliary channels (e.g., ``EOG_Fp1``, ``ECG``) and avoiding
    substring collisions (e.g., ``C10`` vs ``C1``, ``AFp1`` vs ``Fp1``).

    Returns
    -------
    indices : list of int
        Index into ``template_names`` for each entry in ``ch_names``, or
        ``-1`` where no unambiguous match exists.
    """
    n_ch = len(ch_names)
    indices = [-1] * n_ch
    tmpl_lower = [str(name).strip().lower() for name in template_names]
    tmpl_map: dict[str, int] = {}
    for j, t_low in enumerate(tmpl_lower):
        if t_low not in tmpl_map:
            tmpl_map[t_low] = j
    used_tmpl: set[int] = set()

    # Pass 1: Exact case-insensitive trimmed matches
    for i, raw_lbl in enumerate(ch_names):
        lbl = str(raw_lbl).strip().lower()
        if lbl in tmpl_map:
            j = tmpl_map[lbl]
            indices[i] = j
            used_tmpl.add(j)

    # Pass 2: Whole-token matching for BIDS/EDF prefixed/referenced labels
    for i, raw_lbl in enumerate(ch_names):
        if indices[i] >= 0:
            continue
        lbl = str(raw_lbl).strip().lower()
        tokens = re.findall(r"[a-z0-9]+", lbl)
        if not tokens:
            continue
        if any(tok in _AUX_LABEL_TOKENS for tok in tokens):
            continue
        if len(tokens) >= 2 and tokens[0] == "eeg":
            tokens = tokens[1:]
        if len(tokens) >= 2 and tokens[-1] in _REF_LABEL_TOKENS:
            tokens = tokens[:-1]
        if not tokens:
            continue
        candidate = "".join(tokens)
        j = tmpl_map.get(candidate)
        if j is not None and j not in used_tmpl:
            indices[i] = j
            used_tmpl.add(j)

    return indices


@functools.lru_cache(maxsize=1)
def _load_template_leadfield() -> dict:
    """Load and cache 343-channel BEM leadfield data and dipole projection matrices."""
    with np.load(get_leadfield_npz_path(), allow_pickle=False) as data:
        gain_raw = np.asarray(data["gain"], dtype=np.float64)
        grid_loc = np.asarray(data["grid_loc"], dtype=np.float64)
        elec_loc = np.asarray(data["elec_loc"], dtype=np.float64)
        elec_names = [str(s) for s in data["elec_names"]]

    gain_av = gain_raw - gain_raw.mean(axis=0, keepdims=True)
    name_map = {name.lower(): idx for idx, name in enumerate(elec_names)}

    # Canonical head sphere fit and ALS head-frame orthonormal basis on 343 template
    a_loc = np.column_stack([2.0 * elec_loc, np.ones(len(elec_loc))])
    sol_loc, *_ = np.linalg.lstsq(a_loc, np.sum(elec_loc**2, axis=1), rcond=None)
    c_loc = sol_loc[:3]

    vx = elec_loc[name_map["fpz"]] - elec_loc[name_map["oz"]]
    ex = vx / np.linalg.norm(vx)
    vy = elec_loc[name_map["t7"]] - elec_loc[name_map["t8"]]
    ey = vy - np.dot(vy, ex) * ex
    ey = ey / np.linalg.norm(ey)
    ez = np.cross(ex, ey)
    if np.dot(elec_loc[name_map["cz"]] - c_loc, ez) < 0:
        ez = -ez
    r_canon_to_343 = np.vstack([ex, ey, ez])

    u_343_sph = (elec_loc - c_loc) @ r_canon_to_343.T
    u_343_sph /= np.maximum(np.linalg.norm(u_343_sph, axis=1, keepdims=True), 1e-12)
    r_343_norm = np.linalg.norm(elec_loc - c_loc, axis=1)

    grid_1 = c_loc + 0.65 * (grid_loc - c_loc)
    grid_2 = c_loc + 0.90 * (grid_loc - c_loc)
    k_343_raw = np.hstack(
        [
            _dipole_kernel_matrix(elec_loc, grid_1),
            _dipole_kernel_matrix(elec_loc, grid_2),
        ]
    )
    k_343_av = k_343_raw - k_343_raw.mean(axis=0, keepdims=True)

    lam = 1e-4
    n_s = k_343_raw.shape[0]
    gram_av = k_343_av @ k_343_av.T
    reg_av = lam * (np.trace(gram_av) / n_s)
    b_av = np.linalg.solve(gram_av + reg_av * np.eye(n_s), gain_av)

    gram_raw = k_343_raw @ k_343_raw.T
    reg_raw = lam * (np.trace(gram_raw) / n_s)
    b_raw = np.linalg.solve(gram_raw + reg_raw * np.eye(n_s), gain_raw)

    return {
        "gain_raw": gain_raw,
        "gain_av": gain_av,
        "grid_loc": grid_loc,
        "elec_loc": elec_loc,
        "elec_names": elec_names,
        "name_map": name_map,
        "c_loc": c_loc,
        "r_canon_to_343": r_canon_to_343,
        "u_343_sph": u_343_sph,
        "r_343_norm": r_343_norm,
        "grid_1": grid_1,
        "grid_2": grid_2,
        "k_343_raw": k_343_raw,
        "k_343_av": k_343_av,
        "b_raw": b_raw,
        "b_av": b_av,
    }


def _dipole_kernel_matrix(sens_loc: np.ndarray, dip_loc: np.ndarray) -> np.ndarray:
    """Compute homogeneous infinite-medium dipole potential kernel matrix."""
    dx = sens_loc[:, 0:1] - dip_loc[:, 0:1].T
    dy = sens_loc[:, 1:2] - dip_loc[:, 1:2].T
    dz = sens_loc[:, 2:3] - dip_loc[:, 2:3].T
    dist2 = dx * dx + dy * dy + dz * dz
    dist3 = np.maximum(dist2, 0.015**2) ** 1.5
    n_s = sens_loc.shape[0]
    n_d = dip_loc.shape[0]
    k_mat = np.zeros((n_s, 3 * n_d), dtype=np.float64)
    k_mat[:, 0::3] = dx / dist3
    k_mat[:, 1::3] = dy / dist3
    k_mat[:, 2::3] = dz / dist3
    return k_mat


def _has_valid_3d_locations(info: mne.Info) -> bool:
    """Return True if all channels in ``info`` have finite non-zero 3D coords."""
    if info["nchan"] == 0:
        return False
    for ch in info["chs"]:
        loc = np.asarray(ch["loc"][:3], dtype=np.float64)
        if not np.all(np.isfinite(loc)) or np.allclose(loc, 0.0):
            return False
    return True


def _extract_target_unit_directions_als(info: mne.Info) -> np.ndarray:
    """Extract normalized head-sphere sensor unit directions in canonical ALS frame."""
    if not _has_valid_3d_locations(info):
        missing = [
            ch["ch_name"]
            for ch in info["chs"]
            if not np.all(np.isfinite(ch["loc"][:3]))
            or np.allclose(ch["loc"][:3], 0.0)
        ]
        raise ValueError(
            "Valid 3D channel positions (X, Y, Z) are required for all channels to "
            f"compute dipole-kernel leadfield projection; missing/zero for {missing}."
        )

    xyz_ras = np.array([ch["loc"][:3] for ch in info["chs"]], dtype=np.float64)
    # MNE head frame is RAS (+X=Right, +Y=Anterior, +Z=Superior);
    # template ALS frame is (+X=Anterior, +Y=Left, +Z=Superior).
    xyz_als = np.column_stack([xyz_ras[:, 1], -xyz_ras[:, 0], xyz_ras[:, 2]])
    norms = np.linalg.norm(xyz_als, axis=1)
    med_r = float(np.median(norms))
    if med_r > 0.5:
        xyz_als = xyz_als * (0.095 / med_r)

    c_target = np.zeros(3, dtype=np.float64)
    if len(xyz_als) >= 6:
        a_mat = np.column_stack([2.0 * xyz_als, np.ones(len(xyz_als))])
        s_vals = np.linalg.svd(a_mat, compute_uv=False)
        if s_vals[-1] > 0.0 and (s_vals[0] / s_vals[-1]) < 100.0:
            sol, *_ = np.linalg.lstsq(a_mat, np.sum(xyz_als**2, axis=1), rcond=None)
            c_fit = sol[:3]
            r_fit = float(np.sqrt(max(0.0, sol[3] + np.dot(c_fit, c_fit))))
            if 0.05 <= r_fit <= 0.14 and float(np.linalg.norm(c_fit)) <= 0.75 * r_fit:
                c_target = c_fit

    u_target = xyz_als - c_target
    u_norms = np.maximum(np.linalg.norm(u_target, axis=1, keepdims=True), 1e-12)
    return u_target / u_norms


def project_leadfield_dipole(info: mne.Info, use_avgref: bool = True) -> np.ndarray:
    """Project the 343-channel OpenMEEG BEM leadfield onto arbitrary 3D sensor sites.

    Uses canonical 10-20 ALS head-frame alignment and multi-eccentricity (0.65 / 0.90)
    Berg cortical dipole-kernel ridge regression.

    Parameters
    ----------
    info : mne.Info
        Channel info containing valid 3D electrode positions in MNE head coordinates.
    use_avgref : bool, default=True
        Whether to use the average-referenced template gain solver (``True``)
        or the raw unreferenced template gain solver (``False``).

    Returns
    -------
    gain_proj : np.ndarray
        Projected EEG leadfield gain matrix of shape ``(n_channels, 3630)``
        matching channels in ``info``.
    """
    _check_type(info, (mne.Info,), "info")
    cache = _load_template_leadfield()
    u_target = _extract_target_unit_directions_als(info)

    cos_ang = np.clip(u_target @ cache["u_343_sph"].T, -1.0, 1.0)
    ang = np.arccos(cos_ang)
    w_rad = np.exp(-((ang / 0.12) ** 2))
    w_rad /= np.maximum(w_rad.sum(axis=1, keepdims=True), 1e-12)
    r_target_norm = w_rad @ cache["r_343_norm"]

    dir_343_space = u_target @ cache["r_canon_to_343"]
    loc_target = cache["c_loc"] + dir_343_space * r_target_norm[:, None]

    k_target = np.hstack(
        [
            _dipole_kernel_matrix(loc_target, cache["grid_1"]),
            _dipole_kernel_matrix(loc_target, cache["grid_2"]),
        ]
    )
    if use_avgref:
        return (k_target @ cache["k_343_av"].T) @ cache["b_av"]
    return (k_target @ cache["k_343_raw"].T) @ cache["b_raw"]


def _normalize_reference_mode(reference: str | None) -> str:
    """Normalize EEG reference specifier into a canonical form."""
    if reference is None:
        return "average"
    if not isinstance(reference, str):
        raise TypeError(
            f"reference must be a string or None, got {type(reference).__name__}."
        )
    mode = reference.strip()
    if not mode:
        return "average"
    low = mode.lower()
    if low in ("average", "avg", "avgref", "average-reference"):
        return "average"
    if low == "rest":
        return "REST"
    if low in ("tp9tp10", "tp9+tp10", "tp9+tp10(avg)", "linked_tp"):
        return "TP9TP10"
    if low in ("m1m2", "m1+m2", "m1+m2(avg)", "linked_mastoids"):
        return "M1M2"
    if low in ("a1a2", "a1+a2", "a1+a2(avg)", "linked_earlobes"):
        return "A1A2"
    if low in ("none", "raw"):
        return "none"
    return mode


def _lookup_template_gain_row(label: str) -> tuple[bool, np.ndarray | None, str]:
    """Look up a standard 10-05 channel or linked reference pair in the 343 template."""
    cache = _load_template_leadfield()
    gain = cache["gain_raw"]
    name_map = cache["name_map"]
    elec_names = cache["elec_names"]
    low = label.strip().lower()

    if low in _LINKED_REF_PAIRS:
        c1, c2, _ = _LINKED_REF_PAIRS[low]
        i1 = name_map.get(c1.lower())
        i2 = name_map.get(c2.lower())
        if i1 is not None and i2 is not None:
            return True, 0.5 * (gain[i1] + gain[i2]), label
        return False, None, label

    idx = name_map.get(low)
    if idx is not None:
        return True, gain[idx], elec_names[idx]

    matched = _match_template_labels([label], elec_names)
    if matched[0] >= 0:
        j = matched[0]
        return True, gain[j], elec_names[j]
    return False, None, label


def _find_channel_indices(ch_names: list[str], targets: tuple[str, ...]) -> list[int]:
    """Find channel indices by exact match first, falling back to token-safe match."""
    lower_names = [str(c).strip().lower() for c in ch_names]
    out: list[int] = []
    for req in targets:
        req_low = req.strip().lower()
        exact = [idx for idx, nm in enumerate(lower_names) if nm == req_low]
        if exact:
            out.append(exact[0])
            continue
        tok_idx = _match_template_labels(ch_names, [req])
        found = [idx for idx, m in enumerate(tok_idx) if m == 0]
        out.append(found[0] if found else -1)
    return out


def _create_reference_spec(info: mne.Info, reference: str | None = "average") -> dict:
    """Create linear EEG reference operator specification matching MATLAB GEDAI."""
    mode = _normalize_reference_mode(reference)
    ch_names = list(info["ch_names"])
    n_ch = len(ch_names)
    spec = {
        "mode": mode,
        "label": mode,
        "is_rest": False,
        "is_external_ref": False,
        "external_ref_label": "",
        "R": np.eye(n_ch, dtype=np.float64),
        "rest_operator": None,
        "required_indices": [],
        "required_channels": [],
    }
    low = mode.lower()
    if low == "none":
        spec["label"] = "none"
        return spec
    if low == "average":
        spec["R"] = np.eye(n_ch, dtype=np.float64) - np.ones(
            (n_ch, n_ch), dtype=np.float64
        ) / max(n_ch, 1)
        spec["label"] = "average"
        return spec
    if low == "rest":
        spec["is_rest"] = True
        spec["label"] = "REST"
        return spec
    if low in _LINKED_REF_PAIRS:
        c1, c2, lbl = _LINKED_REF_PAIRS[low]
        idxs = _find_channel_indices(ch_names, (c1, c2))
        if all(idx >= 0 for idx in idxs):
            w = np.zeros(n_ch, dtype=np.float64)
            w[idxs[0]] = 0.5
            w[idxs[1]] = 0.5
            spec["R"] = np.eye(n_ch, dtype=np.float64) - np.outer(
                np.ones(n_ch, dtype=np.float64), w
            )
            spec["label"] = lbl
            spec["required_indices"] = list(idxs)
            spec["required_channels"] = [ch_names[k] for k in idxs]
        else:
            spec["is_external_ref"] = True
            spec["external_ref_label"] = mode
            spec["label"] = lbl
        return spec

    # Single-channel reference (in-montage or external template channel)
    idxs = _find_channel_indices(ch_names, (mode,))
    if idxs[0] >= 0:
        idx = idxs[0]
        w = np.zeros(n_ch, dtype=np.float64)
        w[idx] = 1.0
        spec["R"] = np.eye(n_ch, dtype=np.float64) - np.outer(
            np.ones(n_ch, dtype=np.float64), w
        )
        spec["label"] = ch_names[idx]
        spec["required_indices"] = [idx]
        spec["required_channels"] = [ch_names[idx]]
        return spec

    is_std, _, std_name = _lookup_template_gain_row(mode)
    if is_std:
        spec["is_external_ref"] = True
        spec["external_ref_label"] = std_name
        spec["label"] = std_name
        return spec

    raise ValueError(
        f"Required reference channel '{mode}' was not found in dataset channels "
        "or standard 10-05 template montages."
    )


def _compute_rest_operator(gain_raw: np.ndarray, rcond: float = 1e-5) -> np.ndarray:
    """Compute infinity-reference REST linear re-referencing operator from leadfield.

    Implements Yao (2001) with scale-invariant SVD truncation so REST behaves
    identically regardless of overall gain matrix scaling.
    """
    gain_raw = np.asarray(gain_raw, dtype=np.float64)
    n_ch = gain_raw.shape[0]
    r_avg = np.eye(n_ch, dtype=np.float64) - (
        np.ones((n_ch, n_ch), dtype=np.float64) / n_ch
    )
    g_ar = gain_raw - gain_raw.mean(axis=0, keepdims=True)
    u_mat, s_vals, vt_mat = np.linalg.svd(g_ar, full_matrices=False)
    if s_vals.size == 0 or s_vals[0] <= 0.0:
        return r_avg
    keep = s_vals > (rcond * s_vals[0])
    if not np.any(keep):
        return r_avg
    t_rest = (gain_raw @ vt_mat[keep].T) @ (u_mat[:, keep].T / s_vals[keep][:, None])
    offset = np.ones((n_ch, 1), dtype=np.float64) @ t_rest.mean(axis=0, keepdims=True)
    return (np.eye(n_ch, dtype=np.float64) + offset) @ r_avg


def _apply_leadfield_reference(
    gain_raw: np.ndarray,
    spec: dict,
    forward_gain: np.ndarray | None = None,
    forward_names: list[str] | None = None,
) -> tuple[np.ndarray, dict]:
    """Transform a raw gain matrix according to ``spec`` and populate REST operator."""
    out_spec = dict(spec)
    if out_spec["is_rest"]:
        out_spec["rest_operator"] = _compute_rest_operator(gain_raw)
        return gain_raw, out_spec
    if out_spec["is_external_ref"]:
        ext_lbl = out_spec["external_ref_label"]
        if forward_gain is not None and forward_names is not None:
            idxs = _find_channel_indices(forward_names, (ext_lbl,))
            if idxs[0] >= 0:
                g_std = forward_gain[idxs[0]]
                return gain_raw - g_std[None, :], out_spec
            raise ValueError(
                f"External reference '{ext_lbl}' was not found in the supplied Forward."
            )
        is_std, g_std, _ = _lookup_template_gain_row(ext_lbl)
        if not is_std or g_std is None:
            raise ValueError(
                f"Could not locate template leadfield row for external reference "
                f"'{ext_lbl}'."
            )
        return gain_raw - g_std[None, :], out_spec
    return out_spec["R"] @ gain_raw, out_spec


def _cov_from_gain(
    gain: np.ndarray, ch_names: list[str], bads: list[str]
) -> mne.Covariance:
    """Create a symmetric trace-normalized MNE Covariance from a gain matrix."""
    gram = gain @ gain.T
    gram = np.real((gram + gram.T) * 0.5)
    return mne.Covariance(
        gram, names=list(ch_names), bads=list(bads), projs=[], nfree=len(ch_names)
    )


def compute_covariance_from_dipole_projection(
    info: mne.Info, reference: str | None = "average"
) -> mne.Covariance:
    """Compute EEG reference covariance via 3D cortical dipole-kernel projection.

    Projects the 343-channel OpenMEEG BEM leadfield onto the channel positions
    in ``info`` and applies the requested reference algebra.

    Parameters
    ----------
    info : mne.Info
        MNE Info containing valid 3D channel coordinates.
    reference : str | None, default="average"
        Reference mode (``"average"``, ``"REST"``, ``"M1+M2"``, ``"TP9+TP10"``,
        ``"A1+A2"``, ``"none"``, or a channel label such as ``"Cz"``).

    Returns
    -------
    cov : mne.Covariance
        Reference covariance matrix in ``info`` channel order and reference space.
    """
    _check_type(info, (mne.Info,), "info")
    spec = _create_reference_spec(info, reference=reference)
    mode_low = spec["mode"].lower()
    if mode_low == "average":
        g_ref = project_leadfield_dipole(info, use_avgref=True)
    else:
        g_raw = project_leadfield_dipole(info, use_avgref=False)
        g_ref, _ = _apply_leadfield_reference(g_raw, spec)
    return _cov_from_gain(g_ref, list(info["ch_names"]), list(info["bads"]))


def _ensure_cov(reference_cov):
    _check_type(reference_cov, (str, mne.Covariance, mne.Forward), "reference_cov")
    if isinstance(reference_cov, mne.Forward):
        return compute_covariance_from_forward(reference_cov)
    if isinstance(reference_cov, str):
        norm = reference_cov.strip().lower()
        if norm == "leadfield":
            return mne.read_cov(str(get_leadfield_cov_path()))
        if norm in ("projection", "projected", "dipole"):
            return "projection"
        raise ValueError(
            "Reference covariance must be 'leadfield', 'projection', "
            "an mne.Covariance, or an mne.Forward instance; "
            f"got '{reference_cov}' instead."
        )
    return reference_cov


def _is_meg_info(info: mne.Info | None) -> bool:
    if info is None:
        return False
    ch_types = info.get_channel_types(unique=True)
    return any(t in ("mag", "grad", "ref_meg") for t in ch_types)


def _pick_cov(cov, ch_names, info=None):
    """Subset and reorder a Covariance to match ``ch_names`` using token-safe lookup."""
    if isinstance(ch_names, mne.Info):
        info = ch_names
        ch_names = info["ch_names"]
    elif hasattr(ch_names, "info"):
        info = ch_names.info
        ch_names = info["ch_names"]

    ch_names = list(ch_names)
    cov_ch_names = list(cov.ch_names)
    matched = _match_template_labels(ch_names, cov_ch_names)

    picks_cov = [cov_ch_names[j] for j in matched if j >= 0]
    picks_ch_names = [ch_names[i] for i, j in enumerate(matched) if j >= 0]

    if len(picks_cov) == 0:
        msg = (
            "No matching channel names found between inst and cov.\n"
            f"Available channels in covariance are {cov_ch_names}.\n"
            f"but instance has channels {ch_names}."
        )
        if _is_meg_info(info):
            msg += (
                "\nNote: If you are processing MEG data ('mag' or 'grad'), "
                "the default 'leadfield' bundled with GEDAI is an EEG leadfield. "
                "For MEG data, please provide an MEG forward model "
                "(mne.Forward) or reference covariance (mne.Covariance) via "
                "the 'reference_cov' argument."
            )
        raise ValueError(msg)
    if len(picks_cov) < len(ch_names):
        raise ValueError(
            "Only a subset of channels in the instance are present"
            " in the covariance.\n"
            f"Use inst.pick_channels({picks_ch_names}) to select only the channels"
            f" that are in the covariance or provide a covariance that contains"
            f" all channels in the instance."
        )
    cov = cov.copy().pick_channels(picks_cov, ordered=True)
    if info is not None:
        ch_set = set(ch_names)
        remaining_bads = [b for b in info["bads"] if b in ch_set]
    else:
        cov_to_inst = dict(zip(picks_cov, ch_names, strict=False))
        remaining_bads = [
            cov_to_inst[b] for b in cov.get("bads", []) if b in cov_to_inst
        ]
    cov.update(names=ch_names, bads=remaining_bads)
    return cov


def _resolve_reference_and_cov(
    reference_cov: str | mne.Covariance | mne.Forward,
    info: mne.Info,
    reference: str | None = "average",
) -> tuple[mne.Covariance, dict]:
    """Resolve both reference covariance and EEG reference operator for ``info``.

    Supports precomputed 343-ch BEM leadfield (with automatic 3D dipole-kernel
    fallback for custom/non-10-20 montages), explicit ``"projection"``,
    ``mne.Forward``, and custom ``mne.Covariance``.
    """
    _check_type(
        reference_cov, (str, mne.Covariance, mne.Forward), "reference_cov"
    )
    ch_names = list(info["ch_names"])
    bads = list(info["bads"])

    # MEG is reference-free; bypass EEG reference operators
    if _is_meg_info(info):
        spec = {
            "mode": "none",
            "label": "none",
            "is_rest": False,
            "is_external_ref": False,
            "external_ref_label": "",
            "R": np.eye(len(ch_names), dtype=np.float64),
            "rest_operator": None,
            "required_indices": [],
            "required_channels": [],
        }
        if isinstance(reference_cov, mne.Forward):
            cov = compute_covariance_from_forward(reference_cov)
            return _pick_cov(cov, ch_names, info=info), spec
        if isinstance(reference_cov, mne.Covariance):
            return _pick_cov(reference_cov, ch_names, info=info), spec
        # Triggers informative MEG error message on 'leadfield'
        cov_default = _ensure_cov(reference_cov)
        if isinstance(cov_default, str):
            raise ValueError(
                "3D EEG dipole projection cannot be used with MEG channels. "
                "Provide an MEG mne.Forward or mne.Covariance."
            )
        return _pick_cov(cov_default, ch_names, info=info), spec

    spec = _create_reference_spec(info, reference=reference)
    mode_low = spec["mode"].lower()

    if isinstance(reference_cov, str):
        norm = reference_cov.strip().lower()
        if norm in ("projection", "projected", "dipole"):
            if mode_low == "average":
                g_ref = project_leadfield_dipole(info, use_avgref=True)
            else:
                g_raw = project_leadfield_dipole(info, use_avgref=False)
                g_ref, spec = _apply_leadfield_reference(g_raw, spec)
            return _cov_from_gain(g_ref, ch_names, bads), spec

        if norm != "leadfield":
            raise ValueError(
                "Reference covariance must be 'leadfield', 'projection', "
                "an mne.Covariance, or an mne.Forward instance; "
                f"got '{reference_cov}' instead."
            )

        cache = _load_template_leadfield()
        matched = _match_template_labels(ch_names, cache["elec_names"])
        if all(j >= 0 for j in matched):
            if mode_low == "average":
                cov_fif = _ensure_cov("leadfield")
                return _pick_cov(cov_fif, ch_names, info=info), spec
            g_raw = cache["gain_raw"][matched, :]
            g_ref, spec = _apply_leadfield_reference(g_raw, spec)
            return _cov_from_gain(g_ref, ch_names, bads), spec

        if _has_valid_3d_locations(info):
            unmatched = [ch_names[i] for i, j in enumerate(matched) if j < 0]
            logger.info(
                "Channels %s did not match standard 10-05 template labels; "
                "using 3D dipole-kernel leadfield projection.",
                unmatched,
            )
            if mode_low == "average":
                g_ref = project_leadfield_dipole(info, use_avgref=True)
            else:
                g_raw = project_leadfield_dipole(info, use_avgref=False)
                g_ref, spec = _apply_leadfield_reference(g_raw, spec)
            return _cov_from_gain(g_ref, ch_names, bads), spec

        # Raise the standard informative channel-mismatch error
        cov_fif = _ensure_cov("leadfield")
        return _pick_cov(cov_fif, ch_names, info=info), spec

    if isinstance(reference_cov, mne.Forward):
        if reference_cov["coord_frame"] != mne._fiff.constants.FIFF.FIFFV_COORD_HEAD:
            raise ValueError("Forward solution must be in head coordinates.")
        fwd_names = list(reference_cov["info"]["ch_names"])
        fwd_gain = np.asarray(reference_cov["sol"]["data"], dtype=np.float64)
        matched = _match_template_labels(ch_names, fwd_names)
        if any(j < 0 for j in matched):
            cov_full = compute_covariance_from_forward(reference_cov)
            cov_picked = _pick_cov(cov_full, ch_names, info=info)
        else:
            g_raw = fwd_gain[matched, :]
            g_ref, spec = _apply_leadfield_reference(
                g_raw, spec, forward_gain=fwd_gain, forward_names=fwd_names
            )
            cov_picked = _cov_from_gain(g_ref, ch_names, bads)
        return cov_picked, spec

    # Custom mne.Covariance
    cov_picked = _pick_cov(reference_cov, ch_names, info=info)
    if spec["is_rest"] or spec["is_external_ref"]:
        raise ValueError(
            f"Reference mode '{spec['mode']}' requires a leadfield gain matrix "
            "(reference_cov='leadfield', 'projection', or an mne.Forward), "
            "not a precomputed mne.Covariance."
        )
    if mode_low not in ("average", "none"):
        r_mat = spec["R"]
        data = np.real(r_mat @ cov_picked.data @ r_mat.T)
        data = (data + data.T) * 0.5
        cov_picked = cov_picked.copy()
        cov_picked.update(data=data)
    return cov_picked, spec


def compute_covariance_from_forward(forward, reference: str | None = "average"):
    """Compute covariance matrix from the leadfield of a forward solution.

    For EEG forwards, applies the requested linear reference operator
    (default ``"average"``) so the resulting covariance matches the EEG
    reference space used during GEDAI denoising. MEG forwards and
    ``reference="none"`` use the raw unreferenced leadfield ``G @ G.T``.

    Parameters
    ----------
    forward : mne.Forward
        The forward solution from which to compute the covariance matrix.
    reference : str | None, default="average"
        EEG reference mode applied to the forward gain matrix (ignored for MEG).
        Use ``"none"`` to leave an EEG forward unreferenced.

    Returns
    -------
    cov : mne.Covariance
        The computed covariance matrix.
    """
    _check_type(forward, (mne.Forward,), "forward")
    if forward["coord_frame"] != mne._fiff.constants.FIFF.FIFFV_COORD_HEAD:
        raise ValueError("Forward solution must be in head coordinates.")
    gain = np.asarray(forward["sol"]["data"], dtype=np.float64)
    info = forward["info"]
    if _is_meg_info(info) or _normalize_reference_mode(reference).lower() == "none":
        data = gain @ gain.T
    else:
        spec = _create_reference_spec(info, reference=reference)
        g_ref, _ = _apply_leadfield_reference(
            gain,
            spec,
            forward_gain=gain,
            forward_names=list(info["ch_names"]),
        )
        data = g_ref @ g_ref.T
    data = np.real((data + data.T) * 0.5)
    ch_names = info["ch_names"]
    bads = info["bads"]
    nfree = len(ch_names)
    cov = mne.Covariance(
        data, names=ch_names, bads=bads, projs=[], nfree=nfree, verbose=None
    )
    return cov


def compute_covariance_from_channel_positions(info):
    """Compute covariance matrix from channel positions.

    Parameters
    ----------
    info : mne.Info
        The info structure containing channel information.

    Returns
    -------
    cov : mne.Covariance
        The computed covariance matrix.
    """
    ch_positions = [info["chs"][i]["loc"][:3] for i in range(info["nchan"])]
    ch_distance_matrix = sklearn.metrics.pairwise_distances(
        ch_positions, metric="euclidean"
    )
    nonzero = ch_distance_matrix[ch_distance_matrix > 0]
    ell = np.median(nonzero) if nonzero.size else 1.0
    sigma2 = 1.0
    eps = 1e-6

    data = sigma2 * np.exp(-(ch_distance_matrix**2) / (2 * ell**2))
    data += eps * np.eye(data.shape[0])

    ch_names = info["ch_names"]
    bads = info["bads"]
    nfree = len(ch_names)
    cov = mne.Covariance(data, ch_names, bads, nfree=nfree, projs=[], verbose=None)
    return cov
