from __future__ import annotations
import pathlib
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import yaml
from PIL import Image
from PIL.Image import Transpose
from yamldataclassconfig.config import YamlDataClassConfig

from . import Raceline
from .cubic_spline import CubicSpline2D
from .utils import find_track_dir


@dataclass
class TrackSpec(YamlDataClassConfig):
    name: str
    image: str
    resolution: float
    origin: Tuple[float, float, float]
    negate: int
    occupied_thresh: float
    free_thresh: float


@dataclass
class Track:
    spec: TrackSpec
    filepath: str
    ext: str
    occupancy_map: np.ndarray
    centerline: Raceline
    raceline: Raceline

    def __init__(
        self,
        spec: TrackSpec,
        filepath: str,
        ext: str,
        occupancy_map: np.ndarray,
        centerline: Optional[Raceline] = None,
        raceline: Optional[Raceline] = None,
    ):
        """
        Initialize track object.

        Parameters
        ----------
        spec : TrackSpec
            track specification
        filepath : str
            path to the track image
        ext : str
            file extension of the track image file
        occupancy_map : np.ndarray
            occupancy grid map
        centerline : Raceline, optional
            centerline of the track, by default None
        raceline : Raceline, optional
            raceline of the track, by default None
        """
        self.spec = spec
        self.filepath = filepath
        self.ext = ext
        self.occupancy_map = occupancy_map
        self.centerline = centerline
        self.raceline = raceline

    @staticmethod
    def load_spec(track: str, filespec: str) -> TrackSpec:
        """
        Load track specification from yaml file.

        Parameters
        ----------
        track : str
            name of the track
        filespec : str
            path to the yaml file

        Returns
        -------
        TrackSpec
            track specification
        """
        with open(filespec, "r") as yaml_stream:
            map_metadata = yaml.safe_load(yaml_stream)
        
        # Create TrackSpec with all required fields
        track_spec = TrackSpec(
            name=track,
            image=map_metadata.get('image', ''),
            resolution=map_metadata.get('resolution', 0.0),
            origin=map_metadata.get('origin', [0.0, 0.0, 0.0]),
            negate=map_metadata.get('negate', 0),
            occupied_thresh=map_metadata.get('occupied_thresh', 0.45),
            free_thresh=map_metadata.get('free_thresh', 0.196)
        )
        return track_spec

    @staticmethod
    def from_track_name(track: str):
        """
        Load track from track name or full path.

        Parameters
        ----------
        track : str
            Either the name of the track (e.g. "Catalunya") or
            a full path to the track directory.

        Returns
        -------
        Track
            Track object

        Raises
        ------
        FileNotFoundError
            If the track cannot be loaded
        """
        import os
        import pathlib

        try:
            # ✅ Case 1: Direct path to map folder (absolute or relative)
            if os.path.isabs(track) or os.path.exists(track):
                track_dir = pathlib.Path(track)
            else:
                # ✅ Case 2: Fallback to default lookup
                track_dir = find_track_dir(track)

            # ✅ Try multiple possible YAML filenames
            yaml_candidates = [
                track_dir / f"{track_dir.stem}_map.yaml",
                track_dir / f"{track_dir.stem}.yaml",
            ]

            yaml_file = None
            for candidate in yaml_candidates:
                if candidate.exists():
                    yaml_file = candidate
                    break

            if yaml_file is None:
                raise FileNotFoundError(
                    f"No valid track YAML found. Checked: {yaml_candidates}"
                )

            # Load YAML metadata directly
            with open(yaml_file, "r") as yaml_stream:
                map_metadata = yaml.safe_load(yaml_stream)
            
            # Create a simple track spec object
            class SimpleTrackSpec:
                def __init__(self, name, metadata):
                    self.name = name
                    self.image = metadata.get('image', '')
                    self.resolution = metadata.get('resolution', 0.0)
                    self.origin = metadata.get('origin', [0.0, 0.0, 0.0])
                    self.negate = metadata.get('negate', 0)
                    self.occupied_thresh = metadata.get('occupied_thresh', 0.45)
                    self.free_thresh = metadata.get('free_thresh', 0.196)
            
            track_spec = SimpleTrackSpec(track_dir.stem, map_metadata)

            # Load occupancy map
            map_filename = pathlib.Path(track_spec.image)
            image_path = track_dir / map_filename
            image = Image.open(image_path).transpose(Transpose.FLIP_TOP_BOTTOM)
            occupancy_map = np.array(image).astype(np.float32)
            occupancy_map[occupancy_map <= 128] = 0.0
            occupancy_map[occupancy_map > 128] = 255.0

            # Load centerline
            centerline_path = track_dir / f"{track_dir.stem}_centerline.csv"
            centerline = (
                Raceline.from_centerline_file(centerline_path)
                if centerline_path.exists()
                else None
            )

            # Load raceline
            raceline_path = track_dir / f"{track_dir.stem}_raceline.csv"
            raceline = (
                Raceline.from_raceline_file(raceline_path)
                if raceline_path.exists()
                else centerline
            )

            return Track(
                spec=track_spec,
                filepath=str((track_dir / map_filename.stem).absolute()),
                ext=map_filename.suffix,
                occupancy_map=occupancy_map,
                centerline=centerline,
                raceline=raceline,
            )

        except Exception as ex:
            print(f"[TrackLoader] Failed to load track: {track}\n{ex}")
            raise FileNotFoundError(f"It could not load track {track}") from ex

    @staticmethod
    def from_refline(
        x: np.ndarray,
        y: np.ndarray,
        velx: np.ndarray,
    ):
        """
        Create an empty track reference line.

        Parameters
        ----------
        x : np.ndarray
            x-coordinates of the waypoints
        y : np.ndarray
            y-coordinates of the waypoints
        velx : np.ndarray
            velocities at the waypoints

        Returns
        -------
        Track
            track object
        """
        ds = 0.1
        resolution = 0.05
        margin_perc = 0.1

        spline = CubicSpline2D(x=x, y=y)
        ss, xs, ys, yaws, ks, vxs = [], [], [], [], [], []
        for i_s in np.arange(0, spline.s[-1], ds):
            xi, yi = spline.calc_position(i_s)
            yaw = spline.calc_yaw(i_s)
            k = spline.calc_curvature(i_s)

            # find closest waypoint
            closest = np.argmin(np.hypot(x - xi, y - yi))
            v = velx[closest]

            xs.append(xi)
            ys.append(yi)
            yaws.append(yaw)
            ks.append(k)
            ss.append(i_s)
            vxs.append(v)

        refline = Raceline(
            ss=np.array(ss).astype(np.float32),
            xs=np.array(xs).astype(np.float32),
            ys=np.array(ys).astype(np.float32),
            psis=np.array(yaws).astype(np.float32),
            kappas=np.array(ks).astype(np.float32),
            velxs=np.array(vxs).astype(np.float32),
            accxs=np.zeros_like(ss).astype(np.float32),
            spline=spline,
        )

        min_x, max_x = np.min(xs), np.max(xs)
        min_y, max_y = np.min(ys), np.max(ys)
        x_range = max_x - min_x
        y_range = max_y - min_y
        occupancy_map = 255.0 * np.ones(
            (
                int((1 + 2 * margin_perc) * x_range / resolution),
                int((1 + 2 * margin_perc) * y_range / resolution),
            ),
            dtype=np.float32,
        )
        # origin is the bottom left corner
        origin = (min_x - margin_perc * x_range, min_y - margin_perc * y_range, 0.0)

        track_spec = TrackSpec(
            name=None,
            image=None,
            resolution=resolution,
            origin=origin,
            negate=False,
            occupied_thresh=0.65,
            free_thresh=0.196,
        )

        return Track(
            spec=track_spec,
            filepath=None,
            ext=None,
            occupancy_map=occupancy_map,
            raceline=refline,
            centerline=refline,
        )
