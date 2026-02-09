"""GenTL backend implemented using the Harvesters library."""

from __future__ import annotations

import glob
import logging
import os
import threading
import time
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np

from .base import CameraBackend

LOG = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from harvesters.core import Harvester  # type: ignore

    try:
        from harvesters.core import HarvesterTimeoutError  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        HarvesterTimeoutError = TimeoutError  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Harvester = None  # type: ignore
    HarvesterTimeoutError = TimeoutError  # type: ignore


class _SharedHarvester:
    """Singleton manager for shared Harvester instance.
    
    GenTL requires a single Harvester instance to properly enumerate and
    manage multiple cameras. This class provides thread-safe access to a
    shared Harvester with reference counting for proper cleanup.
    """
    
    _instance: Optional["_SharedHarvester"] = None
    _lock = threading.Lock()
    
    def __init__(self):
        self._harvester: Optional[Harvester] = None
        self._cti_file: Optional[str] = None
        self._ref_count = 0
        self._harvester_lock = threading.Lock()
    
    @classmethod
    def get_instance(cls) -> "_SharedHarvester":
        """Get the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def acquire(self, cti_file: str) -> Harvester:
        """Acquire a reference to the shared Harvester.
        
        If this is the first reference, creates and initializes the Harvester.
        If Harvester exists but with different CTI file, raises an error.
        
        Returns:
            The shared Harvester instance.
        """
        with self._harvester_lock:
            if self._harvester is None:
                if Harvester is None:
                    raise RuntimeError(
                        "The 'harvesters' package is required for the GenTL backend. "
                        "Install it via 'pip install harvesters'."
                    )
                self._harvester = Harvester()
                self._harvester.add_file(cti_file)
                self._harvester.update()
                self._cti_file = cti_file
                LOG.info(f"Shared Harvester initialized with CTI: {cti_file}")
            elif self._cti_file != cti_file:
                # Different CTI file requested - this could cause issues
                LOG.warning(
                    f"Harvester already initialized with '{self._cti_file}', "
                    f"ignoring request for '{cti_file}'"
                )
            
            self._ref_count += 1
            LOG.debug(f"Harvester reference count: {self._ref_count}")
            return self._harvester
    
    def release(self) -> None:
        """Release a reference to the shared Harvester.
        
        When the last reference is released, the Harvester is reset and cleaned up.
        """
        with self._harvester_lock:
            if self._ref_count > 0:
                self._ref_count -= 1
                LOG.debug(f"Harvester reference count: {self._ref_count}")
            
            if self._ref_count == 0 and self._harvester is not None:
                try:
                    self._harvester.reset()
                    LOG.info("Shared Harvester reset and cleaned up")
                except Exception as e:
                    LOG.warning(f"Error resetting Harvester: {e}")
                finally:
                    self._harvester = None
                    self._cti_file = None
    
    def get_device_count(self, cti_file: str) -> int:
        """Get device count without acquiring a permanent reference.
        
        Used for device enumeration/checking availability.
        """
        with self._harvester_lock:
            # If harvester already exists, just return current count
            if self._harvester is not None:
                return len(self._harvester.device_info_list)
            
            # Otherwise, create temporary harvester for enumeration
            if Harvester is None:
                return -1
            
            temp_harvester = None
            try:
                temp_harvester = Harvester()
                temp_harvester.add_file(cti_file)
                temp_harvester.update()
                return len(temp_harvester.device_info_list)
            except Exception:
                return -1
            finally:
                if temp_harvester is not None:
                    try:
                        temp_harvester.reset()
                    except Exception:
                        pass
    
    def update_device_list(self) -> None:
        """Update the device list on the shared Harvester."""
        with self._harvester_lock:
            if self._harvester is not None:
                self._harvester.update()


class GenTLCameraBackend(CameraBackend):
    """Capture frames from GenTL-compatible devices via Harvesters."""

    _DEFAULT_CTI_PATTERNS: Tuple[str, ...] = (
        r"C:\\Program Files\\The Imaging Source Europe GmbH\\IC4 GenTL Driver for USB3Vision Devices *\\bin\\*.cti",
        r"C:\\Program Files\\The Imaging Source Europe GmbH\\TIS Grabber\\bin\\win64_x64\\*.cti",
        r"C:\\Program Files\\The Imaging Source Europe GmbH\\TIS Camera SDK\\bin\\win64_x64\\*.cti",
        r"C:\\Program Files (x86)\\The Imaging Source Europe GmbH\\TIS Grabber\\bin\\win64_x64\\*.cti",
    )

    def __init__(self, settings):
        super().__init__(settings)
        props = settings.properties
        self._cti_file: Optional[str] = props.get("cti_file")
        self._serial_number: Optional[str] = props.get("serial_number") or props.get("serial")
        self._pixel_format: str = props.get("pixel_format", "Mono8")
        self._rotate: int = int(props.get("rotate", 0)) % 360
        self._crop: Optional[Tuple[int, int, int, int]] = self._parse_crop(props.get("crop"))
        # Check settings first (from config), then properties (for backward compatibility)
        self._exposure: Optional[float] = (
            settings.exposure if settings.exposure else props.get("exposure")
        )
        self._gain: Optional[float] = settings.gain if settings.gain else props.get("gain")
        self._timeout: float = float(props.get("timeout", 2.0))
        self._cti_search_paths: Tuple[str, ...] = self._parse_cti_paths(
            props.get("cti_search_paths")
        )
        # Parse resolution (width, height) with defaults
        self._resolution: Optional[Tuple[int, int]] = self._parse_resolution(
            props.get("resolution")
        )

        # Trigger mode settings
        # trigger_mode: "freerun" (default), "triggered", or "off"/"on" for explicit control
        self._trigger_mode: str = str(props.get("trigger_mode", "freerun")).lower()
        # trigger_source: e.g., "Line0", "Line1", "Software" (default: "Line0" for triggered mode)
        self._trigger_source: str = props.get("trigger_source", "Line0")
        # trigger_activation: "RisingEdge", "FallingEdge", "AnyEdge", "LevelHigh", "LevelLow"
        self._trigger_activation: str = props.get("trigger_activation", "RisingEdge")
        # trigger_selector: "FrameStart", "ExposureStart", "ExposureActive" (default: "FrameStart")
        self._trigger_selector: str = props.get("trigger_selector", "FrameStart")

        self._shared_harvester: Optional[_SharedHarvester] = None
        self._harvester: Optional[Harvester] = None  # Reference to shared harvester
        self._acquirer = None
        self._device_label: Optional[str] = None

    @classmethod
    def is_available(cls) -> bool:
        return Harvester is not None

    @classmethod
    def get_device_count(cls) -> int:
        """Get the actual number of GenTL devices detected by Harvester.

        Returns the number of devices found, or -1 if detection fails.
        """
        if Harvester is None:
            return -1

        # Use the static helper to find CTI file with default patterns
        cti_file = cls._search_cti_file(cls._DEFAULT_CTI_PATTERNS)
        if not cti_file:
            return -1

        # Use shared harvester for device count
        shared = _SharedHarvester.get_instance()
        return shared.get_device_count(cti_file)

    def open(self) -> None:
        if Harvester is None:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "The 'harvesters' package is required for the GenTL backend. "
                "Install it via 'pip install harvesters'."
            )

        # Use shared Harvester instance for multi-camera support
        cti_file = self._cti_file or self._find_cti_file()
        self._shared_harvester = _SharedHarvester.get_instance()
        self._harvester = self._shared_harvester.acquire(cti_file)

        if not self._harvester.device_info_list:
            raise RuntimeError("No GenTL cameras detected via Harvesters")

        serial = self._serial_number
        index = int(self.settings.index or 0)
        if serial:
            available = self._available_serials()
            matches = [s for s in available if serial in s]
            if not matches:
                raise RuntimeError(
                    f"Camera with serial '{serial}' not found. Available cameras: {available}"
                )
            serial = matches[0]
        else:
            device_count = len(self._harvester.device_info_list)
            if index < 0 or index >= device_count:
                raise RuntimeError(
                    f"Camera index {index} out of range for {device_count} GenTL device(s)"
                )

        self._acquirer = self._create_acquirer(serial, index)

        remote = self._acquirer.remote_device
        node_map = remote.node_map

        # print(dir(node_map))
        """
        ['AcquisitionBurstFrameCount', 'AcquisitionControl', 'AcquisitionFrameRate', 'AcquisitionMode',
        'AcquisitionStart', 'AcquisitionStop', 'AnalogControl', 'AutoFunctionsROI', 'AutoFunctionsROIEnable',
        'AutoFunctionsROIHeight', 'AutoFunctionsROILeft', 'AutoFunctionsROIPreset', 'AutoFunctionsROITop',
        'AutoFunctionsROIWidth', 'BinningHorizontal', 'BinningVertical', 'BlackLevel', 'CameraRegisterAddress',
        'CameraRegisterAddressSpace', 'CameraRegisterControl', 'CameraRegisterRead', 'CameraRegisterValue',
        'CameraRegisterWrite', 'Contrast', 'DecimationHorizontal', 'DecimationVertical', 'Denoise',
        'DeviceControl', 'DeviceFirmwareVersion', 'DeviceModelName', 'DeviceReset', 'DeviceSFNCVersionMajor',
        'DeviceSFNCVersionMinor', 'DeviceSFNCVersionSubMinor', 'DeviceScanType', 'DeviceSerialNumber',
        'DeviceTLType', 'DeviceTLVersionMajor', 'DeviceTLVersionMinor', 'DeviceTLVersionSubMinor',
        'DeviceTemperature', 'DeviceTemperatureSelector', 'DeviceType', 'DeviceUserID', 'DeviceVendorName',
        'DigitalIO', 'ExposureAuto', 'ExposureAutoHighlightReduction', 'ExposureAutoLowerLimit',
        'ExposureAutoReference', 'ExposureAutoUpperLimit', 'ExposureAutoUpperLimitAuto', 'ExposureTime',
        'GPIn', 'GPOut', 'Gain', 'GainAuto', 'GainAutoLowerLimit', 'GainAutoUpperLimit', 'Gamma', 'Height',
        'HeightMax', 'IMXLowLatencyTriggerMode', 'ImageFormatControl', 'OffsetAutoCenter', 'OffsetX', 'OffsetY',
        'PayloadSize', 'PixelFormat', 'ReverseX', 'ReverseY', 'Root', 'SensorHeight', 'SensorWidth', 'Sharpness',
        'ShowOverlay', 'SoftwareAnalogControl', 'SoftwareTransformControl', 'SoftwareTransformEnable',
        'StrobeDelay', 'StrobeDuration', 'StrobeEnable', 'StrobeOperation', 'StrobePolarity', 'TLParamsLocked',
        'TestControl', 'TestPendingAck', 'TimestampLatch', 'TimestampLatchValue', 'TimestampReset', 'ToneMappingAuto',
        'ToneMappingControl', 'ToneMappingEnable', 'ToneMappingGlobalBrightness', 'ToneMappingIntensity',
        'TransportLayerControl', 'TriggerActivation', 'TriggerDebouncer', 'TriggerDelay', 'TriggerDenoise',
        'TriggerMask', 'TriggerMode', 'TriggerOverlap', 'TriggerSelector', 'TriggerSoftware', 'TriggerSource',
        'UserSetControl', 'UserSetDefault', 'UserSetLoad', 'UserSetSave', 'UserSetSelector', 'Width', 'WidthMax']
        """

        self._device_label = self._resolve_device_label(node_map)

        self._configure_pixel_format(node_map)
        self._configure_resolution(node_map)
        self._configure_exposure(node_map)
        self._configure_gain(node_map)
        self._configure_frame_rate(node_map)
        self._configure_trigger_mode(node_map)

        self._acquirer.start()

    def read(self) -> Tuple[np.ndarray, float]:
        if self._acquirer is None:
            raise RuntimeError("GenTL image acquirer not initialised")

        try:
            with self._acquirer.fetch(timeout=self._timeout) as buffer:
                component = buffer.payload.components[0]
                channels = 3 if self._pixel_format in {"RGB8", "BGR8"} else 1
                array = np.asarray(component.data)
                expected = component.height * component.width * channels
                if array.size != expected:
                    array = np.frombuffer(bytes(component.data), dtype=array.dtype)
                try:
                    if channels > 1:
                        frame = array.reshape(component.height, component.width, channels).copy()
                    else:
                        frame = array.reshape(component.height, component.width).copy()
                except ValueError:
                    frame = array.copy()
        except HarvesterTimeoutError as exc:
            raise TimeoutError(str(exc) + " (GenTL timeout)") from exc

        frame = self._convert_frame(frame)
        timestamp = time.time()
        return frame, timestamp

    def set_trigger_mode(self, mode: str) -> None:
        """Change trigger mode at runtime.

        Args:
            mode: "freerun" or "off" for continuous acquisition,
                  "triggered" or "on" for external trigger mode.

        Note: Camera must be opened first. This will briefly stop and restart
        acquisition to apply the new trigger settings.
        """
        if self._acquirer is None:
            raise RuntimeError("Camera not opened. Call open() first.")

        self._trigger_mode = mode.lower()

        # Stop acquisition to change trigger settings
        try:
            self._acquirer.stop()
        except Exception:
            pass

        # Get node map and reconfigure trigger
        remote = self._acquirer.remote_device
        node_map = remote.node_map
        self._configure_trigger_mode(node_map)

        # Restart acquisition
        self._acquirer.start()
        LOG.info(f"Trigger mode changed to '{mode}'")

    def stop(self) -> None:
        if self._acquirer is not None:
            try:
                self._acquirer.stop()
            except Exception:
                pass

    def close(self) -> None:
        if self._acquirer is not None:
            try:
                self._acquirer.stop()
            except Exception:
                pass
            try:
                destroy = getattr(self._acquirer, "destroy", None)
                if destroy is not None:
                    destroy()
            finally:
                self._acquirer = None

        # Release reference to shared Harvester (don't reset it directly)
        if self._shared_harvester is not None:
            self._shared_harvester.release()
            self._shared_harvester = None
        self._harvester = None

        self._device_label = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_cti_paths(self, value) -> Tuple[str, ...]:
        if value is None:
            return self._DEFAULT_CTI_PATTERNS
        if isinstance(value, str):
            return (value,)
        if isinstance(value, Iterable):
            return tuple(str(item) for item in value)
        return self._DEFAULT_CTI_PATTERNS

    def _parse_crop(self, crop) -> Optional[Tuple[int, int, int, int]]:
        if isinstance(crop, (list, tuple)) and len(crop) == 4:
            return tuple(int(v) for v in crop)
        return None

    def _parse_resolution(self, resolution) -> Optional[Tuple[int, int]]:
        """Parse resolution setting.

        Args:
            resolution: Can be a tuple/list [width, height], or None

        Returns:
            Tuple of (width, height) or None if not specified
            Default is (720, 540) if parsing fails but value is provided
        """
        if resolution is None:
            return (720, 540)  # Default resolution

        if isinstance(resolution, (list, tuple)) and len(resolution) == 2:
            try:
                return (int(resolution[0]), int(resolution[1]))
            except (ValueError, TypeError):
                return (720, 540)

        return (720, 540)

    @staticmethod
    def _search_cti_file(patterns: Tuple[str, ...]) -> Optional[str]:
        """Search for a CTI file using the given patterns.

        Returns the first CTI file found, or None if none found.
        """
        for pattern in patterns:
            for file_path in glob.glob(pattern):
                if os.path.isfile(file_path):
                    return file_path
        return None

    def _find_cti_file(self) -> str:
        """Find a CTI file using configured or default search paths.

        Raises RuntimeError if no CTI file is found.
        """
        cti_file = self._search_cti_file(self._cti_search_paths)
        if cti_file is None:
            raise RuntimeError(
                "Could not locate a GenTL producer (.cti) file. Set 'cti_file' in "
                "camera.properties or provide search paths via 'cti_search_paths'."
            )
        return cti_file

    def _available_serials(self) -> List[str]:
        assert self._harvester is not None
        serials: List[str] = []
        for info in self._harvester.device_info_list:
            serial = getattr(info, "serial_number", "")
            if serial:
                serials.append(serial)
        return serials

    def _create_acquirer(self, serial: Optional[str], index: int):
        assert self._harvester is not None
        methods = [
            getattr(self._harvester, "create", None),
            getattr(self._harvester, "create_image_acquirer", None),
        ]
        methods = [m for m in methods if m is not None]
        errors: List[str] = []
        device_info = None
        if not serial:
            device_list = self._harvester.device_info_list
            if 0 <= index < len(device_list):
                device_info = device_list[index]
        for create in methods:
            try:
                if serial:
                    return create({"serial_number": serial})
            except Exception as exc:
                errors.append(f"{create.__name__} serial: {exc}")
        for create in methods:
            try:
                return create(index=index)
            except TypeError:
                try:
                    return create(index)
                except Exception as exc:
                    errors.append(f"{create.__name__} index positional: {exc}")
            except Exception as exc:
                errors.append(f"{create.__name__} index: {exc}")
        if device_info is not None:
            for create in methods:
                try:
                    return create(device_info)
                except Exception as exc:
                    errors.append(f"{create.__name__} device_info: {exc}")
        if not serial and index == 0:
            for create in methods:
                try:
                    return create()
                except Exception as exc:
                    errors.append(f"{create.__name__} default: {exc}")
        joined = "; ".join(errors) or "no creation methods available"
        raise RuntimeError(f"Failed to initialise GenTL image acquirer ({joined})")

    def _configure_pixel_format(self, node_map) -> None:
        try:
            if self._pixel_format in node_map.PixelFormat.symbolics:
                node_map.PixelFormat.value = self._pixel_format
                actual = node_map.PixelFormat.value
                if actual != self._pixel_format:
                    LOG.warning(
                        f"Pixel format mismatch: requested '{self._pixel_format}', got '{actual}'"
                    )
                else:
                    LOG.info(f"Pixel format set to '{actual}'")
            else:
                LOG.warning(
                    f"Pixel format '{self._pixel_format}' not in available formats: "
                    f"{node_map.PixelFormat.symbolics}"
                )
        except Exception as e:
            LOG.warning(f"Failed to set pixel format '{self._pixel_format}': {e}")

    def _configure_resolution(self, node_map) -> None:
        """Configure camera resolution (width and height)."""
        if self._resolution is None:
            return

        requested_width, requested_height = self._resolution
        actual_width, actual_height = None, None

        # Try to set width
        for width_attr in ("Width", "WidthMax"):
            try:
                node = getattr(node_map, width_attr)
                if width_attr == "Width":
                    # Get constraints
                    try:
                        min_w = node.min
                        max_w = node.max
                        inc_w = getattr(node, "inc", 1)
                        # Adjust to valid value
                        width = self._adjust_to_increment(requested_width, min_w, max_w, inc_w)
                        if width != requested_width:
                            LOG.info(
                                f"Width adjusted from {requested_width} to {width} "
                                f"(min={min_w}, max={max_w}, inc={inc_w})"
                            )
                        node.value = int(width)
                        actual_width = node.value
                        break
                    except Exception as e:
                        # Try setting without adjustment
                        try:
                            node.value = int(requested_width)
                            actual_width = node.value
                            break
                        except Exception:
                            LOG.warning(f"Failed to set width via {width_attr}: {e}")
                            continue
            except AttributeError:
                continue

        # Try to set height
        for height_attr in ("Height", "HeightMax"):
            try:
                node = getattr(node_map, height_attr)
                if height_attr == "Height":
                    # Get constraints
                    try:
                        min_h = node.min
                        max_h = node.max
                        inc_h = getattr(node, "inc", 1)
                        # Adjust to valid value
                        height = self._adjust_to_increment(requested_height, min_h, max_h, inc_h)
                        if height != requested_height:
                            LOG.info(
                                f"Height adjusted from {requested_height} to {height} "
                                f"(min={min_h}, max={max_h}, inc={inc_h})"
                            )
                        node.value = int(height)
                        actual_height = node.value
                        break
                    except Exception as e:
                        # Try setting without adjustment
                        try:
                            node.value = int(requested_height)
                            actual_height = node.value
                            break
                        except Exception:
                            LOG.warning(f"Failed to set height via {height_attr}: {e}")
                            continue
            except AttributeError:
                continue

        # Log final resolution
        if actual_width is not None and actual_height is not None:
            if actual_width != requested_width or actual_height != requested_height:
                LOG.warning(
                    f"Resolution mismatch: requested {requested_width}x{requested_height}, "
                    f"got {actual_width}x{actual_height}"
                )
            else:
                LOG.info(f"Resolution set to {actual_width}x{actual_height}")
        else:
            LOG.warning(
                f"Could not verify resolution setting "
                f"(width={actual_width}, height={actual_height})"
            )

    def _configure_exposure(self, node_map) -> None:
        if self._exposure is None:
            return

        # Try to disable auto exposure first
        for attr in ("ExposureAuto",):
            try:
                node = getattr(node_map, attr)
                node.value = "Off"
                LOG.info("Auto exposure disabled")
                break
            except AttributeError:
                continue
            except Exception as e:
                LOG.warning(f"Failed to disable auto exposure: {e}")

        # Set exposure value
        for attr in ("ExposureTime", "Exposure"):
            try:
                node = getattr(node_map, attr)
            except AttributeError:
                continue
            try:
                node.value = float(self._exposure)
                actual = node.value
                if abs(actual - self._exposure) > 1.0:  # Allow 1μs tolerance
                    LOG.warning(f"Exposure mismatch: requested {self._exposure}μs, got {actual}μs")
                else:
                    LOG.info(f"Exposure set to {actual}μs")
                return
            except Exception as e:
                LOG.warning(f"Failed to set exposure via {attr}: {e}")
                continue

        LOG.warning(f"Could not set exposure to {self._exposure}μs (no compatible attribute found)")

    def _configure_gain(self, node_map) -> None:
        if self._gain is None:
            return

        # Try to disable auto gain first
        for attr in ("GainAuto",):
            try:
                node = getattr(node_map, attr)
                node.value = "Off"
                LOG.info("Auto gain disabled")
                break
            except AttributeError:
                continue
            except Exception as e:
                LOG.warning(f"Failed to disable auto gain: {e}")

        # Set gain value
        for attr in ("Gain",):
            try:
                node = getattr(node_map, attr)
            except AttributeError:
                continue
            try:
                node.value = float(self._gain)
                actual = node.value
                if abs(actual - self._gain) > 0.1:  # Allow 0.1 tolerance
                    LOG.warning(f"Gain mismatch: requested {self._gain}, got {actual}")
                else:
                    LOG.info(f"Gain set to {actual}")
                return
            except Exception as e:
                LOG.warning(f"Failed to set gain via {attr}: {e}")
                continue

        LOG.warning(f"Could not set gain to {self._gain} (no compatible attribute found)")

    def _configure_frame_rate(self, node_map) -> None:
        if not self.settings.fps:
            return

        target = float(self.settings.fps)

        # Try to enable frame rate control
        for attr in ("AcquisitionFrameRateEnable", "AcquisitionFrameRateControlEnable"):
            try:
                getattr(node_map, attr).value = True
                LOG.info(f"Frame rate control enabled via {attr}")
                break
            except Exception:
                continue

        # Set frame rate value
        for attr in ("AcquisitionFrameRate", "ResultingFrameRate", "AcquisitionFrameRateAbs"):
            try:
                node = getattr(node_map, attr)
            except AttributeError:
                continue
            try:
                node.value = target
                actual = node.value
                if abs(actual - target) > 0.1:
                    LOG.warning(f"FPS mismatch: requested {target:.2f}, got {actual:.2f}")
                else:
                    LOG.info(f"Frame rate set to {actual:.2f} FPS")
                return
            except Exception as e:
                LOG.warning(f"Failed to set frame rate via {attr}: {e}")
                continue

        LOG.warning(f"Could not set frame rate to {target} FPS (no compatible attribute found)")

    def _configure_trigger_mode(self, node_map) -> None:
        """Configure camera trigger mode for freerun or external triggering.

        Trigger modes:
            - "freerun" or "off": Camera runs in free-running mode (self-timed acquisition)
            - "triggered" or "on": Camera waits for external trigger signal

        For triggered mode, configures:
            - TriggerSelector: Which event to trigger (default: FrameStart)
            - TriggerSource: Input line for trigger (default: Line0)
            - TriggerActivation: Edge type (default: RisingEdge)
            - TriggerMode: On

        Properties used from settings.properties:
            - trigger_mode: "freerun", "triggered", "on", "off"
            - trigger_source: "Line0", "Line1", "Software", etc.
            - trigger_activation: "RisingEdge", "FallingEdge", "AnyEdge", "LevelHigh", "LevelLow"
            - trigger_selector: "FrameStart", "ExposureStart", "ExposureActive"
        """
        mode = self._trigger_mode

        # Normalize mode names
        if mode in ("freerun", "free", "off", "continuous"):
            enable_trigger = False
        elif mode in ("triggered", "trigger", "on", "external", "hardware"):
            enable_trigger = True
        else:
            LOG.warning(f"Unknown trigger mode '{mode}', defaulting to freerun")
            enable_trigger = False

        if not enable_trigger:
            # Set to freerun mode (disable trigger)
            self._set_trigger_off(node_map)
            return

        # Enable triggered mode
        self._set_trigger_on(node_map)

    def _set_trigger_off(self, node_map) -> None:
        """Disable trigger mode (freerun/continuous acquisition)."""
        try:
            # Disable trigger mode
            if hasattr(node_map, "TriggerMode"):
                try:
                    node_map.TriggerMode.value = "Off"
                    LOG.info("Trigger mode set to Off (freerun)")
                except Exception as e:
                    LOG.warning(f"Failed to disable TriggerMode: {e}")
            else:
                LOG.debug("TriggerMode not available, camera may already be in freerun mode")

        except Exception as e:
            LOG.warning(f"Error configuring freerun mode: {e}")

    def _set_trigger_on(self, node_map) -> None:
        """Enable external trigger mode with configured settings."""
        try:
            # Step 1: Set TriggerSelector (which event to trigger)
            if hasattr(node_map, "TriggerSelector"):
                available_selectors = []
                try:
                    available_selectors = list(node_map.TriggerSelector.symbolics)
                except Exception:
                    pass

                if self._trigger_selector in available_selectors:
                    node_map.TriggerSelector.value = self._trigger_selector
                    LOG.info(f"TriggerSelector set to '{self._trigger_selector}'")
                elif available_selectors:
                    # Default to FrameStart if available
                    if "FrameStart" in available_selectors:
                        node_map.TriggerSelector.value = "FrameStart"
                        LOG.warning(
                            f"TriggerSelector '{self._trigger_selector}' not available, "
                            f"using 'FrameStart'. Available: {available_selectors}"
                        )
                    else:
                        LOG.warning(
                            f"Could not set TriggerSelector. Available: {available_selectors}"
                        )

            # Step 2: Set TriggerSource (input line for trigger)
            if hasattr(node_map, "TriggerSource"):
                available_sources = []
                try:
                    available_sources = list(node_map.TriggerSource.symbolics)
                except Exception:
                    pass

                if self._trigger_source in available_sources:
                    node_map.TriggerSource.value = self._trigger_source
                    LOG.info(f"TriggerSource set to '{self._trigger_source}'")
                elif available_sources:
                    # Try to use Line0 as default if available
                    if "Line0" in available_sources:
                        node_map.TriggerSource.value = "Line0"
                        LOG.warning(
                            f"TriggerSource '{self._trigger_source}' not available, "
                            f"using 'Line0'. Available: {available_sources}"
                        )
                    else:
                        LOG.warning(
                            f"Could not set TriggerSource. Available: {available_sources}"
                        )
    
            # Step 3: Set TriggerActivation (edge type)
            if hasattr(node_map, "TriggerActivation"):
                available_activations = []
                try:
                    available_activations = list(node_map.TriggerActivation.symbolics)
                except Exception:
                    pass

                if self._trigger_activation in available_activations:
                    node_map.TriggerActivation.value = self._trigger_activation
                    LOG.info(f"TriggerActivation set to '{self._trigger_activation}'")
                elif available_activations:
                    # Default to RisingEdge if available
                    if "RisingEdge" in available_activations:
                        node_map.TriggerActivation.value = "RisingEdge"
                        LOG.warning(
                            f"TriggerActivation '{self._trigger_activation}' not available, "
                            f"using 'RisingEdge'. Available: {available_activations}"
                        )
                    else:
                        LOG.warning(
                            f"Could not set TriggerActivation. Available: {available_activations}"
                        )

            # Step 4: Enable TriggerMode
            if hasattr(node_map, "TriggerMode"):
                node_map.TriggerMode.value = "On"
                LOG.info(
                    f"Trigger mode enabled: selector={self._trigger_selector}, "
                    f"source={self._trigger_source}, activation={self._trigger_activation}"
                )
            else:
                LOG.warning("TriggerMode not available, cannot enable external triggering")

        except Exception as e:
            LOG.error(f"Failed to configure trigger mode: {e}")
            raise RuntimeError(f"Failed to enable external trigger: {e}")

    def _convert_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame.dtype != np.uint8:
            max_val = float(frame.max()) if frame.size else 0.0
            scale = 255.0 / max_val if max_val > 0.0 else 1.0
            frame = np.clip(frame * scale, 0, 255).astype(np.uint8)

        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 3 and self._pixel_format == "RGB8":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if self._crop is not None:
            top, bottom, left, right = (int(v) for v in self._crop)
            top = max(0, top)
            left = max(0, left)
            bottom = bottom if bottom > 0 else frame.shape[0]
            right = right if right > 0 else frame.shape[1]
            bottom = min(frame.shape[0], bottom)
            right = min(frame.shape[1], right)
            frame = frame[top:bottom, left:right]

        if self._rotate in (90, 180, 270):
            rotations = {
                90: cv2.ROTATE_90_CLOCKWISE,
                180: cv2.ROTATE_180,
                270: cv2.ROTATE_90_COUNTERCLOCKWISE,
            }
            frame = cv2.rotate(frame, rotations[self._rotate])

        return frame.copy()

    def _resolve_device_label(self, node_map) -> Optional[str]:
        candidates = [
            ("DeviceModelName", "DeviceSerialNumber"),
            ("DeviceDisplayName", "DeviceSerialNumber"),
        ]
        for name_attr, serial_attr in candidates:
            try:
                model = getattr(node_map, name_attr).value
            except AttributeError:
                continue
            serial = None
            try:
                serial = getattr(node_map, serial_attr).value
            except AttributeError:
                pass
            if model:
                model_str = str(model)
                serial_str = str(serial) if serial else None
                return f"{model_str} ({serial_str})" if serial_str else model_str
        return None

    def _adjust_to_increment(self, value: int, minimum: int, maximum: int, increment: int) -> int:
        value = max(minimum, min(maximum, int(value)))
        if increment <= 0:
            return value
        offset = value - minimum
        steps = offset // increment
        return minimum + steps * increment

    def device_name(self) -> str:
        if self._device_label:
            return self._device_label
        return super().device_name()
