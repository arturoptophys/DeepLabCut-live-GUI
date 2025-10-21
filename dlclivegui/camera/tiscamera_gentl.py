"""
DeepLabCut Toolbox (deeplabcut.org)
© A. & M. Mathis Labs

Licensed under GNU Lesser General Public License v3.0
"""

import time
import cv2
import numpy as np
from harvesters.core import Harvester

from dlclivegui.camera import Camera, CameraError


class TISCamGenTL(Camera):
    """
    Imaging Source Camera class using GenTL/GenICam interface via Harvesters.
    
    Requires:
        - pip install harvesters
        - The Imaging Source GenTL producer (.cti file)
    """
    
    @staticmethod
    def arg_restrictions():
        """Returns dictionary of argument restrictions for DLCLiveGUI"""
        return {
            "rotate": [0, 90, 180, 270],
            "pixel_format": ["Mono8", "Mono12", "Mono16", "RGB8", "BGR8"]
        }

    def __init__(
        self,
        serial_number="",
        cti_file=None,
        resolution=[720, 528],
        exposure=5000,  # microseconds
        gain=0,
        rotate=90,
        crop=None,
        fps=100,
        pixel_format="Mono8",
        use_tk_display=True,
        display_resize=1.0,
    ):
        """
        Initialize TIS Camera with GenTL interface.
        
        Parameters
        ----------
        serial_number : str
            Camera serial number
        cti_file : str, optional
            Path to GenTL producer (.cti) file. If None, searches common locations
        resolution : list
            [width, height] in pixels
        exposure : float
            Exposure time in microseconds
        gain : float
            Gain value in dB
        rotate : int
            Rotation angle (0, 90, 180, 270)
        crop : list, optional
            Crop parameters [top, bottom, left, right]
        fps : float
            Target frame rate
        pixel_format : str
            Pixel format (Mono8, RGB8, etc.)
        use_tk_display : bool
            Whether to use Tk display
        display_resize : float
            Display resize factor
        """
        
        # Pre-adjust resolution to common valid values
        # Most TIS cameras require width/height divisible by 4, 8, or 16
        # Round down to nearest multiple of 16 to be safe
        resolution = [
            (resolution[0] // 16) * 16 if resolution[0] % 16 != 0 else resolution[0],
            (resolution[1] // 16) * 16 if resolution[1] % 16 != 0 else resolution[1]
        ]
        
        if (rotate == 90) or (rotate == 270):
            resolution = [resolution[1], resolution[0]]
        print(f"Using resolution: {resolution[0]} x {resolution[1]}")
        super().__init__(
            serial_number,
            resolution=resolution,
            exposure=exposure,
            gain=gain,
            rotate=rotate,
            crop=crop,
            fps=fps,
            use_tk_display=use_tk_display,
            display_resize=display_resize,
        )
        
        self.cti_file = cti_file
        self.pixel_format = pixel_format
        self.harvester = None
        self.image_acquirer = None
        
    def _find_cti_file(self):
        """Find GenTL producer file in common installation locations"""
        import os
        import glob
        
        common_paths = [
            r"C:\Program Files\The Imaging Source Europe GmbH\IC4 GenTL Driver for USB3Vision Devices 1.5\bin\*.cti",
            r"C:\Program Files\The Imaging Source Europe GmbH\TIS Grabber\bin\win64_x64\*.cti",
            r"C:\Program Files\The Imaging Source Europe GmbH\TIS Camera SDK\bin\win64_x64\*.cti",
            r"C:\Program Files (x86)\The Imaging Source Europe GmbH\TIS Grabber\bin\win64_x64\*.cti",
        ]
        
        for path_pattern in common_paths:
            files = glob.glob(path_pattern)
            if files:
                return files[0]
        
        raise CameraError(
            "Could not find GenTL producer (.cti) file. "
            "Please specify the path manually using cti_file parameter."
        )

    def _adjust_to_increment(self, value, minimum, maximum, increment):
        """Adjust value to meet camera constraints (min, max, increment)"""
        # Clamp to min/max range
        value = max(minimum, min(maximum, value))
        # Round down to nearest valid increment
        value = minimum + ((value - minimum) // increment) * increment
        return value

    def set_capture_device(self):
        """Initialize camera connection and configure settings"""
        
        try:
            # Initialize Harvester
            self.harvester = Harvester()
            
            # Add CTI file
            if self.cti_file is None:
                self.cti_file = self._find_cti_file()
            
            self.harvester.add_file(self.cti_file)
            self.harvester.update()
            
            # Find and open camera by serial number
            if not self.id:
                # If no serial number, use first available camera
                if len(self.harvester.device_info_list) == 0:
                    raise CameraError("No cameras detected")
                self.image_acquirer = self.harvester.create()
            else:
                # Find camera by serial number
                found = False
                for device_info in self.harvester.device_info_list:
                    if self.id in device_info.serial_number:
                        self.image_acquirer = self.harvester.create(
                            {"serial_number": self.id}
                        )
                        found = True
                        break
                
                if not found:
                    available = [d.serial_number for d in self.harvester.device_info_list]
                    raise CameraError(
                        f"Camera with serial {self.id} not found. "
                        f"Available cameras: {available}"
                    )
            
            # Access remote device (camera) node map
            node_map = self.image_acquirer.remote_device.node_map
            
            # Set pixel format
            if self.pixel_format in node_map.PixelFormat.symbolics:
                node_map.PixelFormat.value = self.pixel_format
            
            # Set resolution with validation
            # Width and height must meet min/max/increment constraints
            desired_width = int(self.im_size[1])
            desired_height = int(self.im_size[0])
            
            # Adjust width to meet constraints
            adjusted_width = self._adjust_to_increment(
                desired_width, 
                node_map.Width.min, 
                node_map.Width.max, 
                node_map.Width.inc
            )
            node_map.Width.value = adjusted_width
            
            # Adjust height to meet constraints
            adjusted_height = self._adjust_to_increment(
                desired_height,
                node_map.Height.min,
                node_map.Height.max,
                node_map.Height.inc
            )
            node_map.Height.value = adjusted_height
            
            # Update im_size with actual values if different from desired
            if adjusted_width != desired_width or adjusted_height != desired_height:
                print(f"Resolution adjusted from {desired_width}x{desired_height} to {adjusted_width}x{adjusted_height}")
                # Always update im_size to match what the camera actually provides
                self.im_size = (adjusted_height, adjusted_width)
            
            # Set exposure (in microseconds)
            # First disable auto-exposure if available
            try:
                node_map.ExposureAuto.value = 'Off'
            except (AttributeError, Exception):
                pass  # Node doesn't exist or auto mode not available
            
            # Then set manual exposure value
            try:
                node_map.ExposureTime.value = float(self.exposure)
            except (AttributeError, Exception):
                try:
                    node_map.Exposure.value = float(self.exposure)
                except (AttributeError, Exception) as e:
                    print(f"Warning: Could not set Exposure: {e}")
            
            # Set gain if available
            if self.gain is not None:
                # Disable auto-gain if available
                try:
                    node_map.GainAuto.value = 'Off'
                except (AttributeError, Exception):
                    pass  # Node doesn't exist or auto mode not available
                
                # Set manual gain value
                try:
                    node_map.Gain.value = float(self.gain)
                except (AttributeError, Exception) as e:
                    print(f"Warning: Could not set Gain: {e}")
            
            # Set frame rate
            try:
                node_map.AcquisitionFrameRateEnable.value = True
            except (AttributeError, Exception):
                pass  # Node doesn't exist or not writable
            
            try:
                if hasattr(node_map, 'AcquisitionFrameRate'):
                    node_map.AcquisitionFrameRate.value = float(self.fps)
            except Exception as e:
                print(f"Warning: Could not set frame rate: {e}")
            
            # Start acquisition
            self.image_acquirer.start()
            
            self.next_frame = time.time()
            
            return True
            
        except Exception as e:
            raise CameraError(f"Failed to initialize camera: {str(e)}")

    def get_image(self):
        """Capture and return a single frame"""
        
        try:
            # Fetch buffer
            with self.image_acquirer.fetch(timeout=2.0) as buffer:
                # Get image component
                component = buffer.payload.components[0]
                
                # Convert to numpy array
                frame = component.data.reshape(
                    component.height, component.width
                )
                
                # Convert based on pixel format
                if self.pixel_format in ["Mono12", "Mono16"]:
                    # Normalize to 8-bit for display
                    frame = (frame / frame.max() * 255).astype(np.uint8)
                    # Convert grayscale to BGR
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif self.pixel_format == "Mono8":
                    # Convert grayscale to BGR
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif self.pixel_format in ["RGB8", "BGR8"]:
                    frame = frame.reshape(component.height, component.width, 3)
                    if self.pixel_format == "RGB8":
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Apply rotation
                if self.rotate == 90:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif self.rotate == 180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif self.rotate == 270:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                # Apply crop
                if self.crop is not None:
                    frame = frame[
                        self.crop[0]:self.crop[1],
                        self.crop[2]:self.crop[3]
                    ]
                
                return frame.copy()
                
        except Exception as e:
            raise CameraError(f"Failed to capture image: {str(e)}")

    def set_exposure(self, exposure):
        """Set camera exposure time in microseconds"""
        
        if self.image_acquirer is not None:
            node_map = self.image_acquirer.remote_device.node_map
            
            # Disable auto-exposure first
            try:
                node_map.ExposureAuto.value = 'Off'
            except (AttributeError, Exception):
                pass
            
            # Set exposure value
            try:
                node_map.ExposureTime.value = float(exposure)
                self.exposure = exposure
            except (AttributeError, Exception):
                try:
                    node_map.Exposure.value = float(exposure)
                    self.exposure = exposure
                except (AttributeError, Exception) as e:
                    print(f"Warning: Could not set Exposure: {e}")

    def set_gain(self, gain):
        """Set camera gain in dB"""
        
        if self.image_acquirer is not None:
            node_map = self.image_acquirer.remote_device.node_map
            
            # Disable auto-gain first
            try:
                node_map.GainAuto.value = 'Off'
            except (AttributeError, Exception):
                pass
            
            # Set gain value
            try:
                node_map.Gain.value = float(gain)
                self.gain = gain
            except (AttributeError, Exception) as e:
                print(f"Warning: Could not set Gain: {e}")

    def get_exposure(self):
        """Get current exposure time in microseconds"""
        
        if self.image_acquirer is not None:
            node_map = self.image_acquirer.remote_device.node_map
            
            try:
                return node_map.ExposureTime.value
            except (AttributeError, Exception):
                try:
                    return node_map.Exposure.value
                except (AttributeError, Exception):
                    pass
        
        return self.exposure

    def get_gain(self):
        """Get current gain in dB"""
        
        if self.image_acquirer is not None:
            node_map = self.image_acquirer.remote_device.node_map
            
            try:
                return node_map.Gain.value
            except (AttributeError, Exception):
                pass
        
        return self.gain

    @staticmethod
    def get_available_cameras(cti_file=None):
        """Get list of available camera serial numbers"""
        
        harvester = Harvester()
        
        if cti_file is None:
            # Try to find CTI file
            try:
                instance = TISCamGenTL()
                cti_file = instance._find_cti_file()
            except:
                return []
        
        harvester.add_file(cti_file)
        harvester.update()
        
        cameras = [d.serial_number for d in harvester.device_info_list]
        
        harvester.reset()
        
        return cameras

    def close_capture_device(self):
        """Stop acquisition and release camera resources"""
        
        try:
            if self.image_acquirer is not None:
                self.image_acquirer.stop()
                self.image_acquirer.destroy()
                self.image_acquirer = None
            
            if self.harvester is not None:
                self.harvester.reset()
                self.harvester = None
                
        except Exception as e:
            print(f"Error closing camera: {str(e)}")

    def __del__(self):
        """Cleanup on deletion"""
        self.close_capture_device()