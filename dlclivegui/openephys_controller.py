"""Remote control interface for OpenEphys via HTTP API and Teensy serial pulse control."""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

from PySide6.QtCore import QObject, QTimer, Signal

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    logging.warning("pyserial not installed - Teensy pulse control unavailable")


@dataclass
class OpenEphysConfig:
    """Configuration for OpenEphys remote control and Teensy pulse control."""

    host: str = "localhost"
    port: int = 37497
    serial_port: str = ""  # COM port for Teensy (e.g., "COM3")
    pulse_frequency: float = 100.0  # Hz


class OpenEphysController(QObject):
    """Controller for remote communication with OpenEphys via HTTP API and Teensy pulse control.

    This class handles:
    - Starting/stopping OpenEphys recording
    - Sending pulse commands to Teensy via serial
    - Sequenced recording start/stop with pulse markers

    Signals:
        recording_started: Emitted when OpenEphys recording has started
        recording_stopped: Emitted when OpenEphys recording has stopped
        ttl_sent: Emitted when pulse has been started
        error: Emitted when an error occurs (str message)
        sequence_complete: Emitted when a start/stop sequence completes
    """

    recording_started = Signal()
    recording_stopped = Signal()
    ttl_sent = Signal()
    error = Signal(str)
    sequence_complete = Signal(str)  # "start" or "stop"

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._config = OpenEphysConfig()
        self._enabled = False
        self._serial_port: Optional[serial.Serial] = None

    @property
    def enabled(self) -> bool:
        """Whether OpenEphys control is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    @property
    def config(self) -> OpenEphysConfig:
        """Current configuration."""
        return self._config

    def configure(
        self,
        host: str = "localhost",
        port: int = 37497,
        serial_port: str = "",
        pulse_frequency: float = 100.0,
    ) -> None:
        """Update the controller configuration.

        Args:
            host: OpenEphys host address
            port: OpenEphys HTTP server port (default 37497)
            serial_port: COM port for Teensy (e.g., "COM3")
            pulse_frequency: Pulse frequency in Hz
        """
        # Close existing serial connection if port changed
        if self._serial_port is not None and self._config.serial_port != serial_port:
            try:
                self._serial_port.close()
            except Exception:
                pass
            self._serial_port = None
        
        self._config = OpenEphysConfig(
            host=host,
            port=port,
            serial_port=serial_port,
            pulse_frequency=pulse_frequency,
        )

    def _base_url(self) -> str:
        """Get the base URL for OpenEphys HTTP API."""
        return f"http://{self._config.host}:{self._config.port}/api"

    def start_recording(self) -> bool:
        """Send command to OpenEphys to start recording.

        Returns:
            True if successful, False otherwise.
        """
        url = f"{self._base_url()}/status"
        data = json.dumps({"mode": "RECORD"}).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="PUT")
        req.add_header("Content-Type", "application/json")

        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                logging.info(f"OpenEphys start recording: {response.status}")
                if response.status == 200:
                    self.recording_started.emit()
                    return True
                return False
        except urllib.error.URLError as e:
            error_msg = f"Failed to start OpenEphys recording: {e}"
            logging.error(error_msg)
            self.error.emit(error_msg)
            return False
        except Exception as e:
            error_msg = f"OpenEphys communication error: {e}"
            logging.error(error_msg)
            self.error.emit(error_msg)
            return False

    def stop_recording(self) -> bool:
        """Send command to OpenEphys to stop recording (go to ACQUIRE mode).

        Returns:
            True if successful, False otherwise.
        """
        url = f"{self._base_url()}/status"
        data = json.dumps({"mode": "ACQUIRE"}).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="PUT")
        req.add_header("Content-Type", "application/json")

        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                logging.info(f"OpenEphys stop recording: {response.status}")
                if response.status == 200:
                    self.recording_stopped.emit()
                    return True
                return False
        except urllib.error.URLError as e:
            error_msg = f"Failed to stop OpenEphys recording: {e}"
            logging.error(error_msg)
            self.error.emit(error_msg)
            return False
        except Exception as e:
            error_msg = f"OpenEphys communication error: {e}"
            logging.error(error_msg)
            self.error.emit(error_msg)
            return False

    def _ensure_serial_connection(self) -> bool:
        """Ensure serial connection to Teensy is established.

        Returns:
            True if connected, False otherwise.
        """
        if not SERIAL_AVAILABLE:
            self.error.emit("pyserial not installed - cannot control Teensy")
            return False
        
        if not self._config.serial_port:
            self.error.emit("No serial port configured for Teensy")
            return False
        
        if self._serial_port is not None and self._serial_port.is_open:
            return True
        
        try:
            self._serial_port = serial.Serial(
                self._config.serial_port,
                baudrate=115200,
                timeout=2.0
            )
            # Wait for Teensy to initialize
            time.sleep(0.5)
            # Read any startup messages
            while self._serial_port.in_waiting:
                line = self._serial_port.readline().decode('utf-8', errors='ignore').strip()
                logging.info(f"Teensy: {line}")
            return True
        except Exception as e:
            error_msg = f"Failed to connect to Teensy on {self._config.serial_port}: {e}"
            logging.error(error_msg)
            self.error.emit(error_msg)
            return False

    def send_ttl(self) -> bool:
        """Send START command to Teensy to begin pulse generation.

        Returns:
            True if successful, False otherwise.
        """
        if not self._ensure_serial_connection():
            return False
        
        try:
            frequency = self._config.pulse_frequency
            command = f"START {frequency}\n"
            self._serial_port.write(command.encode('utf-8'))
            
            # Wait for response
            response = self._serial_port.readline().decode('utf-8', errors='ignore').strip()
            logging.info(f"Teensy response: {response}")
            
            if response.startswith("OK START"):
                self.ttl_sent.emit()
                return True
            else:
                error_msg = f"Teensy error: {response}"
                logging.error(error_msg)
                self.error.emit(error_msg)
                return False
        except Exception as e:
            error_msg = f"Failed to send pulse command to Teensy: {e}"
            logging.error(error_msg)
            self.error.emit(error_msg)
            return False

    def stop_pulse(self) -> bool:
        """Send STOP command to Teensy to stop pulse generation.

        Returns:
            True if successful, False otherwise.
        """
        if not self._ensure_serial_connection():
            return False
        
        try:
            command = "STOP\n"
            self._serial_port.write(command.encode('utf-8'))
            
            # Wait for response
            response = self._serial_port.readline().decode('utf-8', errors='ignore').strip()
            logging.info(f"Teensy response: {response}")
            
            if response.startswith("OK STOP"):
                return True
            else:
                error_msg = f"Teensy error: {response}"
                logging.error(error_msg)
                self.error.emit(error_msg)
                return False
        except Exception as e:
            error_msg = f"Failed to stop Teensy pulse: {e}"
            logging.error(error_msg)
            self.error.emit(error_msg)
            return False
    
    def close_serial(self) -> None:
        """Close serial connection to Teensy."""
        if self._serial_port is not None:
            try:
                # Stop pulse before closing
                self.stop_pulse()
                self._serial_port.close()
            except Exception:
                pass
            self._serial_port = None

    def get_status(self) -> Optional[str]:
        """Query the current status of OpenEphys.

        Returns:
            The current mode ("IDLE", "ACQUIRE", "RECORD") or None on error.
        """
        url = f"{self._base_url()}/status"
        req = urllib.request.Request(url, method="GET")

        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode("utf-8"))
                return data.get("mode")
        except Exception as e:
            logging.error(f"Failed to get OpenEphys status: {e}")
            return None

    def is_connected(self) -> bool:
        """Check if OpenEphys is reachable.

        Returns:
            True if OpenEphys responds to status query, False otherwise.
        """
        return self.get_status() is not None


class OpenEphysRecordingSequencer(QObject):
    """Handles sequenced recording start/stop with TTL markers.

    This class manages the timing sequence for coordinated recording:

    Start sequence:
        1. Start OpenEphys recording
        2. Signal to start video recording
        3. After delay, send TTL pulse

    Stop sequence:
        1. Send TTL pulse
        2. After delay, signal to stop video recording
        3. Stop OpenEphys recording

    Signals:
        start_video_recording: Emitted when video recording should start
        stop_video_recording: Emitted when video recording should stop
        sequence_started: Emitted when a sequence begins (str: "start" or "stop")
        sequence_complete: Emitted when a sequence completes (str: "start" or "stop")
        error: Emitted on error (str message)
    """

    start_video_recording = Signal()
    stop_video_recording = Signal()
    sequence_started = Signal(str)
    sequence_complete = Signal(str)
    error = Signal(str)

    def __init__(
        self,
        controller: OpenEphysController,
        ttl_delay_ms: int = 1000,
        parent: Optional[QObject] = None,
    ):
        """Initialize the sequencer.

        Args:
            controller: OpenEphysController instance for communication
            ttl_delay_ms: Delay in milliseconds between video start/stop and TTL
            parent: Parent QObject
        """
        super().__init__(parent)
        self._controller = controller
        self._ttl_delay_ms = ttl_delay_ms
        self._sequence_in_progress = False

    @property
    def ttl_delay_ms(self) -> int:
        """Delay in milliseconds between video start/stop and pulse."""
        return self._ttl_delay_ms

    @ttl_delay_ms.setter
    def ttl_delay_ms(self, value: int) -> None:
        self._ttl_delay_ms = max(0, value)

    @property
    def is_busy(self) -> bool:
        """Whether a sequence is currently in progress."""
        return self._sequence_in_progress

    def start_recording_sequence(self) -> bool:
        """Begin the start recording sequence.

        Sequence:
            1. Start OpenEphys recording
            2. Emit start_video_recording signal
            3. After ttl_delay_ms, send TTL pulse
            4. Emit sequence_complete("start")

        Returns:
            True if sequence started, False if OpenEphys command failed.
        """
        if self._sequence_in_progress:
            self.error.emit("Recording sequence already in progress")
            return False

        self._sequence_in_progress = True
        self.sequence_started.emit("start")

        # Step 1: Start OpenEphys recording
        if not self._controller.start_recording():
            self._sequence_in_progress = False
            return False

        # Step 2: Start video recording
        self.start_video_recording.emit()

        # Step 3: Send TTL after delay
        QTimer.singleShot(self._ttl_delay_ms, self._send_start_ttl)

        return True

    def _send_start_ttl(self) -> None:
        """Send TTL pulse and complete start sequence."""
        self._controller.send_ttl()
        self._sequence_in_progress = False
        self.sequence_complete.emit("start")

    def stop_recording_sequence(self) -> None:
        """Begin the stop recording sequence.

        Sequence:
            1. Send TTL pulse
            2. After ttl_delay_ms, emit stop_video_recording signal
            3. Stop OpenEphys recording
            4. Emit sequence_complete("stop")
        """
        if self._sequence_in_progress:
            self.error.emit("Recording sequence already in progress")
            return

        self._sequence_in_progress = True
        self.sequence_started.emit("stop")

        # Step 1: Send TTL pulse
        self._controller.send_ttl()

        # Step 2: After delay, stop video and OpenEphys
        QTimer.singleShot(self._ttl_delay_ms, self._complete_stop_sequence)

    def _complete_stop_sequence(self) -> None:
        """Complete the stop sequence after TTL delay."""
        # Step 2: Stop pulse
        self._controller.stop_pulse()
        
        # Step 3: Stop video recording
        self.stop_video_recording.emit()

        # Step 4: Stop OpenEphys recording
        self._controller.stop_recording()

        self._sequence_in_progress = False
        self.sequence_complete.emit("stop")


def list_serial_ports() -> list[str]:
    """List available serial ports.

    Returns:
        List of available COM port names.
    """
    if not SERIAL_AVAILABLE:
        return []
    
    ports = serial.tools.list_ports.comports()
    return [port.device for port in ports]
