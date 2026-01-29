"""Remote control interface for OpenEphys via HTTP API."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

from PySide6.QtCore import QObject, QTimer, Signal


@dataclass
class OpenEphysConfig:
    """Configuration for OpenEphys remote control."""

    host: str = "localhost"
    port: int = 37497
    ttl_line: int = 1
    ttl_duration: int = 500  # milliseconds


class OpenEphysController(QObject):
    """Controller for remote communication with OpenEphys via HTTP API.

    This class handles:
    - Starting/stopping OpenEphys recording
    - Sending TTL pulses via broadcast messages
    - Sequenced recording start/stop with TTL markers

    Signals:
        recording_started: Emitted when OpenEphys recording has started
        recording_stopped: Emitted when OpenEphys recording has stopped
        ttl_sent: Emitted when TTL pulse has been sent
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
        ttl_line: int = 1,
        ttl_duration: int = 500,
    ) -> None:
        """Update the controller configuration.

        Args:
            host: OpenEphys host address
            port: OpenEphys HTTP server port (default 37497)
            ttl_line: Digital output line for TTL pulse (1-8)
            ttl_duration: TTL pulse duration in milliseconds
        """
        self._config = OpenEphysConfig(
            host=host,
            port=port,
            ttl_line=ttl_line,
            ttl_duration=ttl_duration,
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

    def send_ttl(self) -> bool:
        """Send TTL pulse command to OpenEphys via broadcast message.

        Uses the ACQBOARD TRIGGER command format.

        Returns:
            True if successful, False otherwise.
        """
        ttl_line = self._config.ttl_line
        ttl_duration = self._config.ttl_duration

        url = f"{self._base_url()}/message"
        # ACQBOARD TRIGGER command format: ACQBOARD TRIGGER <line> <duration_ms>
        message = f"ACQBOARD TRIGGER {ttl_line} {ttl_duration}"
        data = json.dumps({"text": message}).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="PUT")
        req.add_header("Content-Type", "application/json")

        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                logging.info(
                    f"OpenEphys TTL sent (line {ttl_line}, {ttl_duration}ms): {response.status}"
                )
                if response.status == 200:
                    self.ttl_sent.emit()
                    return True
                return False
        except urllib.error.URLError as e:
            error_msg = f"Failed to send OpenEphys TTL: {e}"
            logging.error(error_msg)
            self.error.emit(error_msg)
            return False
        except Exception as e:
            error_msg = f"OpenEphys communication error: {e}"
            logging.error(error_msg)
            self.error.emit(error_msg)
            return False

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
        """Delay between video recording and TTL pulse in milliseconds."""
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
        # Step 2: Stop video recording
        self.stop_video_recording.emit()

        # Step 3: Stop OpenEphys recording
        self._controller.stop_recording()

        self._sequence_in_progress = False
        self.sequence_complete.emit("stop")
