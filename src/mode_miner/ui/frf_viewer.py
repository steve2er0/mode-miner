"""FRF/Response viewer panel with interactive cursor."""

from typing import Optional, List, Tuple
import numpy as np

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QPushButton
from PySide6.QtCore import Signal
from PySide6.QtGui import QFont

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class FRFViewerWidget(QWidget):
    """FRF plot viewer with interactive peak cursor.
    
    Displays frequency response function with:
    - X-axis: Frequency (Hz)
    - Y-axis: Response magnitude (mobility v/F)
    - Interactive vertical cursor for peak selection
    
    Signals:
        peak_selected: Emitted when a peak is selected, with (frequency, magnitude)
        cursor_moved: Emitted when cursor moves, with frequency
    """
    
    peak_selected = Signal(float, float)  # frequency, magnitude
    cursor_moved = Signal(float)  # frequency
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        # Data
        self._frequencies: Optional[np.ndarray] = None
        self._magnitude: Optional[np.ndarray] = None
        self._phase: Optional[np.ndarray] = None
        self._peaks: List[Tuple[float, float]] = []  # (freq, mag) pairs
        
        # Cursor state
        self._cursor_freq: Optional[float] = None
        self._cursor_line = None
        self._dragging = False
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the UI layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        
        # Header with controls
        header_layout = QHBoxLayout()
        
        header = QLabel("FRF / Response Viewer")
        header.setFont(QFont("", -1, QFont.Bold))
        header.setStyleSheet("color: #e0e0e0; font-size: 13px;")
        header_layout.addWidget(header)
        
        header_layout.addStretch()
        
        # Cursor info
        self._cursor_label = QLabel("Cursor: --")
        self._cursor_label.setStyleSheet("color: #a0a0a0; font-size: 11px;")
        header_layout.addWidget(self._cursor_label)
        
        layout.addLayout(header_layout)
        
        # Matplotlib figure
        self._figure = Figure(figsize=(8, 3), facecolor='#1a1a2e')
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setStyleSheet("background-color: #1a1a2e;")
        layout.addWidget(self._canvas)
        
        # Create axes
        self._ax = self._figure.add_subplot(111)
        self._setup_axes()
        
        # Connect mouse events
        self._canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self._canvas.mpl_connect('button_release_event', self._on_mouse_release)
        self._canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        
        # Placeholder text
        self._show_placeholder()
    
    def _setup_axes(self):
        """Configure axes appearance."""
        self._ax.set_facecolor('#16213e')
        self._ax.set_xlabel('Frequency (Hz)', color='white', fontsize=10)
        self._ax.set_ylabel('Magnitude (v/F)', color='white', fontsize=10)
        self._ax.tick_params(colors='white', labelsize=9)
        self._ax.spines['bottom'].set_color('#4a4a6a')
        self._ax.spines['top'].set_color('#4a4a6a')
        self._ax.spines['left'].set_color('#4a4a6a')
        self._ax.spines['right'].set_color('#4a4a6a')
        self._ax.grid(True, alpha=0.3, color='#4a4a6a')
    
    def _show_placeholder(self):
        """Show placeholder text when no data."""
        self._ax.clear()
        self._setup_axes()
        self._ax.text(
            0.5, 0.5, 
            "Select input/response DOFs and compute FRF",
            ha='center', va='center',
            color='#606060', fontsize=12,
            transform=self._ax.transAxes
        )
        self._canvas.draw()
    
    def set_frf_data(
        self, 
        frequencies: np.ndarray, 
        magnitude: np.ndarray,
        phase: Optional[np.ndarray] = None,
        peaks: Optional[List[Tuple[float, float]]] = None
    ):
        """Set the FRF data to display.
        
        Args:
            frequencies: Frequency array (Hz)
            magnitude: Magnitude array (v/F)
            phase: Optional phase array (degrees)
            peaks: Optional list of (frequency, magnitude) peak locations
        """
        self._frequencies = frequencies
        self._magnitude = magnitude
        self._phase = phase
        self._peaks = peaks or []
        
        self._plot_frf()
    
    def _plot_frf(self):
        """Plot the FRF data."""
        self._ax.clear()
        self._setup_axes()
        
        if self._frequencies is None or self._magnitude is None:
            self._show_placeholder()
            return
        
        # Plot magnitude
        self._ax.semilogy(
            self._frequencies, self._magnitude,
            color='#00d4ff', linewidth=1.5, label='|H(f)|'
        )
        
        # Mark peaks
        if self._peaks:
            peak_freqs = [p[0] for p in self._peaks]
            peak_mags = [p[1] for p in self._peaks]
            self._ax.scatter(
                peak_freqs, peak_mags,
                color='#ff6b6b', s=50, zorder=5,
                marker='o', edgecolors='white', linewidths=1
            )
        
        # Draw cursor if set
        if self._cursor_freq is not None:
            self._draw_cursor()
        
        self._ax.set_xlim(self._frequencies[0], self._frequencies[-1])
        self._figure.tight_layout()
        self._canvas.draw()
    
    def _draw_cursor(self):
        """Draw the vertical cursor line."""
        if self._cursor_freq is None:
            return
        
        ylim = self._ax.get_ylim()
        self._cursor_line = self._ax.axvline(
            self._cursor_freq, color='#ffcc00', linewidth=2,
            linestyle='--', alpha=0.8
        )
    
    def set_cursor_frequency(self, freq: float):
        """Set the cursor to a specific frequency."""
        if self._frequencies is None:
            return
        
        # Clamp to valid range
        freq = np.clip(freq, self._frequencies[0], self._frequencies[-1])
        self._cursor_freq = freq
        
        # Find magnitude at this frequency
        idx = np.argmin(np.abs(self._frequencies - freq))
        mag = self._magnitude[idx]
        
        self._cursor_label.setText(f"Cursor: {freq:.2f} Hz, {mag:.2e}")
        
        self._plot_frf()
        self.cursor_moved.emit(freq)
    
    def snap_to_nearest_peak(self):
        """Snap cursor to the nearest peak."""
        if not self._peaks or self._cursor_freq is None:
            return
        
        # Find nearest peak
        peak_freqs = np.array([p[0] for p in self._peaks])
        idx = np.argmin(np.abs(peak_freqs - self._cursor_freq))
        peak_freq, peak_mag = self._peaks[idx]
        
        self._cursor_freq = peak_freq
        self._cursor_label.setText(f"Peak: {peak_freq:.2f} Hz, {peak_mag:.2e}")
        
        self._plot_frf()
        self.peak_selected.emit(peak_freq, peak_mag)
    
    def _on_mouse_press(self, event):
        """Handle mouse press."""
        if event.inaxes != self._ax:
            return
        if event.button == 1:  # Left click
            self._dragging = True
            self.set_cursor_frequency(event.xdata)
    
    def _on_mouse_release(self, event):
        """Handle mouse release."""
        if event.button == 1:
            self._dragging = False
            # Snap to peak on release
            if self._peaks:
                self.snap_to_nearest_peak()
    
    def _on_mouse_move(self, event):
        """Handle mouse move."""
        if self._dragging and event.inaxes == self._ax:
            self.set_cursor_frequency(event.xdata)
    
    def clear(self):
        """Clear the plot."""
        self._frequencies = None
        self._magnitude = None
        self._phase = None
        self._peaks = []
        self._cursor_freq = None
        self._show_placeholder()

