#!/usr/bin/env python3
"""Wavemap - Structural dynamics visualization tool."""

import sys
import os
from pathlib import Path

# Configure Qt API before any Qt imports
os.environ['QT_API'] = 'pyside6'

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent / "src"))

from PySide6.QtWidgets import QApplication
from wavemap.ui.main_window import MainWindow


def main():
    """Launch the Wavemap application."""
    # Ensure proper Qt attributes for macOS
    if sys.platform == 'darwin':
        # Enable high-DPI support
        os.environ.setdefault('QT_ENABLE_HIGHDPI_SCALING', '1')
    
    app = QApplication(sys.argv)
    app.setApplicationName("Wavemap")
    
    window = MainWindow()
    window.show()
    
    # Process events to ensure window is visible before VTK initializes
    app.processEvents()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

