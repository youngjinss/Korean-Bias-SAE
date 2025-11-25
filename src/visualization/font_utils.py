"""
Font utilities for Korean text rendering in matplotlib.
Adapted from korean-sparse-llm-features-open project.
"""

import os
import subprocess
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path


def download_and_install_noto_font(test: bool = False) -> None:
    """
    Download and install Noto Sans CJK KR font for Korean text rendering.

    Args:
        test: If True, display a test plot with Korean text
    """
    font_dir = Path.home() / '.fonts'
    font_dir.mkdir(exist_ok=True)

    noto_font_path = font_dir / 'NotoSansCJKkr-Regular.otf'

    # Download font if not present
    if not noto_font_path.exists():
        print("Downloading Noto Sans CJK KR font...")
        font_url = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/Korean/NotoSansCJKkr-Regular.otf"
        subprocess.run(['wget', '-O', str(noto_font_path), font_url], check=True)
        print(f"Font downloaded to {noto_font_path}")

    # Rebuild font cache
    fm._load_fontmanager(try_read_cache=False)

    # List available fonts
    available_fonts = [f.name for f in fm.fontManager.ttflist if 'Noto' in f.name]
    print(f"Available fonts: {available_fonts}")

    # Test font installation
    if test:
        print("Testing font installation...")
        plt.figure(figsize=(10, 2))
        plt.text(0.5, 0.5, '한글 테스트 Text', fontsize=20, ha='center', va='center',
                fontfamily='Noto Sans CJK KR')
        plt.title('한글 테스트 Text')
        plt.axis('off')
        plt.tight_layout()
        plt.show()


def setup_korean_font() -> None:
    """
    Configure matplotlib to use Korean-compatible fonts.
    Call this function at the beginning of visualization notebooks.
    """
    # Try to use Noto Sans CJK KR first
    try:
        plt.rcParams['font.family'] = 'Noto Sans CJK KR'
    except:
        # Fallback to system fonts
        import platform
        system = platform.system()

        if system == 'Darwin':  # macOS
            plt.rcParams['font.family'] = 'AppleGothic'
        elif system == 'Windows':
            plt.rcParams['font.family'] = 'Malgun Gothic'
        else:  # Linux
            plt.rcParams['font.family'] = 'NanumGothic'

    # Prevent minus sign rendering issues
    plt.rcParams['axes.unicode_minus'] = False

    print(f"Font configured: {plt.rcParams['font.family']}")


def get_korean_font_path() -> str:
    """
    Get the path to the installed Korean font.

    Returns:
        Path to the font file
    """
    font_dir = Path.home() / '.fonts'
    noto_font_path = font_dir / 'NotoSansCJKkr-Regular.otf'

    if noto_font_path.exists():
        return str(noto_font_path)

    # Search in matplotlib font list
    for font in fm.fontManager.ttflist:
        if 'Noto Sans CJK' in font.name or 'NotoSansCJK' in font.name:
            return font.fname

    return None
