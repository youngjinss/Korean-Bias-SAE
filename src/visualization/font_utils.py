"""
Font utilities for Korean text rendering in matplotlib.
Adapted from korean-sparse-llm-features-open project.

Supports multiple platforms (macOS, Windows, Linux) with proper Korean font detection.
"""

import os
import subprocess
import platform
import warnings
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
from typing import Optional, List


# Font priority list for each platform
FONT_PRIORITIES = {
    'Darwin': [  # macOS
        'AppleGothic',
        'Apple SD Gothic Neo',
        'NanumGothic',
        'Noto Sans CJK KR',
        'Malgun Gothic',
    ],
    'Windows': [
        'Malgun Gothic',
        'NanumGothic',
        'Noto Sans CJK KR',
        'Gulim',
    ],
    'Linux': [
        'Noto Sans CJK KR',
        'NanumGothic',
        'NanumBarunGothic',
        'UnDotum',
        'DejaVu Sans',
    ]
}


def get_available_korean_fonts() -> List[str]:
    """
    Get list of available Korean-compatible fonts on the system.

    Returns:
        List of font names that support Korean
    """
    korean_font_keywords = [
        'Noto Sans CJK', 'NotoSansCJK', 'Nanum', 'Malgun', 'Gothic',
        'Apple SD', 'AppleGothic', 'Gulim', 'Dotum', 'Batang', 'Gungsuh',
        'UnDotum', 'UnBatang', 'KoPub', 'Spoqa Han Sans', 'D2Coding'
    ]

    available_fonts = []
    for font in fm.fontManager.ttflist:
        for keyword in korean_font_keywords:
            if keyword.lower() in font.name.lower():
                available_fonts.append(font.name)
                break

    return list(set(available_fonts))


def find_best_korean_font() -> Optional[str]:
    """
    Find the best available Korean font for the current platform.

    Returns:
        Font name or None if no Korean font found
    """
    system = platform.system()
    priority_fonts = FONT_PRIORITIES.get(system, FONT_PRIORITIES['Linux'])
    available_fonts = get_available_korean_fonts()

    # Also get all system fonts
    all_font_names = set(f.name for f in fm.fontManager.ttflist)

    # Try priority fonts first
    for font in priority_fonts:
        if font in all_font_names or font in available_fonts:
            return font

    # Try any available Korean font
    if available_fonts:
        return available_fonts[0]

    return None


def download_and_install_noto_font(test: bool = False) -> bool:
    """
    Download and install Noto Sans CJK KR font for Korean text rendering.

    Args:
        test: If True, display a test plot with Korean text

    Returns:
        True if font was installed successfully
    """
    # Determine font directory based on platform
    system = platform.system()
    if system == 'Darwin':  # macOS
        font_dir = Path.home() / 'Library' / 'Fonts'
    elif system == 'Windows':
        font_dir = Path(os.environ.get('LOCALAPPDATA', '')) / 'Microsoft' / 'Windows' / 'Fonts'
    else:  # Linux
        font_dir = Path.home() / '.fonts'

    font_dir.mkdir(exist_ok=True, parents=True)
    noto_font_path = font_dir / 'NotoSansCJKkr-Regular.otf'

    # Download font if not present
    if not noto_font_path.exists():
        print("Downloading Noto Sans CJK KR font...")
        font_url = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/Korean/NotoSansCJKkr-Regular.otf"

        try:
            # Try wget first
            result = subprocess.run(
                ['wget', '-q', '-O', str(noto_font_path), font_url],
                capture_output=True
            )
            if result.returncode != 0:
                # Fallback to curl
                subprocess.run(
                    ['curl', '-sL', '-o', str(noto_font_path), font_url],
                    check=True
                )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Failed to download font: {e}")
            print("Please install a Korean font manually.")
            return False

        print(f"Font downloaded to {noto_font_path}")

    # Rebuild font cache
    try:
        fm._load_fontmanager(try_read_cache=False)
    except Exception:
        # Force rebuild by clearing cache
        fm.fontManager = fm.FontManager()

    # List available fonts
    available_fonts = get_available_korean_fonts()
    if available_fonts:
        print(f"Available Korean fonts: {available_fonts[:5]}...")

    # Test font installation
    if test:
        print("Testing font installation...")
        test_korean_font()

    return True


def setup_korean_font(verbose: bool = True) -> str:
    """
    Configure matplotlib to use Korean-compatible fonts.
    Call this function at the beginning of visualization notebooks.

    Args:
        verbose: If True, print font configuration info

    Returns:
        Name of the configured font
    """
    # Find best Korean font
    font_name = find_best_korean_font()

    if font_name:
        plt.rcParams['font.family'] = font_name
    else:
        # Fallback: try common fonts directly
        system = platform.system()
        fallback_fonts = FONT_PRIORITIES.get(system, ['DejaVu Sans'])

        for font in fallback_fonts:
            try:
                plt.rcParams['font.family'] = font
                font_name = font
                break
            except Exception:
                continue

        if not font_name:
            warnings.warn(
                "No Korean font found. Korean text may not display correctly. "
                "Run download_and_install_noto_font() to install a Korean font."
            )
            font_name = 'sans-serif'
            plt.rcParams['font.family'] = font_name

    # Prevent minus sign rendering issues
    plt.rcParams['axes.unicode_minus'] = False

    # Additional font settings for better rendering
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12

    if verbose:
        print(f"Font configured: {plt.rcParams['font.family']}")

    return font_name


def get_korean_font_path() -> Optional[str]:
    """
    Get the path to an installed Korean font.

    Returns:
        Path to the font file or None
    """
    # Check common font directories
    system = platform.system()

    font_dirs = []
    if system == 'Darwin':
        font_dirs = [
            Path.home() / 'Library' / 'Fonts',
            Path('/Library/Fonts'),
            Path('/System/Library/Fonts'),
        ]
    elif system == 'Windows':
        font_dirs = [
            Path(os.environ.get('WINDIR', 'C:\\Windows')) / 'Fonts',
            Path(os.environ.get('LOCALAPPDATA', '')) / 'Microsoft' / 'Windows' / 'Fonts',
        ]
    else:
        font_dirs = [
            Path.home() / '.fonts',
            Path('/usr/share/fonts'),
            Path('/usr/local/share/fonts'),
        ]

    # Look for Noto font first
    noto_names = ['NotoSansCJKkr-Regular.otf', 'NotoSansCJK-Regular.ttc']
    for font_dir in font_dirs:
        if font_dir.exists():
            for noto_name in noto_names:
                noto_path = font_dir / noto_name
                if noto_path.exists():
                    return str(noto_path)

    # Search in matplotlib font list
    for font in fm.fontManager.ttflist:
        if 'Noto Sans CJK' in font.name or 'NotoSansCJK' in font.name:
            return font.fname
        if 'NanumGothic' in font.name:
            return font.fname

    return None


def test_korean_font() -> None:
    """Display a test plot with Korean text to verify font configuration."""
    setup_korean_font(verbose=False)

    fig, ax = plt.subplots(figsize=(10, 3))

    # Test various Korean characters
    test_texts = [
        '한글 테스트 (Hangul Test)',
        '성별, 인종, 종교, 나이',
        '편향 특성 분석 결과',
        '억제 효과: -65.4%',
    ]

    for i, text in enumerate(test_texts):
        ax.text(0.5, 0.8 - i * 0.25, text, fontsize=14, ha='center', va='center',
                transform=ax.transAxes)

    ax.set_title('Korean Font Test (한글 폰트 테스트)', fontsize=16)
    ax.axis('off')
    plt.tight_layout()
    plt.show()

    print(f"Current font: {plt.rcParams['font.family']}")


def ensure_korean_font() -> str:
    """
    Ensure Korean font is available, downloading if necessary.

    Returns:
        Name of the configured font
    """
    font_name = find_best_korean_font()

    if not font_name:
        print("No Korean font found. Attempting to download...")
        if download_and_install_noto_font():
            font_name = find_best_korean_font()

    return setup_korean_font(verbose=True)
