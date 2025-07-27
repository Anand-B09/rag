# Add an empty __init__.py to make the tests directory a package
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))
