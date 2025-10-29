import sys
import os

# Force use of local f1tenth_gym
if 'f1tenth_gym' in sys.modules:
    del sys.modules['f1tenth_gym']

# Add local paths
local_f1tenth_path = os.path.join(os.path.dirname(__file__), 'f1tenth_gym')
local_src_path = os.path.join(os.path.dirname(__file__), 'src')
if local_f1tenth_path not in sys.path:
    sys.path.insert(0, local_f1tenth_path)
if local_src_path not in sys.path:
    sys.path.insert(0, local_src_path)

print("Python path:")
for i, path in enumerate(sys.path[:5]):  # Show first 5 paths
    print(f"  {i}: {path}")

print("\nTrying to import f1tenth_gym...")
try:
    import f1tenth_gym
    print(f"SUCCESS: Using f1tenth_gym from: {f1tenth_gym.__file__}")
    
    # Check if maps directory exists
    maps_path = os.path.join(os.path.dirname(f1tenth_gym.__file__), 'maps')
    print(f"Maps directory exists: {os.path.exists(maps_path)}")
    if os.path.exists(maps_path):
        catalunya_path = os.path.join(maps_path, 'Catalunya')
        print(f"Catalunya directory exists: {os.path.exists(catalunya_path)}")
        if os.path.exists(catalunya_path):
            yaml_file = os.path.join(catalunya_path, 'Catalunya_map.yaml')
            print(f"Catalunya_map.yaml exists: {os.path.exists(yaml_file)}")
    
except ImportError as e:
    print(f"FAILED: {e}")
