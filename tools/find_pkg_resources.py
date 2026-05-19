import os
import sys

for p in sys.path:
    if p and os.path.isdir(p) and 'site-packages' in p.lower():
        for root, dirs, files in os.walk(p):
            for name in dirs + files:
                if 'pkg_resources' in name:
                    print(os.path.join(root, name))

print('---done---')
