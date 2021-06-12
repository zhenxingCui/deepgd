import os
import subprocess

path = os.path.dirname(os.path.abspath(__file__))
output = subprocess.check_output([f'{path}/hde/pmds'])

print('=' * 100)
print(output.decode())