# python -m config
import os

# Please modify it to your project root directory here
config_root = os.path.expanduser("")
if not os.path.exists(config_root):
    os.makedirs(config_root, exist_ok=True)

if __name__ == '__main__':
    print(config_root)