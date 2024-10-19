import subprocess
import re

def find_and_shutdown_nodes(package_name):
    # Find all active nodes
    result = subprocess.run(['ros2', 'node', 'list'], stdout=subprocess.PIPE, text=True)
    active_nodes = result.stdout.splitlines()

    # Filter nodes by package name
    matching_nodes = []
    for node in active_nodes:
        info = subprocess.run(['ros2', 'node', 'info', node], stdout=subprocess.PIPE, text=True).stdout
        if re.search(f'Package: {package_name}', info):
            matching_nodes.append(node)

    # Shutdown the matching nodes
    for node in matching_nodes:
        print(f"Shutting down node: {node}")
        subprocess.run(['ros2', 'lifecycle', 'set', node, 'shutdown'])

if __name__ == "__main__":
    package_name = 'node_test'  # Change this to your package name
    find_and_shutdown_nodes(package_name)
