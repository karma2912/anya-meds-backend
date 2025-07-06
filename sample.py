import os

def get_size(path):
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except FileNotFoundError:
                pass  # In case a file was deleted during the scan
    return total

def format_size(bytes_size):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.2f} PB"

def print_folder_sizes(root_folder):
    print(f"Folder sizes in: {root_folder}\n")
    for item in os.listdir(root_folder):
        item_path = os.path.join(root_folder, item)
        if os.path.isdir(item_path):
            size = get_size(item_path)
            print(f"{item:<30} {format_size(size)}")

# Example usage
root_directory = "D:\Xray-Dataset"
print_folder_sizes(root_directory)
