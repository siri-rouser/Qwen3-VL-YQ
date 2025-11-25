from datasets import load_dataset
import shutil, os

ds = load_dataset("harryhsing/AV-TAU")

for i in range(5):
    video1 = ds["train"]["video"][i]
    
    # Inspect the video data
    print(f"\n--- Video {i+1} ---")
    print(f"Type: {type(video1)}")
    print(f"Video data: {video1}")
    
    # Check if it's a file path or actual video data
    if isinstance(video1, str):
        print(f"Video path: {video1}")
        if os.path.exists(video1):
            print(f"File exists: Yes")
            print(f"File size: {os.path.getsize(video1)} bytes")
        else:
            print(f"File exists: No")
    elif hasattr(video1, 'read'):
        print("Video appears to be a file-like object")
        # Try to get some info without consuming the stream
        if hasattr(video1, 'name'):
            print(f"File name: {video1.name}")
        if hasattr(video1, 'tell'):
            pos = video1.tell()
            print(f"Current position: {pos}")
    elif hasattr(video1, '__len__'):
        print(f"Video data length: {len(video1)}")
    
    # Print first few characters/bytes if it's string or bytes
    if isinstance(video1, (str, bytes)):
        preview = str(video1)[:100] + "..." if len(str(video1)) > 100 else str(video1)
        print(f"Preview: {preview}")
    
    print("-" * 40)

