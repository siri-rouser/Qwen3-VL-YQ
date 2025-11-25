# # #!/bin/bash
# # set -e  # stop on first error
# # set -o pipefail

# # # ----------------------------------------
# # # Directories
# # # ----------------------------------------
# # CACHE_DIR="/root/.cache/huggingface/avtau_video"
# # TARGET_DIR="/root/.cache/huggingface/avtau_video/videos"

# # mkdir -p "$CACHE_DIR/archives"
# # mkdir -p "$TARGET_DIR"

# # # ----------------------------------------
# # # Loop through all 8 parts
# # # ----------------------------------------
# # for i in {1..8}; do
# #     echo "=============================="
# #     echo "📥 Downloading videos_part${i}.tar.gz"
# #     echo "=============================="

# #     huggingface-cli download harryhsing/AV-TAU "archives/videos_part${i}.tar.gz" \
# #         --repo-type dataset \
# #         --local-dir "$CACHE_DIR/archives" \
# #         --local-dir-use-symlinks False

# #     # Move tarball to a predictable name
# #     TAR_FILE=$(find "$CACHE_DIR/archives" -name "videos_part${i}.tar.gz" | head -n 1)

# #     if [ -z "$TAR_FILE" ]; then
# #         echo "❌ Could not find videos_part${i}.tar.gz after download!"
# #         exit 1
# #     fi

# #     echo "📦 Extracting $TAR_FILE ..."
# #     tar -xzf "$TAR_FILE" -C "$TARGET_DIR"

# #     echo "✅ Extracted videos_part${i}.tar.gz to $TARGET_DIR"
# # done

# # echo "🎉 All videos extracted successfully!"
# # echo "You can now verify with: ls -lh $TARGET_DIR | head"

# for i in {1..8};do 
#     huggingface-cli download harryhsing/AV-TAU archives/videos_part${i}.tar.gz --repo-type dataset
# done

SRC_DIR="/root/.cache/huggingface/hub/datasets--harryhsing--AV-TAU/snapshots/147bf0c6bb607e66830107053a1ae2394a0b63ce/archives"
DEST_DIR="/root/.cache/huggingface/datasets/AV-TAU/videos"

mkdir -p "$DEST_DIR"

echo "Extracting all AV-TAU archives..."
for tarfile in "$SRC_DIR"/videos_part*.tar.gz; do
    echo "----------------------------------------"
    echo "📦 Extracting $(basename "$tarfile")"
    echo "----------------------------------------"
    tar -xzf "$tarfile" -C "$DEST_DIR"
done

echo "✅ All video archives extracted successfully!"
echo "📁 Videos are now in: $DEST_DIR"
echo "👉 Example check: ls -lh $DEST_DIR | head"