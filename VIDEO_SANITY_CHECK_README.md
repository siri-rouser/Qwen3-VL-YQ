# Video Loading Sanity Check - Summary

## What I Did

I've added **comprehensive sanity checks** to your training pipeline to verify that videos are being loaded correctly. Here's what was implemented:

## Changes Made

### 1. Modified `qwen-vl-finetune/qwenvl/data/data_processor.py`

Added **5 strategic checkpoints** throughout the data loading pipeline:

#### ✅ Checkpoint 1: Video Path Verification
- **Location**: `_build_messages()` function
- **What it checks**: 
  - Logs every video path being processed
  - Verifies if the video file exists on disk
- **Output**: `📹 [SANITY CHECK] Video {i}: {path} | Exists: {True/False}`

#### ✅ Checkpoint 2: Processor Output Verification  
- **Location**: `preprocess_qwen_visual()` function
- **What it checks**:
  - Confirms `pixel_values_videos` is in the processed result
  - Confirms `video_grid_thw` is in the processed result
  - Logs the shapes of video tensors
- **Output**: `✅ [SANITY CHECK] Video data loaded! pixel_values_videos shape: {...}`
- **Warning if problem**: `⚠️ [SANITY CHECK WARNING] Video in source but NOT in processed result!`

#### ✅ Checkpoint 3: Dataset Item Processing
- **Location**: `_get_item()` method
- **What it checks**:
  - Logs video grid dimensions after processing
- **Output**: `🎬 [SANITY CHECK] Video processed in _get_item: video_grid_thw=[...]`

#### ✅ Checkpoint 4: Batch Collation (Regular)
- **Location**: `DataCollatorForSupervisedDataset.__call__()`
- **What it checks**:
  - Number of videos in each batch
  - Shape of concatenated video tensors
  - Video grid dimensions
- **Output**: `🎥 [SANITY CHECK] BATCH has {n} videos! concat_videos shape: {...}, video_grid_thw: [...]`

#### ✅ Checkpoint 5: Batch Collation (Flattened)
- **Location**: `FlattenedDataCollatorForSupervisedDataset.__call__()`
- **What it checks**: Same as Checkpoint 4 but for flattened/packed data
- **Output**: `🎥 [SANITY CHECK FLATTENED] BATCH has {n} videos! concat_videos shape: {...}`

## Testing Script

### Created `test_video_loading.py`

A standalone test script that runs **4 comprehensive tests** without needing to run full training:

1. **Test 1: Video File Existence**
   - Checks if video files exist on disk
   - Shows file sizes

2. **Test 2: Processor Loading**
   - Loads the Qwen3-VL processor
   - Verifies video processor configuration
   - Shows video processing parameters

3. **Test 3: Single Sample Processing**
   - Processes one video sample through the pipeline
   - Verifies `pixel_values_videos` and `video_grid_thw` are generated

4. **Test 4: Dataset Creation**
   - Creates the full dataset
   - Loads first 3 samples
   - Verifies video data is present in each sample

## How to Use

### Option 1: Run the Test Script (Recommended First)

```bash
cd /homes/yl4300/Qwen3-VL
python test_video_loading.py
```

This will quickly tell you if videos are loading correctly **without starting training**.

**What to look for:**
- ✅ All tests should pass
- ✅ You should see "pixel_values_videos" tensors with actual data
- ✅ Video file sizes should be shown
- ✅ No "FILE NOT FOUND" errors

### Option 2: Run Training with Sanity Checks

```bash
cd /homes/yl4300/Qwen3-VL/qwen-vl-finetune
bash scripts/sft_qwen3_4b.sh
```

**What to look for in training logs:**
- 📹 Video paths with "Exists: True"
- ✅ "Video data loaded!" messages
- 🎬 Video grid dimensions
- 🎥 Batch statistics showing "BATCH has X videos"

**Red flags to watch for:**
- ❌ "Exists: False" for video paths
- ⚠️ "Video in source but NOT in processed result"
- ❌ No video-related sanity check messages at all

## What Each Check Tells You

| Check Point | What It Means If You See It | What It Means If You DON'T See It |
|-------------|------------------------------|-----------------------------------|
| 📹 Video paths | Videos are being found in dataset | Dataset might not contain video samples |
| ✅ Video data loaded | Processor successfully loaded video frames | Video processing failed or skipped |
| 🎬 Video grid | Video tokens are being generated | Video encoding problem |
| 🎥 Batch statistics | Videos are making it to the model | Collation dropping videos |

## Expected Output Example

If everything works, you should see output like:

```
📹 [SANITY CHECK] Video 0: /data/OTA/MononElmStreetNB/MononElmStreetNB_av_101_29_3.mp4 | Exists: True
✅ [SANITY CHECK] Video data loaded! pixel_values_videos shape: torch.Size([1, 768, 1176]), video_grid_thw shape: torch.Size([1, 3])
🎬 [SANITY CHECK] Video processed in _get_item: video_grid_thw=[8, 14, 21]
🎥 [SANITY CHECK] BATCH has 1 videos! concat_videos shape: torch.Size([1, 768, 1176]), video_grid_thw: [[8, 14, 21]]
```

## Troubleshooting

### If videos are NOT loading:

1. **Check video paths**: Make sure video files exist at the paths in your JSON
2. **Check data_path**: Verify the `data_path` in your data_list configuration
3. **Check <video> token**: Ensure your training data has `<video>` placeholder in the text
4. **Check processor**: Verify the processor has a `video_processor` attribute

### If you see warnings:

- **"Video in source but NOT in processed result"**: 
  - Video file might be corrupted
  - Video format might not be supported
  - Processor might have failed silently

## Quick Verification Checklist

Run through this checklist:

- [ ] Run `test_video_loading.py` - does it pass all 4 tests?
- [ ] Check first video file manually - does it exist and play?
- [ ] Look at training logs - do you see the sanity check messages?
- [ ] Check batch logs - are videos appearing in batches?
- [ ] Monitor GPU usage - does it spike when processing videos?

## Summary

Your training pipeline now has **5 strategic checkpoints** that will tell you:
1. ✅ If video files exist
2. ✅ If videos are being loaded by the processor
3. ✅ If video tokens are being generated
4. ✅ If videos are in the training batches
5. ✅ The exact shapes and dimensions of video data

**These checks run during training at rank 0 only**, so you'll see clear output without flooding your logs.

Good luck with your training! 🚀
