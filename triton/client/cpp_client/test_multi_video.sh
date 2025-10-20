#!/bin/bash

echo "========================================="
echo "Multi-Video Concurrent Processing Test"
echo "========================================="

cd build

VIDEO="../../media/30sec.mp4"
OUTPUT_DIR="../output"

# Create output directory if not exists
mkdir -p $OUTPUT_DIR

echo ""
echo "TEST 1: Single Video (Baseline)"
echo "-----------------------------------------"
START=$(date +%s)
START_NS=$(date +%N)

OUTPUT=$(./bin/triton_client parallel $VIDEO $OUTPUT_DIR/single.mp4 8 2>&1)
SINGLE_FPS=$(echo "$OUTPUT" | grep "Average FPS:" | awk '{print $3}')
SINGLE_TIME=$(echo "$OUTPUT" | grep "Total time:" | awk '{print $3}' | sed 's/s//')

END=$(date +%s)
END_NS=$(date +%N)

# Calculate time
ELAPSED=$((END - START))
ELAPSED_NS=$((END_NS - START_NS))
if [ $ELAPSED_NS -lt 0 ]; then
  ELAPSED=$((ELAPSED - 1))
  ELAPSED_NS=$((1000000000 + ELAPSED_NS))
fi
TEST1_TIME=$(python3 -c "print(f'{$ELAPSED + $ELAPSED_NS / 1e9:.2f}')")

echo "Average FPS: $SINGLE_FPS"
echo "Test duration: ${TEST1_TIME}s"
echo "Total throughput: $SINGLE_FPS inferences/sec"

echo ""
echo "TEST 2: 2 Videos Concurrent (8 threads each)"
echo "-----------------------------------------"
START=$(date +%s)
START_NS=$(date +%N)

# Run 2 videos in parallel
TEMP_DIR="$OUTPUT_DIR/multi_test_$$"
mkdir -p $TEMP_DIR

./bin/triton_client parallel $VIDEO $OUTPUT_DIR/multi1.mp4 8 2>&1 > $TEMP_DIR/out1.txt &
PID1=$!
./bin/triton_client parallel $VIDEO $OUTPUT_DIR/multi2.mp4 8 2>&1 > $TEMP_DIR/out2.txt &
PID2=$!

wait $PID1 $PID2

END=$(date +%s)
END_NS=$(date +%N)

# Calculate time
ELAPSED=$((END - START))
ELAPSED_NS=$((END_NS - START_NS))
if [ $ELAPSED_NS -lt 0 ]; then
  ELAPSED=$((ELAPSED - 1))
  ELAPSED_NS=$((1000000000 + ELAPSED_NS))
fi
TOTAL_TIME=$(python3 -c "print(f'{$ELAPSED + $ELAPSED_NS / 1e9:.2f}')")

# Extract FPS
FPS1=$(grep "Average FPS:" $TEMP_DIR/out1.txt | awk '{print $3}')
FPS2=$(grep "Average FPS:" $TEMP_DIR/out2.txt | awk '{print $3}')

echo "Video 1: $FPS1 FPS"
echo "Video 2: $FPS2 FPS"
echo "Test duration: ${TOTAL_TIME}s"

# Calculate total throughput (900 frames * 2 videos = 1800 frames)
TOTAL_THROUGHPUT=$(python3 -c "print(f'{1800 / $TOTAL_TIME:.2f}')")
echo "Total throughput: $TOTAL_THROUGHPUT inferences/sec"
TEST2_TIME=$TOTAL_TIME

echo ""
echo "TEST 3: 4 Videos Concurrent (8 threads each)"
echo "-----------------------------------------"
START=$(date +%s)
START_NS=$(date +%N)

./bin/triton_client parallel $VIDEO $OUTPUT_DIR/multi1.mp4 8 2>&1 > $TEMP_DIR/out1.txt &
PID1=$!
./bin/triton_client parallel $VIDEO $OUTPUT_DIR/multi2.mp4 8 2>&1 > $TEMP_DIR/out2.txt &
PID2=$!
./bin/triton_client parallel $VIDEO $OUTPUT_DIR/multi3.mp4 8 2>&1 > $TEMP_DIR/out3.txt &
PID3=$!
./bin/triton_client parallel $VIDEO $OUTPUT_DIR/multi4.mp4 8 2>&1 > $TEMP_DIR/out4.txt &
PID4=$!

wait $PID1 $PID2 $PID3 $PID4

END=$(date +%s)
END_NS=$(date +%N)

# Calculate time
ELAPSED=$((END - START))
ELAPSED_NS=$((END_NS - START_NS))
if [ $ELAPSED_NS -lt 0 ]; then
  ELAPSED=$((ELAPSED - 1))
  ELAPSED_NS=$((1000000000 + ELAPSED_NS))
fi
TOTAL_TIME=$(python3 -c "print(f'{$ELAPSED + $ELAPSED_NS / 1e9:.2f}')")

# Extract FPS
FPS1=$(grep "Average FPS:" $TEMP_DIR/out1.txt | awk '{print $3}')
FPS2=$(grep "Average FPS:" $TEMP_DIR/out2.txt | awk '{print $3}')
FPS3=$(grep "Average FPS:" $TEMP_DIR/out3.txt | awk '{print $3}')
FPS4=$(grep "Average FPS:" $TEMP_DIR/out4.txt | awk '{print $3}')

echo "Video 1: $FPS1 FPS"
echo "Video 2: $FPS2 FPS"
echo "Video 3: $FPS3 FPS"
echo "Video 4: $FPS4 FPS"
echo "Test duration: ${TOTAL_TIME}s"

# Calculate total throughput (900 frames * 4 videos = 3600 frames)
TOTAL_THROUGHPUT=$(python3 -c "print(f'{3600 / $TOTAL_TIME:.2f}')")
echo "Total throughput: $TOTAL_THROUGHPUT inferences/sec"
TEST3_TIME=$TOTAL_TIME

# Cleanup
rm -rf $TEMP_DIR

echo ""
echo "========================================="
echo "FINAL SUMMARY"
echo "========================================="
echo "TEST 1 - Single Video:"
echo "   - FPS: $SINGLE_FPS"
echo "   - Throughput: $SINGLE_FPS inferences/sec"
echo "   - Duration: ${TEST1_TIME}s"
echo ""
echo "TEST 2 - 2 Videos Concurrent:"
echo "   - Individual FPS: avg $(python3 -c "print(f'{($FPS1 + $FPS2) / 2:.1f}' if '$FPS1' and '$FPS2' else '?')" 2>/dev/null || echo "?")"
TOTAL_2=$(python3 -c "print(f'{1800 / $TEST2_TIME:.2f}')") 2>/dev/null || TOTAL_2="N/A"
echo "   - Total Throughput: $TOTAL_2 inferences/sec"
IMPROVEMENT_2=$(python3 -c "print(f'{($TOTAL_2 / $SINGLE_FPS - 1) * 100:.1f}')") 2>/dev/null || IMPROVEMENT_2="N/A"
echo "   - Improvement: $IMPROVEMENT_2%"
echo "   - Duration: ${TEST2_TIME}s"
echo ""
echo "TEST 3 - 4 Videos Concurrent:"
echo "   - Individual FPS: avg $(python3 -c "print(f'{($FPS1 + $FPS2 + $FPS3 + $FPS4) / 4:.1f}' if '$FPS1' and '$FPS2' else '?')" 2>/dev/null || echo "?")"
TOTAL_4=$(python3 -c "print(f'{3600 / $TEST3_TIME:.2f}')") 2>/dev/null || TOTAL_4="N/A"
echo "   - Total Throughput: $TOTAL_4 inferences/sec"
IMPROVEMENT_4=$(python3 -c "print(f'{($TOTAL_4 / $SINGLE_FPS - 1) * 100:.1f}')") 2>/dev/null || IMPROVEMENT_4="N/A"
echo "   - Improvement: $IMPROVEMENT_4%"
echo "   - Duration: ${TEST3_TIME}s"
echo ""
echo "Total test time: $(python3 -c "print(f'{$TEST1_TIME + $TEST2_TIME + $TEST3_TIME:.2f}')")s"
echo "Multi-video processing increases total throughput!"
echo "========================================="
