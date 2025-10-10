#!/bin/bash

echo "========================================="
echo "🎬 멀티 비디오 동시 처리 테스트"
echo "========================================="

cd build

VIDEO="../../media/30sec.mp4"
OUTPUT_DIR="../output"

# Create output directory if not exists
mkdir -p $OUTPUT_DIR

echo ""
echo "1️⃣  단일 비디오 (기준)"
echo "-----------------------------------------"
START=$(date +%s)
START_NS=$(date +%N)

OUTPUT=$(./bin/triton_client parallel $VIDEO $OUTPUT_DIR/single.mp4 8 2>&1)
SINGLE_FPS=$(echo "$OUTPUT" | grep "Average FPS:" | awk '{print $3}')
SINGLE_TIME=$(echo "$OUTPUT" | grep "Total time:" | awk '{print $3}' | sed 's/s//')

END=$(date +%s)
END_NS=$(date +%N)

# 시간 계산
ELAPSED=$((END - START))
ELAPSED_NS=$((END_NS - START_NS))
if [ $ELAPSED_NS -lt 0 ]; then
  ELAPSED=$((ELAPSED - 1))
  ELAPSED_NS=$((1000000000 + ELAPSED_NS))
fi
TEST1_TIME=$(python3 -c "print(f'{$ELAPSED + $ELAPSED_NS / 1e9:.2f}')")

echo "Average FPS: $SINGLE_FPS"
echo "⏱️  테스트 소요 시간: ${TEST1_TIME}초"
echo "Total throughput: $SINGLE_FPS inferences/sec"

echo ""
echo "2️⃣  2개 비디오 동시 처리 (각 8 threads)"
echo "-----------------------------------------"
START=$(date +%s)
START_NS=$(date +%N)

# 2개 비디오 병렬 실행 및 결과 저장
TEMP_DIR="$OUTPUT_DIR/multi_test_$$"
mkdir -p $TEMP_DIR

./bin/triton_client parallel $VIDEO $OUTPUT_DIR/multi1.mp4 8 2>&1 > $TEMP_DIR/out1.txt &
PID1=$!
./bin/triton_client parallel $VIDEO $OUTPUT_DIR/multi2.mp4 8 2>&1 > $TEMP_DIR/out2.txt &
PID2=$!

wait $PID1 $PID2

END=$(date +%s)
END_NS=$(date +%N)

# 시간 계산
ELAPSED=$((END - START))
ELAPSED_NS=$((END_NS - START_NS))
if [ $ELAPSED_NS -lt 0 ]; then
  ELAPSED=$((ELAPSED - 1))
  ELAPSED_NS=$((1000000000 + ELAPSED_NS))
fi
TOTAL_TIME=$(python3 -c "print(f'{$ELAPSED + $ELAPSED_NS / 1e9:.2f}')")

# FPS 추출
FPS1=$(grep "Average FPS:" $TEMP_DIR/out1.txt | awk '{print $3}')
FPS2=$(grep "Average FPS:" $TEMP_DIR/out2.txt | awk '{print $3}')

echo "Video 1: $FPS1 FPS"
echo "Video 2: $FPS2 FPS"
echo "⏱️  테스트 소요 시간: ${TOTAL_TIME}초"

# 총 throughput 계산 (900 frames * 2 videos = 1800 frames)
TOTAL_THROUGHPUT=$(python3 -c "print(f'{1800 / $TOTAL_TIME:.2f}')")
echo "📊 총 throughput: $TOTAL_THROUGHPUT inferences/sec"
TEST2_TIME=$TOTAL_TIME

echo ""
echo "3️⃣  4개 비디오 동시 처리 (각 8 threads)"
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

# 시간 계산
ELAPSED=$((END - START))
ELAPSED_NS=$((END_NS - START_NS))
if [ $ELAPSED_NS -lt 0 ]; then
  ELAPSED=$((ELAPSED - 1))
  ELAPSED_NS=$((1000000000 + ELAPSED_NS))
fi
TOTAL_TIME=$(python3 -c "print(f'{$ELAPSED + $ELAPSED_NS / 1e9:.2f}')")

# FPS 추출
FPS1=$(grep "Average FPS:" $TEMP_DIR/out1.txt | awk '{print $3}')
FPS2=$(grep "Average FPS:" $TEMP_DIR/out2.txt | awk '{print $3}')
FPS3=$(grep "Average FPS:" $TEMP_DIR/out3.txt | awk '{print $3}')
FPS4=$(grep "Average FPS:" $TEMP_DIR/out4.txt | awk '{print $3}')

echo "Video 1: $FPS1 FPS"
echo "Video 2: $FPS2 FPS"
echo "Video 3: $FPS3 FPS"
echo "Video 4: $FPS4 FPS"
echo "⏱️  테스트 소요 시간: ${TOTAL_TIME}초"

# 총 throughput 계산 (900 frames * 4 videos = 3600 frames)
TOTAL_THROUGHPUT=$(python3 -c "print(f'{3600 / $TOTAL_TIME:.2f}')")
echo "📊 총 throughput: $TOTAL_THROUGHPUT inferences/sec"
TEST3_TIME=$TOTAL_TIME

# 정리
rm -rf $TEMP_DIR

echo ""
echo "========================================="
echo "📊 최종 결과 요약"
echo "========================================="
echo "1️⃣  단일 비디오:"
echo "   • FPS: $SINGLE_FPS"
echo "   • Throughput: $SINGLE_FPS inferences/sec"
echo "   • ⏱️  소요 시간: ${TEST1_TIME}초"
echo ""
echo "2️⃣  2개 비디오 동시:"
echo "   • 개별 FPS: 평균 $(python3 -c "print(f'{($FPS1 + $FPS2) / 2:.1f}' if '$FPS1' and '$FPS2' else '?')" 2>/dev/null || echo "?")"
TOTAL_2=$(python3 -c "print(f'{1800 / $TEST2_TIME:.2f}')") 2>/dev/null || TOTAL_2="계산 실패"
echo "   • 📊 총 Throughput: $TOTAL_2 inferences/sec"
IMPROVEMENT_2=$(python3 -c "print(f'{($TOTAL_2 / $SINGLE_FPS - 1) * 100:.1f}')") 2>/dev/null || IMPROVEMENT_2="?"
echo "   • 향상: $IMPROVEMENT_2%"
echo "   • ⏱️  소요 시간: ${TEST2_TIME}초"
echo ""
echo "3️⃣  4개 비디오 동시:"
echo "   • 개별 FPS: 평균 $(python3 -c "print(f'{($FPS1 + $FPS2 + $FPS3 + $FPS4) / 4:.1f}' if '$FPS1' and '$FPS2' else '?')" 2>/dev/null || echo "?")"
TOTAL_4=$(python3 -c "print(f'{3600 / $TEST3_TIME:.2f}')") 2>/dev/null || TOTAL_4="계산 실패"
echo "   • 📊 총 Throughput: $TOTAL_4 inferences/sec"
IMPROVEMENT_4=$(python3 -c "print(f'{($TOTAL_4 / $SINGLE_FPS - 1) * 100:.1f}')") 2>/dev/null || IMPROVEMENT_4="?"
echo "   • 향상: $IMPROVEMENT_4%"
echo "   • ⏱️  소요 시간: ${TEST3_TIME}초"
echo ""
echo "⏱️  총 테스트 시간: $(python3 -c "print(f'{$TEST1_TIME + $TEST2_TIME + $TEST3_TIME:.2f}')")초"
echo "💡 멀티 비디오 처리로 총 throughput 증가!"
echo "========================================="

