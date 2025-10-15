#!/usr/bin/env python3
"""
High concurrency load test for JPEG ensemble model
"""
import numpy as np
import cv2
import tritonclient.grpc as grpcclient
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import sys

def create_jpeg_image(quality=90):
    """테스트용 JPEG 이미지 생성"""
    image = np.random.randint(0, 255, (800, 800, 3), dtype=np.uint8)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, encoded_image = cv2.imencode('.jpg', image, encode_param)
    if not success:
        raise ValueError("Failed to encode image")
    return encoded_image.tobytes()

def single_inference(client, jpeg_bytes, model_name):
    """단일 추론 실행"""
    try:
        start = time.time()
        
        # 입력 준비
        jpeg_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        inputs = [grpcclient.InferInput("IMAGE_BYTES", [len(jpeg_bytes)], "UINT8")]
        inputs[0].set_data_from_numpy(jpeg_array)
        outputs = [grpcclient.InferRequestedOutput("OUTPUT__0")]
        
        # 추론 실행
        result = client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
        
        end = time.time()
        latency = (end - start) * 1000  # ms
        
        return {'success': True, 'latency': latency}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def burst_test(url, model_name, num_requests, num_workers, quality):
    """Burst 모드: 한번에 많은 요청 전송"""
    print(f"\n{'='*60}")
    print(f"Burst Test Configuration:")
    print(f"  Server: {url}")
    print(f"  Model: {model_name}")
    print(f"  Total Requests: {num_requests}")
    print(f"  Concurrent Workers: {num_workers}")
    print(f"  JPEG Quality: {quality}")
    print(f"{'='*60}\n")
    
    # 클라이언트 생성
    client = grpcclient.InferenceServerClient(url=url)
    
    if not client.is_server_live() or not client.is_model_ready(model_name):
        print("❌ Server or model is not ready")
        return
    
    # JPEG 이미지 생성
    print("Creating test JPEG image...")
    jpeg_bytes = create_jpeg_image(quality)
    print(f"JPEG size: {len(jpeg_bytes)} bytes ({len(jpeg_bytes)/1024:.1f} KB)\n")
    
    # 부하 테스트 실행
    print(f"Sending {num_requests} requests with {num_workers} concurrent workers...")
    
    start_time = time.time()
    results = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(single_inference, client, jpeg_bytes, model_name) 
                   for _ in range(num_requests)]
        
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            
            if i % 50 == 0:
                success_count = sum(1 for r in results if r['success'])
                print(f"  Progress: {i}/{num_requests} ({success_count} successful)")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # 결과 분석
    successes = [r for r in results if r['success']]
    failures = [r for r in results if not r['success']]
    
    if successes:
        latencies = [r['latency'] for r in successes]
        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)
        
        print(f"\n{'='*60}")
        print(f"Burst Test Results:")
        print(f"{'='*60}")
        print(f"  Total Requests: {num_requests}")
        print(f"  Successful: {len(successes)}")
        print(f"  Failed: {len(failures)}")
        print(f"  Success Rate: {len(successes)/num_requests*100:.2f}%")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  Throughput: {len(successes)/total_time:.2f} requests/sec")
        print(f"\n  Latency Statistics (ms):")
        print(f"    Mean:   {np.mean(latencies_sorted):.2f}")
        print(f"    Median: {latencies_sorted[n // 2]:.2f}")
        print(f"    P90:    {latencies_sorted[int(n * 0.9)]:.2f}")
        print(f"    P95:    {latencies_sorted[int(n * 0.95)]:.2f}")
        print(f"    P99:    {latencies_sorted[int(n * 0.99)]:.2f}")
        print(f"    Min:    {latencies_sorted[0]:.2f}")
        print(f"    Max:    {latencies_sorted[-1]:.2f}")
        print(f"{'='*60}\n")
    
    if failures:
        print(f"⚠️  {len(failures)} requests failed")
        print(f"Sample errors: {failures[:3]}")

def sustained_test(url, model_name, duration, num_workers, quality):
    """Sustained 모드: 지속적인 부하"""
    print(f"\n{'='*60}")
    print(f"Sustained Test Configuration:")
    print(f"  Server: {url}")
    print(f"  Model: {model_name}")
    print(f"  Duration: {duration}s")
    print(f"  Concurrent Workers: {num_workers}")
    print(f"  JPEG Quality: {quality}")
    print(f"{'='*60}\n")
    
    # 클라이언트 풀 생성
    clients = [grpcclient.InferenceServerClient(url=url) for _ in range(num_workers)]
    
    # JPEG 이미지 생성
    jpeg_bytes = create_jpeg_image(quality)
    print(f"JPEG size: {len(jpeg_bytes)} bytes\n")
    
    # 공유 카운터
    counter = {'success': 0, 'failure': 0, 'latencies': []}
    lock = threading.Lock()
    stop_event = threading.Event()
    
    def worker(worker_id):
        """워커 스레드"""
        client = clients[worker_id % len(clients)]
        while not stop_event.is_set():
            result = single_inference(client, jpeg_bytes, model_name)
            with lock:
                if result['success']:
                    counter['success'] += 1
                    counter['latencies'].append(result['latency'])
                else:
                    counter['failure'] += 1
    
    # 워커 시작
    print(f"Starting {num_workers} workers for {duration} seconds...\n")
    threads = []
    start_time = time.time()
    
    for i in range(num_workers):
        t = threading.Thread(target=worker, args=(i,))
        t.start()
        threads.append(t)
    
    # 실행 시간 대기
    time.sleep(duration)
    
    # 종료
    stop_event.set()
    for t in threads:
        t.join()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # 결과
    with lock:
        total = counter['success'] + counter['failure']
        latencies = sorted(counter['latencies'])
        n = len(latencies)
        
        print(f"\n{'='*60}")
        print(f"Sustained Test Results:")
        print(f"{'='*60}")
        print(f"  Total Requests: {total}")
        print(f"  Successful: {counter['success']}")
        print(f"  Failed: {counter['failure']}")
        print(f"  Success Rate: {counter['success']/total*100:.2f}%")
        print(f"  Duration: {total_time:.2f}s")
        print(f"  Throughput: {counter['success']/total_time:.2f} requests/sec")
        
        if latencies:
            print(f"\n  Latency Statistics (ms):")
            print(f"    Mean:   {np.mean(latencies):.2f}")
            print(f"    Median: {latencies[n // 2]:.2f}")
            print(f"    P90:    {latencies[int(n * 0.9)]:.2f}")
            print(f"    P95:    {latencies[int(n * 0.95)]:.2f}")
            print(f"    P99:    {latencies[int(n * 0.99)]:.2f}")
            print(f"    Min:    {min(latencies):.2f}")
            print(f"    Max:    {max(latencies):.2f}")
        print(f"{'='*60}\n")

def sustained_test_with_rate(url, model_name, duration, num_workers, target_rate=None, quality=90):
    """
    Request rate 제어가 가능한 sustained test
    target_rate: None이면 최대 속도, 숫자면 해당 RPS로 제한
    Returns dict with metrics for summary reporting
    """
    # 클라이언트 풀
    clients = [grpcclient.InferenceServerClient(url=url) for _ in range(num_workers)]
    
    # JPEG 이미지
    jpeg_bytes = create_jpeg_image(quality)
    
    # 공유 상태
    counter = {'success': 0, 'failure': 0, 'latencies': []}
    lock = threading.Lock()
    stop_event = threading.Event()
    
    def worker_with_rate(worker_id):
        """Rate limit이 있는 워커"""
        client = clients[worker_id % len(clients)]
        # 각 워커가 보내야 하는 요청 간격 (초)
        interval = num_workers / target_rate if target_rate else 0
        
        while not stop_event.is_set():
            loop_start = time.time()
            result = single_inference(client, jpeg_bytes, model_name)
            
            with lock:
                if result['success']:
                    counter['success'] += 1
                    counter['latencies'].append(result['latency'])
                else:
                    counter['failure'] += 1
            
            # Rate limiting
            if target_rate:
                elapsed = time.time() - loop_start
                sleep_time = interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
    
    def worker_unlimited(worker_id):
        """최대 속도로 요청하는 워커"""
        client = clients[worker_id % len(clients)]
        while not stop_event.is_set():
            result = single_inference(client, jpeg_bytes, model_name)
            with lock:
                if result['success']:
                    counter['success'] += 1
                    counter['latencies'].append(result['latency'])
                else:
                    counter['failure'] += 1
    
    # 워커 시작
    threads = []
    start_time = time.time()
    
    worker_func = worker_with_rate if target_rate else worker_unlimited
    
    for i in range(num_workers):
        t = threading.Thread(target=worker_func, args=(i,))
        t.start()
        threads.append(t)
    
    # 실행
    time.sleep(duration)
    stop_event.set()
    
    for t in threads:
        t.join()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # 결과 리턴
    with lock:
        if counter['latencies']:
            latencies = sorted(counter['latencies'])
            n = len(latencies)
            
            result = {
                'total_requests': counter['success'] + counter['failure'],
                'success': counter['success'],
                'failure': counter['failure'],
                'duration': total_time,
                'throughput': counter['success'] / total_time,
                'mean_latency': np.mean(latencies),
                'median_latency': latencies[n // 2],
                'p95_latency': latencies[int(n * 0.95)],
                'p99_latency': latencies[int(n * 0.99)],
                'min_latency': min(latencies),
                'max_latency': max(latencies)
            }
            
            print(f"  ✓ Requests: {result['success']}, "
                  f"Throughput: {result['throughput']:.2f} RPS, "
                  f"Latency: {result['mean_latency']:.2f} ms (mean), "
                  f"{result['p99_latency']:.2f} ms (P99)")
            
            return result
    
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='JPEG Load Tester for Triton (perf_analyzer style)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Burst test: 300 concurrent requests
  %(prog)s -m yolov11_ensemble --requests 300 --workers 300
  
  # Sustained test: 30 seconds with 100 workers
  %(prog)s -m yolov11_ensemble --mode sustained --workers 100 --duration 30
  
  # Request rate sweep (like perf_analyzer --request-rate-range)
  %(prog)s -m yolov11_ensemble --request-rate-range 10:200:10 --workers 32
  
  # Concurrency sweep (like perf_analyzer --concurrency-range)
  %(prog)s -m yolov11_ensemble --concurrency-range 1:16:1 --duration 10
        """)
    
    # Server configuration
    parser.add_argument('-m', '--model', type=str, default='yolov11_ensemble',
                        help='Model name (default: yolov11_ensemble)')
    parser.add_argument('-u', '--url', type=str, default='localhost:8001',
                        help='Server URL (default: localhost:8001)')
    
    # Test configuration
    parser.add_argument('--mode', type=str, choices=['burst', 'sustained'], default='burst',
                        help='Test mode: burst or sustained (default: burst)')
    parser.add_argument('--requests', type=int, default=300,
                        help='Number of requests for burst test (default: 300)')
    parser.add_argument('--workers', type=int, default=300,
                        help='Number of concurrent workers (default: 300)')
    parser.add_argument('--duration', type=int, default=30,
                        help='Duration for sustained test in seconds (default: 30)')
    
    # Request rate testing (like perf_analyzer)
    parser.add_argument('--request-rate-range', type=str, default=None,
                        help='Request rate range: start:end:step (e.g., 10:200:10)')
    parser.add_argument('--concurrency-range', type=str, default=None,
                        help='Concurrency range: start:end:step (e.g., 1:16:1)')
    
    # JPEG configuration
    parser.add_argument('--quality', type=int, default=90,
                        help='JPEG quality 1-100 (default: 90)')
    
    args = parser.parse_args()
    
    # Request rate range test
    if args.request_rate_range:
        try:
            parts = args.request_rate_range.split(':')
            start = int(parts[0])
            end = int(parts[1])
            step = int(parts[2]) if len(parts) > 2 else 10
        except:
            print("Error: Invalid request-rate-range format. Use: start:end:step")
            sys.exit(1)
        
        print(f"\n{'='*60}")
        print(f"Request Rate Range Test: {start} to {end} RPS (step {step})")
        print(f"{'='*60}")
        
        results = []
        for rate in range(start, end + 1, step):
            print(f"\n>>> Testing at {rate} RPS...")
            result = sustained_test_with_rate(args.url, args.model, args.duration, args.workers, rate, args.quality)
            if result:
                results.append((rate, result))
            time.sleep(2)
        
        # Summary
        print(f"\n{'='*60}")
        print(f"Request Rate Range Summary:")
        print(f"{'='*60}")
        print(f"{'Target RPS':<15} {'Actual RPS':<15} {'Latency (ms)':<20} {'P99 (ms)'}")
        print("-" * 60)
        for rate, r in results:
            print(f"{rate:<15} {r['throughput']:<15.2f} {r['mean_latency']:<20.2f} {r['p99_latency']:.2f}")
        print(f"{'='*60}\n")
    
    # Concurrency range test
    elif args.concurrency_range:
        try:
            parts = args.concurrency_range.split(':')
            start = int(parts[0])
            end = int(parts[1])
            step = int(parts[2]) if len(parts) > 2 else 1
        except:
            print("Error: Invalid concurrency-range format. Use: start:end:step")
            sys.exit(1)
        
        print(f"\n{'='*60}")
        print(f"Concurrency Range Test: {start} to {end} workers (step {step})")
        print(f"{'='*60}")
        
        results = []
        for workers in range(start, end + 1, step):
            print(f"\n>>> Testing with {workers} concurrent workers...")
            result = sustained_test_with_rate(args.url, args.model, args.duration, workers, None, args.quality)
            if result:
                results.append((workers, result))
            time.sleep(2)
        
        # Summary
        print(f"\n{'='*60}")
        print(f"Concurrency Range Summary:")
        print(f"{'='*60}")
        print(f"{'Workers':<15} {'Throughput':<15} {'Latency (ms)':<20} {'P99 (ms)'}")
        print("-" * 60)
        for workers, r in results:
            print(f"{workers:<15} {r['throughput']:<15.2f} {r['mean_latency']:<20.2f} {r['p99_latency']:.2f}")
        print(f"{'='*60}\n")
    
    # Standard burst or sustained test
    elif args.mode == 'burst':
        burst_test(args.url, args.model, args.requests, args.workers, args.quality)
    else:
        sustained_test(args.url, args.model, args.duration, args.workers, args.quality)

