#!/usr/bin/env python3
"""
Protocol comparison tool for HTTP vs gRPC with Triton Inference Server.
This tool helps you understand the performance differences between HTTP and gRPC.
"""
import argparse
import time
import statistics
import numpy as np
import cv2
from typing import List, Dict, Any, Tuple
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

from config import ClientConfig
from utils.models import PerformanceStats
from utils import preprocess, postprocess


class ProtocolComparison:
    """Compare HTTP vs gRPC performance for Triton inference."""
    
    def __init__(self, config: ClientConfig):
        self.config = config
        self.grpc_client = None
        self.http_client = None
        self.class_names = []
        self._load_class_names()
        
    def _load_class_names(self):
        """Load class names from data.yaml."""
        import yaml
        try:
            with open(self.config.paths.data_yaml) as f:
                data = yaml.safe_load(f)
            self.class_names = list(data["names"].values())
        except Exception as e:
            print(f"Warning: Could not load class names: {e}")
            self.class_names = [f"class_{i}" for i in range(100)]
    
    def connect_grpc(self) -> bool:
        """Connect to Triton server via gRPC."""
        try:
            self.grpc_client = grpcclient.InferenceServerClient(
                url=self.config.server.url,
                verbose=self.config.server.verbose
            )
            
            if not self.grpc_client.is_server_live():
                print("gRPC: Server is not live")
                return False
            if not self.grpc_client.is_server_ready():
                print("gRPC: Server is not ready")
                return False
            if not self.grpc_client.is_model_ready(self.config.model.name):
                print(f"gRPC: Model {self.config.model.name} is not ready")
                return False
                
            print("✅ gRPC connection successful")
            return True
            
        except Exception as e:
            print(f"❌ gRPC connection failed: {e}")
            return False
    
    def connect_http(self) -> bool:
        """Connect to Triton server via HTTP."""
        try:
            # Convert gRPC URL to HTTP URL
            http_url = self.config.server.url.replace(':8001', ':8000')
            self.http_client = httpclient.InferenceServerClient(
                url=http_url,
                verbose=self.config.server.verbose
            )
            
            if not self.http_client.is_server_live():
                print("HTTP: Server is not live")
                return False
            if not self.http_client.is_server_ready():
                print("HTTP: Server is not ready")
                return False
            if not self.http_client.is_model_ready(self.config.model.name):
                print(f"HTTP: Model {self.config.model.name} is not ready")
                return False
                
            print("✅ HTTP connection successful")
            return True
            
        except Exception as e:
            print(f"❌ HTTP connection failed: {e}")
            return False
    
    def run_grpc_inference(self, input_data: np.ndarray) -> Tuple[float, np.ndarray]:
        """Run inference via gRPC and return latency + result."""
        start_time = time.time()
        
        # Prepare inputs/outputs
        inputs = []
        outputs = []
        inputs.append(grpcclient.InferInput(
            "INPUT__0", 
            [1, 3, self.config.model.input_height, self.config.model.input_width], 
            "FP32"
        ))
        inputs[0].set_data_from_numpy(input_data)
        outputs.append(grpcclient.InferRequestedOutput("OUTPUT__0"))
        
        # Run inference
        results = self.grpc_client.infer(
            model_name=self.config.model.name,
            inputs=inputs,
            outputs=outputs,
            client_timeout=self.config.server.timeout
        )
        
        latency = time.time() - start_time
        output_tensor = results.as_numpy("OUTPUT__0")
        return latency, output_tensor
    
    def run_http_inference(self, input_data: np.ndarray) -> Tuple[float, np.ndarray]:
        """Run inference via HTTP and return latency + result."""
        start_time = time.time()
        
        # Prepare inputs/outputs
        inputs = []
        outputs = []
        inputs.append(httpclient.InferInput(
            "INPUT__0", 
            [1, 3, self.config.model.input_height, self.config.model.input_width], 
            "FP32"
        ))
        inputs[0].set_data_from_numpy(input_data)
        outputs.append(httpclient.InferRequestedOutput("OUTPUT__0"))
        
        # Run inference
        results = self.http_client.infer(
            model_name=self.config.model.name,
            inputs=inputs,
            outputs=outputs
        )
        
        latency = time.time() - start_time
        output_tensor = results.as_numpy("OUTPUT__0")
        return latency, output_tensor
    
    def benchmark_protocol(self, protocol: str, num_iterations: int = 100) -> Dict[str, Any]:
        """Benchmark a specific protocol."""
        print(f"\n🔄 Benchmarking {protocol}...")
        
        # Create test data
        test_data = np.random.randn(1, 3, self.config.model.input_height, self.config.model.input_width).astype(np.float32)
        
        latencies = []
        successful_requests = 0
        
        for i in range(num_iterations):
            try:
                if protocol == "gRPC":
                    latency, _ = self.run_grpc_inference(test_data)
                else:  # HTTP
                    latency, _ = self.run_http_inference(test_data)
                
                latencies.append(latency)
                successful_requests += 1
                
                if i % 10 == 0:
                    print(f"  {protocol}: {i+1}/{num_iterations} requests completed")
                    
            except Exception as e:
                print(f"  {protocol}: Request {i+1} failed: {e}")
        
        if not latencies:
            return {"error": f"No successful requests for {protocol}"}
        
        # Calculate statistics
        avg_latency = statistics.mean(latencies)
        p50_latency = np.percentile(latencies, 50)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        min_latency = min(latencies)
        max_latency = max(latencies)
        throughput = successful_requests / sum(latencies)
        
        return {
            "protocol": protocol,
            "successful_requests": successful_requests,
            "total_requests": num_iterations,
            "success_rate": successful_requests / num_iterations,
            "avg_latency": avg_latency,
            "p50_latency": p50_latency,
            "p95_latency": p95_latency,
            "p99_latency": p99_latency,
            "min_latency": min_latency,
            "max_latency": max_latency,
            "throughput": throughput,
            "latencies": latencies
        }
    
    def compare_protocols(self, num_iterations: int = 100) -> Dict[str, Any]:
        """Compare HTTP vs gRPC performance."""
        print("🚀 Starting Protocol Comparison")
        print("=" * 60)
        
        results = {}
        
        # Test gRPC
        if self.connect_grpc():
            grpc_results = self.benchmark_protocol("gRPC", num_iterations)
            results["grpc"] = grpc_results
        else:
            print("❌ Skipping gRPC tests due to connection failure")
        
        # Test HTTP
        if self.connect_http():
            http_results = self.benchmark_protocol("HTTP", num_iterations)
            results["http"] = http_results
        else:
            print("❌ Skipping HTTP tests due to connection failure")
        
        return results
    
    def print_comparison_report(self, results: Dict[str, Any]):
        """Print a detailed comparison report."""
        print("\n" + "=" * 80)
        print("📊 PROTOCOL COMPARISON REPORT")
        print("=" * 80)
        
        if "grpc" in results and "error" not in results["grpc"]:
            grpc = results["grpc"]
            print(f"\n🔵 gRPC Results:")
            print(f"  Success Rate: {grpc['success_rate']:.2%}")
            print(f"  Throughput: {grpc['throughput']:.2f} req/s")
            print(f"  Average Latency: {grpc['avg_latency']*1000:.2f}ms")
            print(f"  P50 Latency: {grpc['p50_latency']*1000:.2f}ms")
            print(f"  P95 Latency: {grpc['p95_latency']*1000:.2f}ms")
            print(f"  P99 Latency: {grpc['p99_latency']*1000:.2f}ms")
            print(f"  Min Latency: {grpc['min_latency']*1000:.2f}ms")
            print(f"  Max Latency: {grpc['max_latency']*1000:.2f}ms")
        
        if "http" in results and "error" not in results["http"]:
            http = results["http"]
            print(f"\n🟢 HTTP Results:")
            print(f"  Success Rate: {http['success_rate']:.2%}")
            print(f"  Throughput: {http['throughput']:.2f} req/s")
            print(f"  Average Latency: {http['avg_latency']*1000:.2f}ms")
            print(f"  P50 Latency: {http['p50_latency']*1000:.2f}ms")
            print(f"  P95 Latency: {http['p95_latency']*1000:.2f}ms")
            print(f"  P99 Latency: {http['p99_latency']*1000:.2f}ms")
            print(f"  Min Latency: {http['min_latency']*1000:.2f}ms")
            print(f"  Max Latency: {http['max_latency']*1000:.2f}ms")
        
        # Direct comparison
        if ("grpc" in results and "error" not in results["grpc"] and 
            "http" in results and "error" not in results["http"]):
            
            grpc = results["grpc"]
            http = results["http"]
            
            print(f"\n📈 Direct Comparison:")
            print(f"  Latency (gRPC vs HTTP): {grpc['avg_latency']*1000:.2f}ms vs {http['avg_latency']*1000:.2f}ms")
            print(f"  Throughput (gRPC vs HTTP): {grpc['throughput']:.2f} vs {http['throughput']:.2f} req/s")
            
            latency_improvement = ((http['avg_latency'] - grpc['avg_latency']) / http['avg_latency']) * 100
            throughput_improvement = ((grpc['throughput'] - http['throughput']) / http['throughput']) * 100
            
            print(f"\n🎯 Performance Analysis:")
            if latency_improvement > 0:
                print(f"  gRPC is {latency_improvement:.1f}% faster in latency")
            else:
                print(f"  HTTP is {abs(latency_improvement):.1f}% faster in latency")
            
            if throughput_improvement > 0:
                print(f"  gRPC has {throughput_improvement:.1f}% higher throughput")
            else:
                print(f"  HTTP has {abs(throughput_improvement):.1f}% higher throughput")
        
        print("\n" + "=" * 80)
        print("💡 Recommendations:")
        print("=" * 80)
        
        if ("grpc" in results and "error" not in results["grpc"] and 
            "http" in results and "error" not in results["http"]):
            
            grpc = results["grpc"]
            http = results["http"]
            
            if grpc['avg_latency'] < http['avg_latency'] and grpc['throughput'] > http['throughput']:
                print("🏆 gRPC is the clear winner for this workload!")
                print("   - Lower latency")
                print("   - Higher throughput")
                print("   - Better for high-performance applications")
            elif http['avg_latency'] < grpc['avg_latency'] and http['throughput'] > grpc['throughput']:
                print("🏆 HTTP is the clear winner for this workload!")
                print("   - Lower latency")
                print("   - Higher throughput")
                print("   - Better for this specific use case")
            else:
                print("🤔 Mixed results - choose based on your priorities:")
                if grpc['avg_latency'] < http['avg_latency']:
                    print("   - Choose gRPC if latency is critical")
                if http['throughput'] > grpc['throughput']:
                    print("   - Choose HTTP if throughput is critical")
        
        print("\n📚 Protocol Characteristics:")
        print("   gRPC:")
        print("   ✅ Binary protocol (more efficient)")
        print("   ✅ HTTP/2 multiplexing")
        print("   ✅ Built-in load balancing")
        print("   ✅ Streaming support")
        print("   ❌ More complex setup")
        print("   ❌ Less debugging tools")
        
        print("\n   HTTP:")
        print("   ✅ Simple and widely supported")
        print("   ✅ Easy debugging with standard tools")
        print("   ✅ Works through firewalls/proxies")
        print("   ✅ Human-readable")
        print("   ❌ Text-based (less efficient)")
        print("   ❌ No multiplexing in HTTP/1.1")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser for protocol comparison."""
    parser = argparse.ArgumentParser(description="Compare HTTP vs gRPC performance")
    
    parser.add_argument('-u', '--url', type=str, default='localhost:8001',
                       help='Server URL (gRPC port)')
    parser.add_argument('-m', '--model', type=str, default='yolov11',
                       help='Model name')
    parser.add_argument('-n', '--iterations', type=int, default=100,
                       help='Number of test iterations')
    parser.add_argument('--width', type=int, default=800,
                       help='Input width')
    parser.add_argument('--height', type=int, default=800,
                       help='Input height')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')
    
    return parser


def main():
    """Main entry point for protocol comparison."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Create configuration
    config = ClientConfig()
    config.model.name = args.model
    config.model.input_width = args.width
    config.model.input_height = args.height
    config.server.url = args.url
    config.server.verbose = args.verbose
    
    # Run comparison
    comparison = ProtocolComparison(config)
    results = comparison.compare_protocols(args.iterations)
    comparison.print_comparison_report(results)


if __name__ == '__main__':
    main()
