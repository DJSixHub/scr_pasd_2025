#!/usr/bin/env python3
"""
Script to extract training metrics from Ray model actors
"""

import ray
import json

def get_training_metrics():
    """Connect to Ray and get training metrics from all model actors"""
    try:
        # Connect to the Ray cluster
        ray.init(address="ray://localhost:10001", namespace="distributed-ml")
        print("✅ Connected to Ray cluster")
        
        # Get all model actors
        model_names = [a for a in ray.util.list_named_actors() if not a.startswith("__")]
        print(f"📊 Found {len(model_names)} model actors: {model_names}")
        
        metrics_data = {}
        
        for model_name in model_names:
            try:
                print(f"\n🔍 Getting metrics for {model_name}...")
                actor = ray.get_actor(model_name)
                
                # Get metrics from model actor
                metrics = ray.get(actor.get_metrics.remote())
                model_name_clean = ray.get(actor.get_name.remote())
                
                metrics_data[model_name_clean] = metrics
                
                print(f"  ✅ Retrieved metrics: {list(metrics.keys())}")
                
            except Exception as e:
                print(f"  ❌ Error getting metrics for {model_name}: {e}")
                metrics_data[model_name] = {"error": str(e)}
        
        return metrics_data
        
    except Exception as e:
        print(f"❌ Error connecting to Ray: {e}")
        return {}
    finally:
        if ray.is_initialized():
            ray.shutdown()

def display_metrics(metrics_data):
    """Display metrics in a nice format"""
    if not metrics_data:
        print("❌ No metrics data available")
        return
    
    print("\n" + "="*80)
    print("🎯 TRAINING METRICS SUMMARY")
    print("="*80)
    
    for model_name, metrics in metrics_data.items():
        if "error" in metrics:
            print(f"\n❌ {model_name}: {metrics['error']}")
            continue
            
        print(f"\n📊 {model_name}")
        print("-" * 50)
        
        # Parse model name to extract dataset and algorithm
        parts = model_name.split("_")
        if len(parts) >= 2:
            algorithm = parts[0]
            dataset = parts[1]
            print(f"   Algorithm: {algorithm}")
            print(f"   Dataset: {dataset}")
        
        # Display all metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if key == "training_time":
                    print(f"   ⏱️  {key}: {value:.2f} seconds")
                elif key in ["accuracy", "precision", "recall", "f1"]:
                    print(f"   🎯 {key}: {value:.3f}")
                elif "mse" in key.lower() or "rmse" in key.lower():
                    print(f"   📐 {key}: {value:.4f}")
                else:
                    print(f"   📈 {key}: {value}")
            else:
                print(f"   ℹ️  {key}: {value}")
    
    # Save to JSON file
    output_file = "training_metrics.json"
    with open(output_file, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    print(f"\n💾 Detailed metrics saved to: {output_file}")
    print(f"📁 Training plots available in: ./training_plots/")

if __name__ == "__main__":
    print("🚀 Extracting training metrics from distributed ML platform...")
    metrics = get_training_metrics()
    display_metrics(metrics)
