#!/usr/bin/env python3
"""
Memory Optimization Helper for SAR Knowledge Distillation Training

This script analyzes your system and suggests optimal memory settings for training.
"""

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Some features will be limited.")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. System memory analysis will be limited.")


class MemoryAnalyzer:
    """Analyzes system memory and provides optimization recommendations"""

    def __init__(self):
        self.gpu_info = self._get_gpu_info() if TORCH_AVAILABLE else {}
        self.cpu_info = self._get_cpu_info()
        self.system_ram = self._get_system_ram()

    def _get_gpu_info(self) -> Dict:
        """Get GPU memory information"""
        if not torch.cuda.is_available():
            return {"available": False, "count": 0}

        gpu_count = torch.cuda.device_count()
        gpus = []

        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            gpus.append({
                "id": i,
                "name": props.name,
                "memory_gb": memory_gb,
                "memory_bytes": props.total_memory,
                "compute_capability": f"{props.major}.{props.minor}"
            })

        return {
            "available": True,
            "count": gpu_count,
            "gpus": gpus,
            "primary_gpu": gpus[0] if gpus else None
        }

    def _get_cpu_info(self) -> Dict:
        """Get CPU information"""
        info = {"cores": os.cpu_count()}

        if PSUTIL_AVAILABLE:
            info.update({
                "physical_cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count(logical=True),
            })

        return info

    def _get_system_ram(self) -> Dict:
        """Get system RAM information"""
        if PSUTIL_AVAILABLE:
            mem = psutil.virtual_memory()
            return {
                "total_gb": mem.total / (1024**3),
                "available_gb": mem.available / (1024**3),
                "percent_used": mem.percent
            }
        else:
            # Fallback method
            try:
                # Linux/Unix
                with open('/proc/meminfo', 'r') as f:
                    lines = f.readlines()
                    mem_total = None
                    mem_available = None

                    for line in lines:
                        if line.startswith('MemTotal:'):
                            mem_total = int(line.split()[1]) * 1024  # Convert KB to bytes
                        elif line.startswith('MemAvailable:'):
                            mem_available = int(line.split()[1]) * 1024

                    if mem_total and mem_available:
                        return {
                            "total_gb": mem_total / (1024**3),
                            "available_gb": mem_available / (1024**3),
                            "percent_used": (1 - mem_available/mem_total) * 100
                        }
            except:
                pass

            return {"total_gb": "unknown", "available_gb": "unknown", "percent_used": "unknown"}

    def estimate_model_memory(self, model_name: str, dtype: str = "float16") -> Dict:
        """Estimate memory requirements for a model"""
        # Rough estimates based on known model sizes
        model_sizes = {
            # Teacher models (MoE models are larger due to expert parameters)
            "huihui-ai/Huihui-MoE-1B-A0.6B": 2.2,  # GB in float32, includes MoE overhead

            # Student models
            "HuggingFaceTB/SmolLM-135M": 0.54,  # 135M parameters
            "HuggingFaceTB/SmolLM-360M": 1.44,  # 360M parameters
            "microsoft/DialoGPT-small": 0.5,   # ~117M parameters

            # Common base models
            "gpt2": 0.5,                        # 124M parameters
            "gpt2-medium": 1.4,                 # 355M parameters
            "gpt2-large": 3.0,                  # 774M parameters
        }

        # Get base size in GB (assumes float32)
        base_size_gb = model_sizes.get(model_name, 1.0)  # Default fallback

        # Adjust for dtype
        dtype_multipliers = {
            "float32": 1.0,
            "float16": 0.5,
            "bfloat16": 0.5,
            "int8": 0.25,
            "int4": 0.125
        }

        actual_size_gb = base_size_gb * dtype_multipliers.get(dtype, 1.0)

        return {
            "base_size_gb": base_size_gb,
            "actual_size_gb": actual_size_gb,
            "dtype": dtype,
            "estimated": model_name not in model_sizes
        }

    def analyze_training_memory(self, teacher_model: str, student_model: str,
                               batch_size: int, seq_length: int, dtype: str = "float16") -> Dict:
        """Analyze memory requirements for training"""

        teacher_mem = self.estimate_model_memory(teacher_model, dtype)
        student_mem = self.estimate_model_memory(student_model, dtype)

        # Estimate activation memory (rough approximation)
        # Activations scale with batch_size * seq_length * hidden_dim * num_layers
        # This is a rough estimate
        activation_multiplier = batch_size * seq_length / (1024 * 512)  # Normalize to batch=1, seq=512

        teacher_activation_gb = teacher_mem["actual_size_gb"] * activation_multiplier * 2  # Forward + backward
        student_activation_gb = student_mem["actual_size_gb"] * activation_multiplier * 2

        # Gradient memory (roughly same as model parameters)
        gradient_gb = teacher_mem["actual_size_gb"] + student_mem["actual_size_gb"]

        # Optimizer state (AdamW uses ~2x parameter memory)
        optimizer_gb = gradient_gb * 2

        # Total memory estimate
        total_gb = (teacher_mem["actual_size_gb"] + student_mem["actual_size_gb"] +
                   teacher_activation_gb + student_activation_gb + gradient_gb + optimizer_gb)

        # Add safety margin
        total_with_margin_gb = total_gb * 1.3

        return {
            "teacher_model_gb": teacher_mem["actual_size_gb"],
            "student_model_gb": student_mem["actual_size_gb"],
            "teacher_activations_gb": teacher_activation_gb,
            "student_activations_gb": student_activation_gb,
            "gradients_gb": gradient_gb,
            "optimizer_gb": optimizer_gb,
            "total_estimated_gb": total_gb,
            "total_with_margin_gb": total_with_margin_gb,
            "breakdown": {
                "models": teacher_mem["actual_size_gb"] + student_mem["actual_size_gb"],
                "activations": teacher_activation_gb + student_activation_gb,
                "gradients": gradient_gb,
                "optimizer": optimizer_gb
            }
        }


def suggest_optimal_settings(analyzer: MemoryAnalyzer, teacher_model: str, student_model: str) -> Dict:
    """Suggest optimal training settings based on available memory"""

    if not analyzer.gpu_info["available"]:
        return {
            "error": "No GPU available. This training requires CUDA.",
            "suggestions": []
        }

    primary_gpu = analyzer.gpu_info["primary_gpu"]
    gpu_memory_gb = primary_gpu["memory_gb"]

    suggestions = []
    configs = []

    # Test different configurations
    test_configs = [
        {"batch_size": 1, "seq_length": 512, "dtype": "float16", "grad_accum": 16},
        {"batch_size": 1, "seq_length": 512, "dtype": "float16", "grad_accum": 32},
        {"batch_size": 1, "seq_length": 256, "dtype": "float16", "grad_accum": 32},
        {"batch_size": 2, "seq_length": 256, "dtype": "float16", "grad_accum": 16},
        {"batch_size": 1, "seq_length": 1024, "dtype": "float16", "grad_accum": 8},
    ]

    for config in test_configs:
        memory_analysis = analyzer.analyze_training_memory(
            teacher_model, student_model,
            config["batch_size"], config["seq_length"], config["dtype"]
        )

        required_gb = memory_analysis["total_with_margin_gb"]
        will_fit = required_gb <= gpu_memory_gb * 0.9  # Use 90% as safety threshold

        config_result = {
            **config,
            "required_memory_gb": required_gb,
            "will_fit": will_fit,
            "memory_efficiency": min(100, (required_gb / gpu_memory_gb) * 100),
            "memory_analysis": memory_analysis
        }

        configs.append(config_result)

    # Sort by feasibility and efficiency
    feasible_configs = [c for c in configs if c["will_fit"]]
    if feasible_configs:
        # Sort by memory efficiency (lower is better, but still feasible)
        recommended = sorted(feasible_configs, key=lambda x: x["memory_efficiency"])[0]
    else:
        # If nothing fits, suggest the most memory-efficient option with warnings
        recommended = sorted(configs, key=lambda x: x["memory_efficiency"])[0]

    # Generate optimization suggestions
    optimization_suggestions = []

    if not feasible_configs:
        optimization_suggestions.append("⚠️ No configuration fits in available GPU memory. Consider these options:")
        optimization_suggestions.append("   • Use teacher CPU offloading (--offload_teacher_to_cpu)")
        optimization_suggestions.append("   • Reduce sequence length further (--block_size 256 or 128)")
        optimization_suggestions.append("   • Increase gradient accumulation steps")
        optimization_suggestions.append("   • Use aggressive memory clearing (--clear_cache_every_step)")
    else:
        optimization_suggestions.append("✅ Found feasible configurations!")

    # Always suggest memory optimizations
    optimization_suggestions.extend([
        "💡 Memory optimization tips:",
        "   • Use float16 precision (--model_dtype float16)",
        "   • Enable teacher CPU offloading (--offload_teacher_to_cpu)",
        "   • Enable cache clearing (--clear_cache_every_step)",
        "   • Use gradient checkpointing (enabled by default)",
        "   • Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    ])

    return {
        "gpu_memory_gb": gpu_memory_gb,
        "recommended_config": recommended,
        "all_configs": configs,
        "feasible_configs": feasible_configs,
        "optimization_suggestions": optimization_suggestions
    }


def generate_training_command(config: Dict, teacher_model: str, student_model: str,
                            output_dir: str = "outputs/sar_kd_optimized") -> str:
    """Generate the optimal training command"""

    cmd_parts = [
        "python train_sar_kd_memory_optimized.py",
        f"--teacher_model {teacher_model}",
        f"--student_model {student_model}",
        f"--model_dtype {config['dtype']}",
        f"--per_device_batch_size {config['batch_size']}",
        f"--grad_accum_steps {config['grad_accum']}",
        f"--block_size {config['seq_length']}",
        f"--output_dir {output_dir}",
        "--offload_teacher_to_cpu",  # Always enable for memory efficiency
        "--clear_cache_every_step",
        "--aggressive_gc"
    ]

    return " \\\n  ".join(cmd_parts)


def main():
    parser = argparse.ArgumentParser(description="Memory optimization helper for SAR KD training")
    parser.add_argument("--teacher_model", default="huihui-ai/Huihui-MoE-1B-A0.6B",
                       help="Teacher model name")
    parser.add_argument("--student_model", default="HuggingFaceTB/SmolLM-135M",
                       help="Student model name")
    parser.add_argument("--output_format", choices=["human", "json"], default="human",
                       help="Output format")
    parser.add_argument("--generate_command", action="store_true",
                       help="Generate optimized training command")

    args = parser.parse_args()

    # Analyze system
    analyzer = MemoryAnalyzer()

    # Get suggestions
    suggestions = suggest_optimal_settings(analyzer, args.teacher_model, args.student_model)

    if args.output_format == "json":
        # JSON output for programmatic use
        output = {
            "system_info": {
                "gpu_info": analyzer.gpu_info,
                "cpu_info": analyzer.cpu_info,
                "system_ram": analyzer.system_ram
            },
            "suggestions": suggestions
        }
        print(json.dumps(output, indent=2))
    else:
        # Human-readable output
        print("🔍 SAR Knowledge Distillation Memory Optimizer")
        print("=" * 60)

        # System info
        print("\n📊 System Information:")
        if analyzer.gpu_info["available"]:
            gpu = analyzer.gpu_info["primary_gpu"]
            print(f"  GPU: {gpu['name']}")
            print(f"  GPU Memory: {gpu['memory_gb']:.1f} GB")
            print(f"  Compute Capability: {gpu['compute_capability']}")
        else:
            print("  GPU: None (CUDA not available)")

        print(f"  CPU Cores: {analyzer.cpu_info['cores']}")
        if isinstance(analyzer.system_ram["total_gb"], (int, float)):
            print(f"  System RAM: {analyzer.system_ram['total_gb']:.1f} GB")

        # Memory analysis
        print(f"\n🧠 Memory Analysis for:")
        print(f"  Teacher: {args.teacher_model}")
        print(f"  Student: {args.student_model}")

        if "error" in suggestions:
            print(f"\n❌ Error: {suggestions['error']}")
            return

        recommended = suggestions["recommended_config"]
        print(f"\n✨ Recommended Configuration:")
        print(f"  Batch Size: {recommended['batch_size']}")
        print(f"  Sequence Length: {recommended['seq_length']}")
        print(f"  Data Type: {recommended['dtype']}")
        print(f"  Gradient Accumulation: {recommended['grad_accum']}")
        print(f"  Estimated Memory: {recommended['required_memory_gb']:.1f} GB")
        print(f"  Will Fit: {'✅ Yes' if recommended['will_fit'] else '❌ No'}")

        # Show all configurations
        print(f"\n📋 All Tested Configurations:")
        print(f"{'Batch':>5} {'SeqLen':>6} {'AccumGrad':>9} {'Memory(GB)':>10} {'Fits?':>6}")
        print("-" * 45)
        for config in suggestions["all_configs"]:
            fits_icon = "✅" if config["will_fit"] else "❌"
            print(f"{config['batch_size']:>5} {config['seq_length']:>6} {config['grad_accum']:>9} "
                  f"{config['required_memory_gb']:>10.1f} {fits_icon:>6}")

        # Optimization suggestions
        print(f"\n💡 Optimization Suggestions:")
        for suggestion in suggestions["optimization_suggestions"]:
            print(f"  {suggestion}")

        # Generate command if requested
        if args.generate_command:
            print(f"\n🚀 Optimized Training Command:")
            print("-" * 40)
            command = generate_training_command(recommended, args.teacher_model, args.student_model)
            print(command)
            print()
            print("💾 To save this command to a file:")
            print("python memory_optimizer_helper.py --generate_command > run_training.sh")
            print("chmod +x run_training.sh")


if __name__ == "__main__":
    main()
