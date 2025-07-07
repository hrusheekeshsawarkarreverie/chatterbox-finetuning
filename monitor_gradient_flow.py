#!/usr/bin/env python3
"""
Monitor gradient flow during training by injecting hooks
"""

import torch
import torch.nn as nn
import time
import threading
from pathlib import Path
import json
import logging

class GradientFlowMonitor:
    """Monitor gradient flow for Hindi embeddings during training"""
    
    def __init__(self, output_dir="./gradient_logs", freeze_vocab_size=704):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.freeze_vocab_size = freeze_vocab_size
        self.gradient_history = []
        self.monitoring = False
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'gradient_monitor.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def register_hooks(self, model):
        """Register gradient monitoring hooks"""
        self.logger.info("Registering gradient monitoring hooks...")
        
        def text_emb_hook(module, grad_input, grad_output):
            if not self.monitoring:
                return
                
            if hasattr(module, 'weight') and module.weight.grad is not None:
                grad = module.weight.grad
                english_grad = grad[:self.freeze_vocab_size]
                hindi_grad = grad[self.freeze_vocab_size:]
                
                # Calculate statistics
                stats = {
                    'timestamp': time.time(),
                    'layer': 'text_emb',
                    'english_grad_mean': float(english_grad.mean()),
                    'english_grad_std': float(english_grad.std()),
                    'english_grad_max': float(english_grad.abs().max()),
                    'hindi_grad_mean': float(hindi_grad.mean()),
                    'hindi_grad_std': float(hindi_grad.std()),
                    'hindi_grad_max': float(hindi_grad.abs().max()),
                    'hindi_grad_norm': float(torch.norm(hindi_grad)),
                    'english_grad_norm': float(torch.norm(english_grad))
                }
                
                self.gradient_history.append(stats)
                
                # Log critical issues
                if stats['hindi_grad_max'] < 1e-8:
                    self.logger.warning("🚨 Hindi embedding gradients are essentially zero!")
                elif stats['hindi_grad_max'] < 1e-6:
                    self.logger.warning("⚠️  Hindi embedding gradients are very small!")
                
                # Log every 10th gradient update
                if len(self.gradient_history) % 10 == 0:
                    self.logger.info(f"Text Emb - Hindi grad: {stats['hindi_grad_max']:.2e}, English grad: {stats['english_grad_max']:.2e}")
        
        def text_head_hook(module, grad_input, grad_output):
            if not self.monitoring:
                return
                
            if hasattr(module, 'weight') and module.weight.grad is not None:
                grad = module.weight.grad
                english_grad = grad[:self.freeze_vocab_size]
                hindi_grad = grad[self.freeze_vocab_size:]
                
                # Calculate statistics
                stats = {
                    'timestamp': time.time(),
                    'layer': 'text_head',
                    'english_grad_mean': float(english_grad.mean()),
                    'english_grad_std': float(english_grad.std()),
                    'english_grad_max': float(english_grad.abs().max()),
                    'hindi_grad_mean': float(hindi_grad.mean()),
                    'hindi_grad_std': float(hindi_grad.std()),
                    'hindi_grad_max': float(hindi_grad.abs().max()),
                    'hindi_grad_norm': float(torch.norm(hindi_grad)),
                    'english_grad_norm': float(torch.norm(english_grad))
                }
                
                self.gradient_history.append(stats)
                
                # Log critical issues
                if stats['hindi_grad_max'] < 1e-8:
                    self.logger.warning("🚨 Hindi head gradients are essentially zero!")
                elif stats['hindi_grad_max'] < 1e-6:
                    self.logger.warning("⚠️  Hindi head gradients are very small!")
        
        # Register hooks
        if hasattr(model, 't3'):
            model.t3.text_emb.register_backward_hook(text_emb_hook)
            model.t3.text_head.register_backward_hook(text_head_hook)
        elif hasattr(model, 'model') and hasattr(model.model, 't3'):
            model.model.t3.text_emb.register_backward_hook(text_emb_hook)
            model.model.t3.text_head.register_backward_hook(text_head_hook)
        else:
            self.logger.error("Could not find t3 model to register hooks!")
            
    def start_monitoring(self):
        """Start gradient monitoring"""
        self.monitoring = True
        self.logger.info("✅ Gradient monitoring started")
        
    def stop_monitoring(self):
        """Stop gradient monitoring"""
        self.monitoring = False
        self.logger.info("⏹️  Gradient monitoring stopped")
        
    def save_gradient_history(self):
        """Save gradient history to file"""
        if not self.gradient_history:
            return
            
        output_file = self.output_dir / f"gradient_history_{int(time.time())}.json"
        with open(output_file, 'w') as f:
            json.dump(self.gradient_history, f, indent=2)
        
        self.logger.info(f"💾 Gradient history saved to {output_file}")
        
    def analyze_gradient_flow(self):
        """Analyze gradient flow patterns"""
        if not self.gradient_history:
            self.logger.warning("No gradient history to analyze")
            return
            
        self.logger.info("🔍 Analyzing gradient flow patterns...")
        
        # Separate by layer
        emb_gradients = [g for g in self.gradient_history if g['layer'] == 'text_emb']
        head_gradients = [g for g in self.gradient_history if g['layer'] == 'text_head']
        
        if emb_gradients:
            avg_hindi_grad = sum(g['hindi_grad_max'] for g in emb_gradients) / len(emb_gradients)
            avg_english_grad = sum(g['english_grad_max'] for g in emb_gradients) / len(emb_gradients)
            
            self.logger.info(f"Text Embedding Analysis:")
            self.logger.info(f"  Average Hindi grad magnitude: {avg_hindi_grad:.2e}")
            self.logger.info(f"  Average English grad magnitude: {avg_english_grad:.2e}")
            self.logger.info(f"  Hindi/English ratio: {avg_hindi_grad/avg_english_grad:.3f}")
            
            if avg_hindi_grad < 1e-7:
                self.logger.warning("🚨 Hindi embedding gradients are extremely small!")
            elif avg_hindi_grad < 1e-5:
                self.logger.warning("⚠️  Hindi embedding gradients are small - consider higher learning rate")
            else:
                self.logger.info("✅ Hindi embedding gradients appear healthy")
                
        if head_gradients:
            avg_hindi_grad = sum(g['hindi_grad_max'] for g in head_gradients) / len(head_gradients)
            avg_english_grad = sum(g['english_grad_max'] for g in head_gradients) / len(head_gradients)
            
            self.logger.info(f"Text Head Analysis:")
            self.logger.info(f"  Average Hindi grad magnitude: {avg_hindi_grad:.2e}")
            self.logger.info(f"  Average English grad magnitude: {avg_english_grad:.2e}")
            self.logger.info(f"  Hindi/English ratio: {avg_hindi_grad/avg_english_grad:.3f}")

def inject_into_trainer(trainer, monitor_dir="./gradient_logs"):
    """Inject gradient monitoring into an existing trainer"""
    
    monitor = GradientFlowMonitor(monitor_dir)
    monitor.register_hooks(trainer.model)
    
    # Hook into trainer callbacks
    original_on_step_end = trainer.callback_handler.on_step_end
    original_on_train_begin = trainer.callback_handler.on_train_begin
    original_on_train_end = trainer.callback_handler.on_train_end
    
    def new_on_train_begin(args, state, control, **kwargs):
        monitor.start_monitoring()
        return original_on_train_begin(args, state, control, **kwargs)
    
    def new_on_step_end(args, state, control, **kwargs):
        # Save gradient history periodically
        if state.global_step % 100 == 0:
            monitor.save_gradient_history()
        return original_on_step_end(args, state, control, **kwargs)
    
    def new_on_train_end(args, state, control, **kwargs):
        monitor.stop_monitoring()
        monitor.analyze_gradient_flow()
        monitor.save_gradient_history()
        return original_on_train_end(args, state, control, **kwargs)
    
    trainer.callback_handler.on_train_begin = new_on_train_begin
    trainer.callback_handler.on_step_end = new_on_step_end
    trainer.callback_handler.on_train_end = new_on_train_end
    
    return monitor

# Example usage for manual monitoring
def main():
    """Example of manual gradient monitoring"""
    print("Gradient Flow Monitor")
    print("=" * 40)
    
    # This would be used in your training script
    # monitor = GradientFlowMonitor()
    # monitor.register_hooks(model)
    # monitor.start_monitoring()
    
    # ... training loop ...
    
    # monitor.stop_monitoring()
    # monitor.analyze_gradient_flow()
    # monitor.save_gradient_history()
    
    print("To use this monitor, import and call inject_into_trainer(trainer)")
    print("Example:")
    print("  from monitor_gradient_flow import inject_into_trainer")
    print("  monitor = inject_into_trainer(trainer)")

if __name__ == "__main__":
    main() 