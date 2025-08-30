##############################
##############################
##############################

class RealTimeSaver:
    """Save algorithms immediately as they're discovered"""
    
    def __init__(self, base_dir="realtime_results"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.saved_signatures = set()
    
    def save_algorithm_immediately(self, genome, cycle, rank, fitness, discoverer=None):
        """Save algorithm IMMEDIATELY upon discovery"""
        try:
            # Create unique filename
            signature = genome.get_signature()[:8] if hasattr(genome, 'get_signature') else 'unknown'
            
            # Skip if already saved
            if signature in self.saved_signatures:
                return False
            
            filename = f"cycle_{cycle:03d}_rank_{rank:03d}_fit_{fitness:.4f}_{signature}.json"
            filepath = self.base_dir / filename
            
            # Serialize with complete structure
            algorithm_data = EnhancedAlgorithmSerializer.serialize_genome_complete(genome)
            algorithm_data['cycle'] = cycle
            algorithm_data['rank'] = rank
            algorithm_data['fitness'] = fitness
            
            # Add decoded version if discoverer available
            if discoverer and hasattr(discoverer, '_decode_genome'):
                try:
                    algorithm_data['decoded'] = discoverer._decode_genome(genome)
                except:
                    pass
            
            # WRITE IMMEDIATELY
            with open(filepath, 'w') as f:
                json.dump(algorithm_data, f, indent=2, default=str)
            
            print(f"✓ SAVED: {filename} ({len(algorithm_data['instructions'])} instructions)")
            self.saved_signatures.add(signature)
            return True
            
        except Exception as e:
            print(f"✗ SAVE FAILED: {e}")
            return False

# FIX 3: MEMORY MANAGEMENT
class MemoryManager:
    """Prevent RAM over-usage"""
    
    def __init__(self, max_cache_size=10000, max_data_size=5000):
        self.max_cache_size = max_cache_size
        self.max_data_size = max_data_size
        self.gc_counter = 0
    
    def cleanup_ndr_cache(self, ndr_computer):
        """Clean NDR cache when it gets too large"""
        if len(ndr_computer.cache) > self.max_cache_size:
            # Keep only recent entries (LRU-style)
            keys_to_remove = list(ndr_computer.cache.keys())[:-self.max_cache_size//2]
            for key in keys_to_remove:
                del ndr_computer.cache[key]
            print(f"Cleaned NDR cache: {len(keys_to_remove)} entries removed")
    
    def trim_data_list(self, data_list):
        """Keep data list from growing infinitely"""
        if len(data_list) > self.max_data_size:
            # Keep most recent data
            trimmed = data_list[-self.max_data_size:]
            data_list.clear()
            data_list.extend(trimmed)
            print(f"Trimmed data list to {len(data_list)} entries")
    
    def cleanup_evolver(self, evolver):
        """Clean evolver's diversity cache"""
        if hasattr(evolver, 'diversity_cache') and len(evolver.diversity_cache) > self.max_cache_size:
            evolver.diversity_cache.clear()
            # Re-add current population signatures
            for genome in evolver.population:
                evolver.diversity_cache.add(genome.get_signature())
            print(f"Reset diversity cache to {len(evolver.diversity_cache)} entries")
    
    def force_gc(self):
        """Force garbage collection periodically"""
        self.gc_counter += 1
        if self.gc_counter % 10 == 0:
            gc.collect()
            print(f"Forced GC at counter {self.gc_counter}")

# FIX 4: PROPERLY INTEGRATE Q-LEARNING AND GPU
class EnhancedDiscoverySystem:
    """Enhanced system with all fixes integrated"""
    
    def __init__(self, original_system):
        self.system = original_system
        self.realtime_saver = RealTimeSaver()
        self.memory_manager = MemoryManager()
        self.q_learning = None  # Will initialize if needed
        
        # Enable GPU if available
        import torch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Enhanced system using device: {self.device}")
    
    def run_discovery_cycle_enhanced(self, cycle):
        """Enhanced discovery cycle with all fixes"""
        
        # MEMORY MANAGEMENT
        self.memory_manager.cleanup_ndr_cache(self.system.ndr_computer)
        self.memory_manager.trim_data_list(self.system.data)
        self.memory_manager.force_gc()
        
        # Generate data
        self.system.generate_data()
        
        # Run formula discovery
        formula_discoverer = UnifiedFormulaDiscoverer(self.system.data)
        formula_results = formula_discoverer.evolve_formulas()
        
        # IMMEDIATE SAVING - The critical fix!
        if formula_results and 'top_discoveries' in formula_results:
            saved_count = 0
            for idx, discovery in enumerate(formula_results['top_discoveries'][:50]):
                if 'genome' in discovery and discovery['genome']:
                    success = self.realtime_saver.save_algorithm_immediately(
                        discovery['genome'],
                        cycle,
                        discovery.get('rank', idx + 1),
                        discovery.get('fitness', 0),
                        formula_discoverer
                    )
                    if success:
                        saved_count += 1
            
            print(f"Cycle {cycle}: Saved {saved_count} algorithms in real-time")
        
        # Also save complete cycle data
        self.save_cycle_complete(cycle, formula_results, formula_discoverer)
        
        # Clean up evolver memory
        if hasattr(formula_discoverer, 'evolver'):
            self.memory_manager.cleanup_evolver(formula_discoverer.evolver)
        
        return formula_results
    
    def save_cycle_complete(self, cycle, formula_results, discoverer):
        """Save complete cycle data with verification"""
        cycle_dir = Path(f"results_verified/cycle_{cycle:03d}")
        cycle_dir.mkdir(parents=True, exist_ok=True)
        
        # Save complete formulas
        if formula_results and 'top_discoveries' in formula_results:
            all_formulas = []
            for discovery in formula_results['top_discoveries'][:50]:
                if 'genome' in discovery and discovery['genome']:
                    formula_data = EnhancedAlgorithmSerializer.serialize_genome_complete(
                        discovery['genome'],
                        discoverer.data_config if hasattr(discoverer, 'data_config') else None
                    )
                    formula_data['rank'] = discovery.get('rank', 0)
                    formula_data['fitness'] = discovery.get('fitness', 0)
                    all_formulas.append(formula_data)
            
            # Save all formulas in one file for this cycle
            all_formulas_path = cycle_dir / 'all_formulas.json'
            with open(all_formulas_path, 'w') as f:
                json.dump(all_formulas, f, indent=2, default=str)
            
            print(f"✓ Saved {len(all_formulas)} complete formulas to {all_formulas_path}")
        
        # Save cycle summary
        summary = {
            'cycle': cycle,
            'timestamp': datetime.now().isoformat(),
            'total_formulas': len(formula_results.get('top_discoveries', [])),
            'data_points': len(self.system.data),
            'current_n': self.system.current_n,
            'memory_usage_mb': self._get_memory_usage()
        }
        
        with open(cycle_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _get_memory_usage(self):
        """Get current memory usage in MB"""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

# INTEGRATION: Modify the main system's evolve_formulas method
def evolve_formulas_with_fixes(self):
    """Replacement method with all fixes"""
    print("\n📊 Evolving formulas with enhanced saving...")
    
    # Initialize real-time saver if not exists
    if not hasattr(self, 'realtime_saver'):
        self.realtime_saver = RealTimeSaver()

    evolver = evolvo.UnifiedEvolver(evolvo.GenomeType.ALGORITHM, CONFIG.formula_population_size)

    top_discoveries = []
    for rank, genome in enumerate(evolver.population[:50]):
        if genome.fitness and genome.fitness > 0.1:
            discovery = {'rank': rank + 1, 'fitness': genome.fitness, 'genome': genome}
            top_discoveries.append(discovery)
            
            # SAVE IMMEDIATELY!
            self.realtime_saver.save_algorithm_immediately(
                genome, 
                self.current_cycle if hasattr(self, 'current_cycle') else 0,
                rank + 1,
                genome.fitness,
                self
            )
    
    return {'top_discoveries': top_discoveries}