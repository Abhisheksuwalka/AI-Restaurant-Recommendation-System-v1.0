"""
Test Runner and Configuration

Provides comprehensive test execution with different test suites,
reporting, and configuration options for the AI recommendation system.
"""

import unittest
import sys
import os
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import coverage

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class TestSuiteRunner:
    """Advanced test suite runner with reporting and configuration"""
    
    def __init__(self):
        self.test_suites = {
            'unit': ['test_data_preprocessor', 'test_collaborative_filtering', 
                    'test_content_based_filtering', 'test_sentiment_analyzer'],
            'integration': ['test_integration', 'test_hybrid_recommender'],
            'performance': ['test_performance'],
            'emotional': ['test_emotional_intelligence'],
            'llm': ['test_llm_recommender'],
            'all': []  # Will be populated with all tests
        }
        
        # Populate 'all' with all test modules
        all_tests = set()
        for suite_tests in self.test_suites.values():
            all_tests.update(suite_tests)
        self.test_suites['all'] = list(all_tests)
        
        self.coverage_enabled = False
        self.verbose = False
        self.failfast = False
        
    def setup_coverage(self, source_dirs: List[str] = None):
        """Set up coverage tracking"""
        if source_dirs is None:
            source_dirs = ['models', 'data']
        
        self.cov = coverage.Coverage(
            source=source_dirs,
            omit=['*/tests/*', '*/venv/*', '*/__pycache__/*']
        )
        self.cov.start()
        self.coverage_enabled = True
        
    def discover_tests(self, test_modules: List[str]) -> unittest.TestSuite:
        """Discover and load tests from specified modules"""
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        for module_name in test_modules:
            try:
                # Import the test module
                test_module = __import__(f'tests.{module_name}', fromlist=[module_name])
                
                # Load tests from the module
                module_suite = loader.loadTestsFromModule(test_module)
                suite.addTest(module_suite)
                
                print(f"✓ Loaded tests from {module_name}")
                
            except ImportError as e:
                print(f"✗ Failed to import {module_name}: {e}")
            except Exception as e:
                print(f"✗ Error loading tests from {module_name}: {e}")
        
        return suite
    
    def run_test_suite(self, suite_name: str, **kwargs) -> Dict:
        """Run a specific test suite"""
        if suite_name not in self.test_suites:
            raise ValueError(f"Unknown test suite: {suite_name}")
        
        # Update runner configuration
        self.verbose = kwargs.get('verbose', False)
        self.failfast = kwargs.get('failfast', False)
        
        # Set up coverage if requested
        if kwargs.get('coverage', False):
            self.setup_coverage()
        
        test_modules = self.test_suites[suite_name]
        print(f"\\n🧪 Running {suite_name} test suite ({len(test_modules)} modules)")
        print("=" * 60)
        
        # Discover tests
        suite = self.discover_tests(test_modules)
        
        # Configure test runner
        runner = unittest.TextTestRunner(
            verbosity=2 if self.verbose else 1,
            failfast=self.failfast,
            stream=sys.stdout
        )
        
        # Run tests
        start_time = time.time()
        result = runner.run(suite)
        end_time = time.time()
        
        # Generate results
        results = {
            'suite_name': suite_name,
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'skipped': len(result.skipped) if hasattr(result, 'skipped') else 0,
            'success': result.wasSuccessful(),
            'execution_time': end_time - start_time,
            'failure_details': [str(f[1]) for f in result.failures],
            'error_details': [str(e[1]) for e in result.errors]
        }
        
        # Handle coverage
        if self.coverage_enabled:
            self.cov.stop()
            self.cov.save()
            
            # Generate coverage report
            coverage_data = self.generate_coverage_report()
            results['coverage'] = coverage_data
        
        return results
    
    def generate_coverage_report(self) -> Dict:
        """Generate coverage report"""
        if not self.coverage_enabled:
            return {}
        
        # Get coverage data
        total_coverage = self.cov.report(show_missing=False, file=None)
        
        # Get detailed coverage by file
        file_coverage = {}
        for filename in self.cov.get_data().measured_files():
            analysis = self.cov.analysis2(filename)
            covered_lines = len(analysis.executed)
            total_lines = len(analysis.statements)
            
            if total_lines > 0:
                file_coverage[filename] = {
                    'covered_lines': covered_lines,
                    'total_lines': total_lines,
                    'coverage_percent': (covered_lines / total_lines) * 100
                }
        
        return {
            'total_coverage': total_coverage,
            'file_coverage': file_coverage
        }
    
    def run_multiple_suites(self, suite_names: List[str], **kwargs) -> Dict:
        """Run multiple test suites"""
        all_results = {}
        overall_start = time.time()
        
        for suite_name in suite_names:
            print(f"\\n{'='*20} {suite_name.upper()} TESTS {'='*20}")
            
            try:
                results = self.run_test_suite(suite_name, **kwargs)
                all_results[suite_name] = results
                
                # Print summary
                self.print_suite_summary(results)
                
            except Exception as e:
                print(f"❌ Failed to run {suite_name} suite: {e}")
                all_results[suite_name] = {'error': str(e)}
        
        overall_end = time.time()
        
        # Generate overall summary
        overall_results = self.generate_overall_summary(all_results, overall_end - overall_start)
        
        return overall_results
    
    def print_suite_summary(self, results: Dict):
        """Print summary for a test suite"""
        suite_name = results['suite_name']
        
        if results['success']:
            status = "✅ PASSED"
        else:
            status = "❌ FAILED"
        
        print(f"\\n{status} - {suite_name} suite")
        print(f"Tests run: {results['tests_run']}")
        print(f"Failures: {results['failures']}")
        print(f"Errors: {results['errors']}")
        print(f"Skipped: {results['skipped']}")
        print(f"Time: {results['execution_time']:.2f}s")
        
        if 'coverage' in results and results['coverage']:
            print(f"Coverage: {results['coverage']['total_coverage']:.1f}%")
    
    def generate_overall_summary(self, all_results: Dict, total_time: float) -> Dict:
        """Generate overall test summary"""
        total_tests = sum(r.get('tests_run', 0) for r in all_results.values() if 'tests_run' in r)
        total_failures = sum(r.get('failures', 0) for r in all_results.values() if 'failures' in r)
        total_errors = sum(r.get('errors', 0) for r in all_results.values() if 'errors' in r)
        total_skipped = sum(r.get('skipped', 0) for r in all_results.values() if 'skipped' in r)
        
        success_suites = sum(1 for r in all_results.values() if r.get('success', False))
        total_suites = len([r for r in all_results.values() if 'success' in r])
        
        overall_success = total_failures == 0 and total_errors == 0
        
        summary = {
            'overall_success': overall_success,
            'total_suites': total_suites,
            'successful_suites': success_suites,
            'total_tests': total_tests,
            'total_failures': total_failures,
            'total_errors': total_errors,
            'total_skipped': total_skipped,
            'total_time': total_time,
            'suite_results': all_results
        }
        
        # Print overall summary
        print("\\n" + "="*60)
        print("OVERALL TEST SUMMARY")
        print("="*60)
        
        if overall_success:
            print("🎉 ALL TESTS PASSED!")
        else:
            print("💥 SOME TESTS FAILED")
        
        print(f"Suites: {success_suites}/{total_suites} passed")
        print(f"Tests: {total_tests} run, {total_failures} failures, {total_errors} errors")
        print(f"Total time: {total_time:.2f}s")
        
        return summary
    
    def save_results(self, results: Dict, output_file: str = "test_results.json"):
        """Save test results to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\\n📄 Results saved to {output_file}")

def create_test_config():
    """Create test configuration"""
    return {
        'fast_mode': False,  # Skip slow tests
        'coverage_threshold': 80,  # Minimum coverage percentage
        'max_test_time': 300,  # Maximum time per test suite (seconds)
        'parallel_execution': False,  # Run tests in parallel
        'report_format': 'json',  # Output format: json, xml, html
        'output_directory': 'test_reports'
    }

def main():
    """Main test runner entry point"""
    parser = argparse.ArgumentParser(description='AI Recommendation System Test Runner')
    
    parser.add_argument('suites', nargs='*', default=['all'],
                       help='Test suites to run (unit, integration, performance, emotional, llm, all)')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    parser.add_argument('--coverage', '-c', action='store_true',
                       help='Enable coverage reporting')
    
    parser.add_argument('--failfast', '-f', action='store_true',
                       help='Stop on first failure')
    
    parser.add_argument('--output', '-o', default='test_results.json',
                       help='Output file for results')
    
    parser.add_argument('--fast', action='store_true',
                       help='Skip slow tests')
    
    args = parser.parse_args()
    
    # Create test runner
    runner = TestSuiteRunner()
    
    # Configure test environment
    if args.fast:
        os.environ['SKIP_SLOW_TESTS'] = '1'
    
    # Run tests
    try:
        if len(args.suites) == 1:
            results = runner.run_test_suite(
                args.suites[0],
                verbose=args.verbose,
                coverage=args.coverage,
                failfast=args.failfast
            )
        else:
            results = runner.run_multiple_suites(
                args.suites,
                verbose=args.verbose,
                coverage=args.coverage,
                failfast=args.failfast
            )
        
        # Save results
        if args.output:
            runner.save_results(results, args.output)
        
        # Exit with appropriate code
        if isinstance(results, dict) and 'overall_success' in results:
            exit_code = 0 if results['overall_success'] else 1
        elif isinstance(results, dict) and 'success' in results:
            exit_code = 0 if results['success'] else 1
        else:
            exit_code = 1
        
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\\n⏹ Tests interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\\n💥 Test runner failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
