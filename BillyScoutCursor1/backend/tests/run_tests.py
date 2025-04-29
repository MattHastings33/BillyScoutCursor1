import unittest
import sys
import json
from datetime import datetime
from test_box_score_scraper import TestBoxScoreScraper

def run_tests():
    # Create a test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBoxScoreScraper)
    
    # Create a test runner with custom result handler
    class TestResult(unittest.TextTestResult):
        def __init__(self, stream, descriptions, verbosity):
            super().__init__(stream, descriptions, verbosity)
            self.test_results = []
            
        def addSuccess(self, test):
            super().addSuccess(test)
            self.test_results.append({
                'test': test.id(),
                'status': 'success',
                'error': None
            })
            
        def addError(self, test, err):
            super().addError(test, err)
            self.test_results.append({
                'test': test.id(),
                'status': 'error',
                'error': str(err[1])
            })
            
        def addFailure(self, test, err):
            super().addFailure(test, err)
            self.test_results.append({
                'test': test.id(),
                'status': 'failure',
                'error': str(err[1])
            })
            
        def printSummary(self):
            super().printSummary()
            
            # Generate detailed report
            report = {
                'timestamp': datetime.now().isoformat(),
                'total_tests': self.testsRun,
                'failures': len(self.failures),
                'errors': len(self.errors),
                'test_results': self.test_results
            }
            
            # Save report to file
            with open('test_report.json', 'w') as f:
                json.dump(report, f, indent=2)
            
            # Print summary
            print("\nTest Summary:")
            print(f"Total Tests: {self.testsRun}")
            print(f"Failures: {len(self.failures)}")
            print(f"Errors: {len(self.errors)}")
            
            if self.failures:
                print("\nFailures:")
                for failure in self.failures:
                    print(f"- {failure[1]}")
                    
            if self.errors:
                print("\nErrors:")
                for error in self.errors:
                    print(f"- {error[1]}")
            
            print("\nDetailed report saved to test_report.json")

    # Run the tests
    runner = TestResult(sys.stdout, unittest.TextTestResult.descriptions, 2)
    suite.run(runner)

if __name__ == '__main__':
    run_tests() 